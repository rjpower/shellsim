//! Versioned raw Wasm function-table calls for the SDK34 FFI proof.
//!
//! C ABI lowering belongs to guest libffi. This boundary accepts only bounded
//! Wasm scalar slots and checks the table function's exact signature before a
//! guest-to-guest call. No value or pointer becomes a host address.

use super::{
    dynamic, fibers, memory, threaded_dynamic, Host, ERRNO_FAULT, ERRNO_INVAL, ERRNO_NOSPC,
    ERRNO_SUCCESS,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use wasmtime::{
    AsContextMut, Caller, Error, Func, FuncType, Global, Instance, Linker, Mutability, Ref,
    StoreContextMut, Val, ValType,
};

pub(super) const NAMESPACE: &str = "shellsim_ffi_v1";
const MAX_PARAMS: usize = 16;
pub(super) const MAX_CLOSURES: usize = 64;
pub(super) const CLOSURE_METADATA_BYTES: u64 = 4096;
const STACK_GUARD_BYTES: u32 = 64;

pub(super) fn supported(name: &str) -> bool {
    matches!(
        name,
        "invoke"
            | "closure_alloc"
            | "closure_alloc_typed"
            | "closure_reserve"
            | "closure_define_typed"
            | "closure_release"
    )
}

struct Closure {
    index: u32,
    active: Arc<AtomicBool>,
    prepared: bool,
}

#[derive(Default)]
pub(super) struct State {
    closures: Vec<Closure>,
}

#[derive(Clone, Copy)]
pub(super) enum Scalar {
    I32,
    I64,
    F32,
    F64,
}

impl Scalar {
    fn parse(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::I32),
            2 => Some(Self::I64),
            3 => Some(Self::F32),
            4 => Some(Self::F64),
            _ => None,
        }
    }

    fn ty(self) -> ValType {
        match self {
            Self::I32 => ValType::I32,
            Self::I64 => ValType::I64,
            Self::F32 => ValType::F32,
            Self::F64 => ValType::F64,
        }
    }

    fn value(self, bits: u64) -> Val {
        match self {
            Self::I32 => Val::I32(bits as i32),
            Self::I64 => Val::I64(bits as i64),
            Self::F32 => Val::F32(bits as u32),
            Self::F64 => Val::F64(bits),
        }
    }

    fn bits(self, value: &Val) -> Option<u64> {
        match (self, value) {
            (Self::I32, Val::I32(value)) => Some(*value as u32 as u64),
            (Self::I64, Val::I64(value)) => Some(*value as u64),
            (Self::F32, Val::F32(value)) => Some(*value as u64),
            (Self::F64, Val::F64(value)) => Some(*value),
            _ => None,
        }
    }
}

async fn synchronize(caller: &mut Caller<'_, Host>) -> Result<(), Error> {
    // Static threaded FFI is rejected by admission. During main instantiation
    // the validated v3 thread exists before its local Instance is available.
    if caller.data().thread.is_some() {
        threaded_dynamic::callbacks::synchronize(caller).await?;
    }
    Ok(())
}

fn main_instance(host: &Host) -> Option<Instance> {
    host.threaded_dynamic.main.or(host.dynamic.main)
}

/// Immutable callback metadata may cross Stores; Func handles may not.
#[derive(Clone)]
pub(super) enum CallbackSpec {
    Legacy {
        dispatch: u32,
        userdata: u32,
    },
    Typed {
        dispatch: u32,
        userdata: u32,
        tags: Arc<[Scalar]>,
        result: Option<Scalar>,
    },
}

pub(super) fn make_callback(
    mut store: StoreContextMut<'_, Host>,
    spec: &CallbackSpec,
    active: Arc<AtomicBool>,
) -> Func {
    match spec {
        CallbackSpec::Legacy { dispatch, userdata } => {
            let dispatch = *dispatch;
            let userdata = *userdata;
            let signature = FuncType::new(store.engine(), [ValType::I32], [ValType::I32]);
            Func::new_async(&mut store, signature, move |caller, params, results| {
                let active = active.clone();
                Box::new(async move {
                    let Val::I32(argument) = params[0] else {
                        return Err(Error::msg("FFI callback argument type changed"));
                    };
                    results[0] = Val::I32(
                        dispatch_callback(caller, dispatch, userdata, argument, active).await?,
                    );
                    Ok(())
                })
            })
        }
        CallbackSpec::Typed {
            dispatch,
            userdata,
            tags,
            result,
        } => make_typed_closure(store, *dispatch, *userdata, tags.clone(), *result, active),
    }
}

fn same_types(mut actual: impl Iterator<Item = ValType>, expected: &[ValType]) -> bool {
    expected.iter().all(|expected| {
        actual
            .next()
            .is_some_and(|actual| ValType::eq(&actual, expected))
    }) && actual.next().is_none()
}

fn stack_contract(caller: &mut Caller<'_, Host>) -> Result<(Global, u32, u32), Error> {
    let main =
        main_instance(caller.data()).ok_or_else(|| Error::msg("FFI callback runtime vanished"))?;
    let stack = main
        .get_global(&mut *caller, "__stack_pointer")
        .ok_or_else(|| Error::msg("FFI callback stack is unavailable"))?;
    if let Some((low, high)) = caller.data().threaded_dynamic.stack_bounds {
        let floor = low
            .checked_add(STACK_GUARD_BYTES)
            .ok_or_else(|| Error::msg("FFI callback stack floor is invalid"))?;
        if stack.ty(&*caller).mutability() != Mutability::Var
            || high <= floor
            || stack.get(&mut *caller).i32().is_none()
        {
            return Err(Error::msg("FFI callback stack contract is invalid"));
        }
        return Ok((stack, floor, high));
    }
    let stack_low = main
        .get_global(&mut *caller, "__stack_low")
        .ok_or_else(|| Error::msg("FFI callback stack floor is unavailable"))?;
    let stack_high = main
        .get_global(&mut *caller, "__stack_high")
        .ok_or_else(|| Error::msg("FFI callback stack ceiling is unavailable"))?;
    if stack.ty(&*caller).mutability() != Mutability::Var
        || stack_low.ty(&*caller).mutability() != Mutability::Const
        || stack_high.ty(&*caller).mutability() != Mutability::Const
    {
        return Err(Error::msg("FFI callback stack contract is invalid"));
    }
    let floor = stack_low
        .get(&mut *caller)
        .i32()
        .map(|value| value as u32)
        .and_then(|value| value.checked_add(STACK_GUARD_BYTES))
        .ok_or_else(|| Error::msg("FFI callback stack floor is invalid"))?;
    let high = stack_high
        .get(&mut *caller)
        .i32()
        .map(|value| value as u32)
        .filter(|value| *value > floor)
        .ok_or_else(|| Error::msg("FFI callback stack ceiling is invalid"))?;
    if stack.get(&mut *caller).i32().is_none() {
        return Err(Error::msg("FFI callback stack pointer is invalid"));
    }
    Ok((stack, floor, high))
}

async fn invoke(
    mut caller: Caller<'_, Host>,
    index: u32,
    tags_ptr: u32,
    values_ptr: u32,
    count: u32,
    result_tag: u32,
    result_ptr: u32,
) -> Result<i32, Error> {
    synchronize(&mut caller).await?;
    if count as usize > MAX_PARAMS || result_tag > 4 {
        return Ok(ERRNO_INVAL);
    }
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    let count = count as usize;
    let mut tags = [0u8; MAX_PARAMS];
    let mut value_bytes = [0u8; MAX_PARAMS * 8];
    if memory
        .read(&caller, tags_ptr as usize, &mut tags[..count])
        .is_err()
        || memory
            .read(&caller, values_ptr as usize, &mut value_bytes[..count * 8])
            .is_err()
    {
        return Ok(ERRNO_FAULT);
    }
    if result_tag != 0
        && (result_ptr as usize)
            .checked_add(8)
            .is_none_or(|end| end > memory.data_size(&caller))
    {
        return Ok(ERRNO_FAULT);
    }
    let mut params = Vec::with_capacity(count);
    let mut types = Vec::with_capacity(count);
    for (position, code) in tags[..count].iter().copied().enumerate() {
        let Some(tag) = Scalar::parse(code) else {
            return Ok(ERRNO_INVAL);
        };
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&value_bytes[position * 8..position * 8 + 8]);
        types.push(tag.ty());
        params.push(tag.value(u64::from_le_bytes(bytes)));
    }
    let result = if result_tag == 0 {
        None
    } else {
        let Some(tag) = Scalar::parse(result_tag as u8) else {
            return Ok(ERRNO_INVAL);
        };
        Some(tag)
    };
    let Some(main) = main_instance(caller.data()) else {
        return Ok(ERRNO_INVAL);
    };
    let Some(table) = main.get_table(&mut caller, "__indirect_function_table") else {
        return Ok(ERRNO_INVAL);
    };
    let Some(Ref::Func(Some(function))) = table.get(&mut caller, index as u64) else {
        return Ok(ERRNO_INVAL);
    };
    let signature = function.ty(&caller);
    let results = result.map_or_else(Vec::new, |tag| vec![tag.ty()]);
    if !same_types(signature.params(), &types) || !same_types(signature.results(), &results) {
        return Ok(ERRNO_INVAL);
    }
    if !caller
        .data()
        .machine
        .get()
        .resources
        .charge_cpu(count as u64 + 4)
    {
        return Err(super::exhausted());
    }
    let mut output = result.map_or_else(Vec::new, |tag| vec![tag.value(0)]);
    let _fiber = fibers::begin(&mut caller)?;
    function
        .call_async(&mut caller, &params, &mut output)
        .await?;
    if let Some(tag) = result {
        let bits = tag
            .bits(&output[0])
            .ok_or_else(|| Error::msg("FFI result type changed during call"))?;
        memory.write(&mut caller, result_ptr as usize, &bits.to_le_bytes())?;
    }
    Ok(ERRNO_SUCCESS)
}

async fn dispatch_callback(
    mut caller: Caller<'_, Host>,
    dispatch_index: u32,
    userdata: u32,
    argument: i32,
    active: Arc<AtomicBool>,
) -> Result<i32, Error> {
    if !active.load(Ordering::Acquire) {
        return Err(Error::msg("released FFI callback"));
    }
    synchronize(&mut caller).await?;
    if !active.load(Ordering::Acquire) {
        return Err(Error::msg("released FFI callback"));
    }
    if !caller.data().machine.get().resources.charge_cpu(8) {
        return Err(super::exhausted());
    }
    let main =
        main_instance(caller.data()).ok_or_else(|| Error::msg("FFI callback runtime vanished"))?;
    let table = main
        .get_table(&mut caller, "__indirect_function_table")
        .ok_or_else(|| Error::msg("FFI callback table vanished"))?;
    let Some(Ref::Func(Some(function))) = table.get(&mut caller, dispatch_index as u64) else {
        return Err(Error::msg("FFI callback dispatcher vanished"));
    };
    let signature = function.ty(&caller);
    if !same_types(signature.params(), &[ValType::I32, ValType::I32])
        || !same_types(signature.results(), &[ValType::I32])
    {
        return Err(Error::msg("FFI callback dispatcher signature changed"));
    }
    let mut output = [Val::I32(0)];
    let _fiber = fibers::begin(&mut caller)?;
    function
        .call_async(
            &mut caller,
            &[Val::I32(userdata as i32), Val::I32(argument)],
            &mut output,
        )
        .await?;
    let Val::I32(value) = output[0] else {
        return Err(Error::msg("FFI callback result type changed"));
    };
    Ok(value)
}

async fn dispatch_callback_typed(
    mut caller: Caller<'_, Host>,
    dispatch_index: u32,
    userdata: u32,
    params: &[Val],
    tags: &[Scalar],
    result_tag: Option<Scalar>,
    active: Arc<AtomicBool>,
) -> Result<Option<Val>, Error> {
    if !active.load(Ordering::Acquire) {
        return Err(Error::msg("released FFI callback"));
    }
    synchronize(&mut caller).await?;
    if !active.load(Ordering::Acquire) {
        return Err(Error::msg("released FFI callback"));
    }
    if !caller
        .data()
        .machine
        .get()
        .resources
        .charge_cpu(params.len() as u64 + 8)
    {
        return Err(super::exhausted());
    }
    let main =
        main_instance(caller.data()).ok_or_else(|| Error::msg("FFI callback runtime vanished"))?;
    let table = main
        .get_table(&mut caller, "__indirect_function_table")
        .ok_or_else(|| Error::msg("FFI callback table vanished"))?;
    let Some(Ref::Func(Some(function))) = table.get(&mut caller, dispatch_index as u64) else {
        return Err(Error::msg("FFI callback dispatcher vanished"));
    };
    let dispatcher = function.typed::<(u32, u32, u32), i32>(&caller)?;
    let (stack, floor, high) = stack_contract(&mut caller)?;
    let memory =
        memory(&mut caller).ok_or_else(|| Error::msg("FFI callback memory is unavailable"))?;
    let old_sp = stack
        .get(&mut caller)
        .i32()
        .ok_or_else(|| Error::msg("FFI callback stack pointer is invalid"))?
        as u32;
    let frame_size = u32::try_from((tags.len() + 1) * 8)?;
    let new_sp = old_sp
        .checked_sub(frame_size)
        .map(|pointer| pointer & !15)
        .filter(|pointer| *pointer >= floor)
        .ok_or_else(|| Error::msg("FFI callback C stack exhausted"))?;
    if old_sp > high {
        return Err(Error::msg(
            "FFI callback C stack pointer is outside linker bounds",
        ));
    }
    if high as usize > memory.data_size(&caller)
        || new_sp
            .checked_add(frame_size)
            .is_none_or(|end| end as usize > memory.data_size(&caller))
    {
        return Err(Error::msg("FFI callback stack frame exceeds memory"));
    }
    let mut slots = [0u8; MAX_PARAMS * 8];
    for (position, (tag, value)) in tags.iter().zip(params).enumerate() {
        let bits = tag
            .bits(value)
            .ok_or_else(|| Error::msg("FFI callback argument type changed"))?;
        slots[position * 8..position * 8 + 8].copy_from_slice(&bits.to_le_bytes());
    }
    let _fiber = fibers::begin(&mut caller)?;
    let call = async {
        stack.set(&mut caller, Val::I32(new_sp as i32))?;
        memory.write(&mut caller, new_sp as usize, &slots[..tags.len() * 8])?;
        let result_ptr = new_sp + u32::try_from(tags.len() * 8)?;
        memory.write(&mut caller, result_ptr as usize, &[0; 8])?;
        let status = dispatcher
            .call_async(&mut caller, (userdata, new_sp, result_ptr))
            .await?;
        if status != 0 {
            return Err(Error::msg("FFI callback dispatcher failed"));
        }
        if let Some(tag) = result_tag {
            let mut bytes = [0; 8];
            memory.read(&caller, result_ptr as usize, &mut bytes)?;
            Ok(Some(tag.value(u64::from_le_bytes(bytes))))
        } else {
            Ok(None)
        }
    }
    .await;
    stack.set(&mut caller, Val::I32(old_sp as i32))?;
    call
}

async fn closure_alloc(
    mut caller: Caller<'_, Host>,
    dispatch_index: u32,
    userdata: u32,
    output_ptr: u32,
) -> Result<i32, Error> {
    synchronize(&mut caller).await?;
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    if (output_ptr as usize)
        .checked_add(4)
        .is_none_or(|end| end > memory.data_size(&caller))
    {
        return Ok(ERRNO_FAULT);
    }
    if caller.data().ffi.closures.len() >= MAX_CLOSURES {
        return Ok(ERRNO_NOSPC);
    }
    let Some(main) = main_instance(caller.data()) else {
        return Ok(ERRNO_INVAL);
    };
    let Some(table) = main.get_table(&mut caller, "__indirect_function_table") else {
        return Ok(ERRNO_INVAL);
    };
    let Some(Ref::Func(Some(dispatcher))) = table.get(&mut caller, dispatch_index as u64) else {
        return Ok(ERRNO_INVAL);
    };
    let signature = dispatcher.ty(&caller);
    if !same_types(signature.params(), &[ValType::I32, ValType::I32])
        || !same_types(signature.results(), &[ValType::I32])
        || table.size(&caller) > u32::MAX as u64
    {
        return Ok(ERRNO_INVAL);
    }
    if !caller.data().machine.get().resources.charge_cpu(16) {
        return Err(super::exhausted());
    }
    if caller.data().threaded_dynamic.main.is_some() {
        let Some(slot) = threaded_dynamic::callbacks::reserve(&mut caller).await? else {
            return Ok(ERRNO_NOSPC);
        };
        let spec = CallbackSpec::Legacy {
            dispatch: dispatch_index,
            userdata,
        };
        if !threaded_dynamic::callbacks::define(&mut caller, slot, spec).await? {
            return Ok(ERRNO_INVAL);
        }
        memory.write(&mut caller, output_ptr as usize, &slot.to_le_bytes())?;
        return Ok(ERRNO_SUCCESS);
    }
    let reservation = CLOSURE_METADATA_BYTES;
    dynamic::reserve(&mut caller, reservation)?;
    let slot = match table.grow(&mut caller, 1, Ref::Func(None)) {
        Ok(slot) => slot,
        Err(_) => {
            dynamic::unreserve(&mut caller, reservation);
            return if caller.data().machine.get().resources.is_stopped() {
                Err(super::exhausted())
            } else {
                Ok(ERRNO_NOSPC)
            };
        }
    };
    let active = Arc::new(AtomicBool::new(true));
    let closure_active = active.clone();
    let callback_type = FuncType::new(caller.engine(), [ValType::I32], [ValType::I32]);
    let function = Func::new_async(
        &mut caller,
        callback_type,
        move |caller, params, results| {
            let active = closure_active.clone();
            Box::new(async move {
                let Val::I32(argument) = params[0] else {
                    return Err(Error::msg("FFI callback argument type changed"));
                };
                results[0] = Val::I32(
                    dispatch_callback(caller, dispatch_index, userdata, argument, active).await?,
                );
                Ok(())
            })
        },
    );
    table.set(&mut caller, slot, Ref::Func(Some(function)))?;
    caller.data_mut().ffi.closures.push(Closure {
        index: slot as u32,
        active,
        prepared: true,
    });
    memory.write(
        &mut caller,
        output_ptr as usize,
        &(slot as u32).to_le_bytes(),
    )?;
    Ok(ERRNO_SUCCESS)
}

async fn closure_alloc_typed(
    mut caller: Caller<'_, Host>,
    dispatch_index: u32,
    userdata: u32,
    tags_ptr: u32,
    count: u32,
    result_tag: u32,
    output_ptr: u32,
) -> Result<i32, Error> {
    synchronize(&mut caller).await?;
    if count as usize > MAX_PARAMS || result_tag > 4 {
        return Ok(ERRNO_INVAL);
    }
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    if (output_ptr as usize)
        .checked_add(4)
        .is_none_or(|end| end > memory.data_size(&caller))
    {
        return Ok(ERRNO_FAULT);
    }
    let mut codes = [0; MAX_PARAMS];
    if memory
        .read(&caller, tags_ptr as usize, &mut codes[..count as usize])
        .is_err()
    {
        return Ok(ERRNO_FAULT);
    }
    let mut tags = Vec::with_capacity(count as usize);
    for code in &codes[..count as usize] {
        let Some(tag) = Scalar::parse(*code) else {
            return Ok(ERRNO_INVAL);
        };
        tags.push(tag);
    }
    let result = if result_tag == 0 {
        None
    } else {
        Some(Scalar::parse(result_tag as u8).expect("bounded scalar tag"))
    };
    if caller.data().ffi.closures.len() >= MAX_CLOSURES {
        return Ok(ERRNO_NOSPC);
    }
    let Some(main) = main_instance(caller.data()) else {
        return Ok(ERRNO_INVAL);
    };
    let Some(table) = main.get_table(&mut caller, "__indirect_function_table") else {
        return Ok(ERRNO_INVAL);
    };
    let Some(Ref::Func(Some(dispatcher))) = table.get(&mut caller, dispatch_index as u64) else {
        return Ok(ERRNO_INVAL);
    };
    let signature = dispatcher.ty(&caller);
    if !same_types(
        signature.params(),
        &[ValType::I32, ValType::I32, ValType::I32],
    ) || !same_types(signature.results(), &[ValType::I32])
        || table.size(&caller) > u32::MAX as u64
        || stack_contract(&mut caller).is_err()
    {
        return Ok(ERRNO_INVAL);
    }
    if !caller
        .data()
        .machine
        .get()
        .resources
        .charge_cpu(count as u64 + 16)
    {
        return Err(super::exhausted());
    }
    if caller.data().threaded_dynamic.main.is_some() {
        let Some(slot) = threaded_dynamic::callbacks::reserve(&mut caller).await? else {
            return Ok(ERRNO_NOSPC);
        };
        let spec = CallbackSpec::Typed {
            dispatch: dispatch_index,
            userdata,
            tags: tags.into(),
            result,
        };
        if !threaded_dynamic::callbacks::define(&mut caller, slot, spec).await? {
            return Ok(ERRNO_INVAL);
        }
        memory.write(&mut caller, output_ptr as usize, &slot.to_le_bytes())?;
        return Ok(ERRNO_SUCCESS);
    }
    let reservation = CLOSURE_METADATA_BYTES;
    dynamic::reserve(&mut caller, reservation)?;
    let slot = match table.grow(&mut caller, 1, Ref::Func(None)) {
        Ok(slot) => slot,
        Err(_) => {
            dynamic::unreserve(&mut caller, reservation);
            return if caller.data().machine.get().resources.is_stopped() {
                Err(super::exhausted())
            } else {
                Ok(ERRNO_NOSPC)
            };
        }
    };
    let active = Arc::new(AtomicBool::new(true));
    let function = make_typed_closure(
        caller.as_context_mut(),
        dispatch_index,
        userdata,
        tags.into(),
        result,
        active.clone(),
    );
    table.set(&mut caller, slot, Ref::Func(Some(function)))?;
    caller.data_mut().ffi.closures.push(Closure {
        index: slot as u32,
        active,
        prepared: true,
    });
    memory.write(
        &mut caller,
        output_ptr as usize,
        &(slot as u32).to_le_bytes(),
    )?;
    Ok(ERRNO_SUCCESS)
}

fn make_typed_closure(
    mut caller: StoreContextMut<'_, Host>,
    dispatch_index: u32,
    userdata: u32,
    tags: Arc<[Scalar]>,
    result: Option<Scalar>,
    active: Arc<AtomicBool>,
) -> Func {
    let signature = FuncType::new(
        caller.engine(),
        tags.iter().map(|tag| tag.ty()),
        result.iter().map(|tag| tag.ty()),
    );
    Func::new_async(&mut caller, signature, move |caller, params, results| {
        let active = active.clone();
        let tags = tags.clone();
        Box::new(async move {
            let value = dispatch_callback_typed(
                caller,
                dispatch_index,
                userdata,
                params,
                &tags,
                result,
                active,
            )
            .await?;
            if let Some(value) = value {
                results[0] = value;
            }
            Ok(())
        })
    })
}

async fn closure_reserve(mut caller: Caller<'_, Host>, output_ptr: u32) -> Result<i32, Error> {
    synchronize(&mut caller).await?;
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    if (output_ptr as usize)
        .checked_add(4)
        .is_none_or(|end| end > memory.data_size(&caller))
    {
        return Ok(ERRNO_FAULT);
    }
    if caller.data().ffi.closures.len() >= MAX_CLOSURES {
        return Ok(ERRNO_NOSPC);
    }
    let Some(main) = main_instance(caller.data()) else {
        return Ok(ERRNO_INVAL);
    };
    let Some(table) = main.get_table(&mut caller, "__indirect_function_table") else {
        return Ok(ERRNO_INVAL);
    };
    if table.size(&caller) == 0 || table.size(&caller) > u32::MAX as u64 {
        return Ok(ERRNO_INVAL);
    }
    if !caller.data().machine.get().resources.charge_cpu(8) {
        return Err(super::exhausted());
    }
    if caller.data().threaded_dynamic.main.is_some() {
        let Some(slot) = threaded_dynamic::callbacks::reserve(&mut caller).await? else {
            return Ok(ERRNO_NOSPC);
        };
        memory.write(&mut caller, output_ptr as usize, &slot.to_le_bytes())?;
        return Ok(ERRNO_SUCCESS);
    }
    dynamic::reserve(&mut caller, CLOSURE_METADATA_BYTES)?;
    let slot = match table.grow(&mut caller, 1, Ref::Func(None)) {
        Ok(slot) => slot,
        Err(_) => {
            dynamic::unreserve(&mut caller, CLOSURE_METADATA_BYTES);
            return if caller.data().machine.get().resources.is_stopped() {
                Err(super::exhausted())
            } else {
                Ok(ERRNO_NOSPC)
            };
        }
    };
    caller.data_mut().ffi.closures.push(Closure {
        index: slot as u32,
        active: Arc::new(AtomicBool::new(true)),
        prepared: false,
    });
    memory.write(
        &mut caller,
        output_ptr as usize,
        &(slot as u32).to_le_bytes(),
    )?;
    Ok(ERRNO_SUCCESS)
}

async fn closure_define_typed(
    mut caller: Caller<'_, Host>,
    index: u32,
    dispatch_index: u32,
    userdata: u32,
    tags_ptr: u32,
    count: u32,
    result_tag: u32,
) -> Result<i32, Error> {
    synchronize(&mut caller).await?;
    if count as usize > MAX_PARAMS || result_tag > 4 {
        return Ok(ERRNO_INVAL);
    }
    let Some(memory) = memory(&mut caller) else {
        return Ok(ERRNO_FAULT);
    };
    let mut codes = [0; MAX_PARAMS];
    if memory
        .read(&caller, tags_ptr as usize, &mut codes[..count as usize])
        .is_err()
    {
        return Ok(ERRNO_FAULT);
    }
    let mut tags = Vec::with_capacity(count as usize);
    for code in &codes[..count as usize] {
        let Some(tag) = Scalar::parse(*code) else {
            return Ok(ERRNO_INVAL);
        };
        tags.push(tag);
    }
    let result = if result_tag == 0 {
        None
    } else {
        Some(Scalar::parse(result_tag as u8).expect("bounded scalar tag"))
    };
    let Some(main) = main_instance(caller.data()) else {
        return Ok(ERRNO_INVAL);
    };
    let Some(table) = main.get_table(&mut caller, "__indirect_function_table") else {
        return Ok(ERRNO_INVAL);
    };
    if !matches!(table.get(&mut caller, index as u64), Some(Ref::Func(None))) {
        return Ok(ERRNO_INVAL);
    }
    let Some(Ref::Func(Some(dispatcher))) = table.get(&mut caller, dispatch_index as u64) else {
        return Ok(ERRNO_INVAL);
    };
    let signature = dispatcher.ty(&caller);
    if !same_types(
        signature.params(),
        &[ValType::I32, ValType::I32, ValType::I32],
    ) || !same_types(signature.results(), &[ValType::I32])
        || stack_contract(&mut caller).is_err()
    {
        return Ok(ERRNO_INVAL);
    }
    if !caller
        .data()
        .machine
        .get()
        .resources
        .charge_cpu(count as u64 + 12)
    {
        return Err(super::exhausted());
    }
    if caller.data().threaded_dynamic.main.is_some() {
        let spec = CallbackSpec::Typed {
            dispatch: dispatch_index,
            userdata,
            tags: tags.into(),
            result,
        };
        return Ok(
            if threaded_dynamic::callbacks::define(&mut caller, index, spec).await? {
                ERRNO_SUCCESS
            } else {
                ERRNO_INVAL
            },
        );
    }
    let Some(position) = caller.data().ffi.closures.iter().position(|closure| {
        closure.index == index && closure.active.load(Ordering::Acquire) && !closure.prepared
    }) else {
        return Ok(ERRNO_INVAL);
    };
    let active = caller.data().ffi.closures[position].active.clone();
    let function = make_typed_closure(
        caller.as_context_mut(),
        dispatch_index,
        userdata,
        tags.into(),
        result,
        active,
    );
    table.set(&mut caller, index as u64, Ref::Func(Some(function)))?;
    caller.data_mut().ffi.closures[position].prepared = true;
    Ok(ERRNO_SUCCESS)
}

async fn closure_release(mut caller: Caller<'_, Host>, index: u32) -> Result<i32, Error> {
    synchronize(&mut caller).await?;
    if caller.data().threaded_dynamic.main.is_some() {
        return Ok(
            if threaded_dynamic::callbacks::release(&mut caller, index).await? {
                ERRNO_SUCCESS
            } else {
                ERRNO_INVAL
            },
        );
    }
    let Some(position) = caller
        .data()
        .ffi
        .closures
        .iter()
        .position(|closure| closure.index == index && closure.active.load(Ordering::Acquire))
    else {
        return Ok(ERRNO_INVAL);
    };
    let Some(main) = main_instance(caller.data()) else {
        return Ok(ERRNO_INVAL);
    };
    let Some(table) = main.get_table(&mut caller, "__indirect_function_table") else {
        return Ok(ERRNO_INVAL);
    };
    if !caller.data().machine.get().resources.charge_cpu(4) {
        return Err(super::exhausted());
    }
    table.set(&mut caller, index as u64, Ref::Func(None))?;
    caller.data().ffi.closures[position]
        .active
        .store(false, Ordering::Release);
    Ok(ERRNO_SUCCESS)
}

pub(super) fn register(linker: &mut Linker<Host>) {
    linker
        .func_wrap_async(
            NAMESPACE,
            "invoke",
            |caller: Caller<'_, Host>, (index, tags, values, count, result_tag, result): (u32, u32, u32, u32, u32, u32)| {
                Box::new(invoke(caller, index, tags, values, count, result_tag, result))
            },
        )
        .expect("unique FFI import");
    linker
        .func_wrap_async(
            NAMESPACE,
            "closure_alloc",
            |caller: Caller<'_, Host>, (dispatch, userdata, output): (u32, u32, u32)| {
                Box::new(closure_alloc(caller, dispatch, userdata, output))
            },
        )
        .expect("unique FFI import");
    linker
        .func_wrap_async(
            NAMESPACE,
            "closure_alloc_typed",
            |caller: Caller<'_, Host>,
             (dispatch, userdata, tags, count, result_tag, output): (
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
            )| {
                Box::new(closure_alloc_typed(
                    caller, dispatch, userdata, tags, count, result_tag, output,
                ))
            },
        )
        .expect("unique FFI import");
    linker
        .func_wrap_async(NAMESPACE, "closure_reserve", |caller, (output,): (u32,)| {
            Box::new(closure_reserve(caller, output))
        })
        .expect("unique FFI import");
    linker
        .func_wrap_async(
            NAMESPACE,
            "closure_define_typed",
            |caller,
             (index, dispatch, userdata, tags, count, result_tag): (
                u32,
                u32,
                u32,
                u32,
                u32,
                u32,
            )| {
                Box::new(closure_define_typed(
                    caller, index, dispatch, userdata, tags, count, result_tag,
                ))
            },
        )
        .expect("unique FFI import");
    linker
        .func_wrap_async(NAMESPACE, "closure_release", |caller, (index,): (u32,)| {
            Box::new(closure_release(caller, index))
        })
        .expect("unique FFI import");
}
