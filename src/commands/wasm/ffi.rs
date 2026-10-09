//! Versioned raw Wasm function-table calls for the SDK34 FFI proof.
//!
//! C ABI lowering belongs to guest libffi. This boundary accepts only bounded
//! Wasm scalar slots and checks the table function's exact signature before a
//! guest-to-guest call. No value or pointer becomes a host address.

use super::{dynamic, fibers, memory, Host, ERRNO_FAULT, ERRNO_INVAL, ERRNO_NOSPC, ERRNO_SUCCESS};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use wasmtime::{Caller, Error, Func, FuncType, Linker, Ref, Val, ValType};

pub(super) const NAMESPACE: &str = "shellsim_ffi_v1";
const MAX_PARAMS: usize = 16;
const MAX_CLOSURES: usize = 64;
const CLOSURE_METADATA_BYTES: u64 = 4096;

pub(super) fn supported(name: &str) -> bool {
    matches!(name, "invoke" | "closure_alloc" | "closure_release")
}

struct Closure {
    index: u32,
    active: Arc<AtomicBool>,
}

#[derive(Default)]
pub(super) struct State {
    closures: Vec<Closure>,
}

#[derive(Clone, Copy)]
enum Scalar {
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

fn same_types(mut actual: impl Iterator<Item = ValType>, expected: &[ValType]) -> bool {
    expected.iter().all(|expected| {
        actual
            .next()
            .is_some_and(|actual| ValType::eq(&actual, expected))
    }) && actual.next().is_none()
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
    let Some(main) = caller.data().dynamic.main else {
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
    if !caller.data().machine.get().resources.charge_cpu(8) {
        return Err(super::exhausted());
    }
    let main = caller
        .data()
        .dynamic
        .main
        .ok_or_else(|| Error::msg("FFI callback runtime vanished"))?;
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

async fn closure_alloc(
    mut caller: Caller<'_, Host>,
    dispatch_index: u32,
    userdata: u32,
    output_ptr: u32,
) -> Result<i32, Error> {
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
    let Some(main) = caller.data().dynamic.main else {
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
    });
    memory.write(
        &mut caller,
        output_ptr as usize,
        &(slot as u32).to_le_bytes(),
    )?;
    Ok(ERRNO_SUCCESS)
}

fn closure_release(mut caller: Caller<'_, Host>, index: u32) -> Result<i32, Error> {
    let Some(position) = caller
        .data()
        .ffi
        .closures
        .iter()
        .position(|closure| closure.index == index && closure.active.load(Ordering::Acquire))
    else {
        return Ok(ERRNO_INVAL);
    };
    let Some(main) = caller.data().dynamic.main else {
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
        .func_wrap(NAMESPACE, "closure_release", closure_release)
        .expect("unique FFI import");
}
