//! Measure host Wasmtime compilation separately from guest compiler execution.
//!
//! This uses the command engine's fuel, shared-memory, threads, EH, GC, and async-stack
//! profile. It never grants a Wasm module imports or executes its code. Enable Winch for an
//! explicit comparison with `--features wasmtime/winch`; unsupported settings remain enabled
//! so a faster compiler cannot hide a missing runtime capability.

use sha2::{Digest, Sha256};
use std::time::Instant;
use wasmtime::{
    Collector, Config, Engine, Error, Module, ModuleVersionStrategy, OptLevel, Strategy,
};

fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let source_path = args.first().ok_or_else(|| Error::msg(
        "usage: wasmtime_compile FILE [cranelift|winch] [speed|none|speed_and_size] [parallel|serial] [full|minimal]"
    ))?;
    let compiler = args.get(1).map(String::as_str).unwrap_or("cranelift");
    let optimization = args.get(2).map(String::as_str).unwrap_or("speed");
    let parallel = args.get(3).map(String::as_str).unwrap_or("parallel");
    let metadata = args.get(4).map(String::as_str).unwrap_or("full");
    let mut config = Config::new();
    config
        .consume_fuel(true)
        .async_stack_size(2 * 1024 * 1024)
        .wasm_exceptions(true)
        .wasm_threads(true)
        .shared_memory(true)
        .wasm_gc(false)
        .collector(Collector::DeferredReferenceCounting)
        .strategy(match compiler {
            "cranelift" => Strategy::Cranelift,
            "winch" => Strategy::Winch,
            _ => return Err(Error::msg("unsupported compiler")),
        })
        .cranelift_opt_level(match optimization {
            "speed" => OptLevel::Speed,
            "none" => OptLevel::None,
            "speed_and_size" => OptLevel::SpeedAndSize,
            _ => return Err(Error::msg("unsupported optimization level")),
        })
        .parallel_compilation(match parallel {
            "parallel" => true,
            "serial" => false,
            _ => return Err(Error::msg("unsupported compilation parallelism")),
        });
    match metadata {
        "full" => {}
        "minimal" => {
            config.debug_symbols(false).generate_address_map(false);
        }
        _ => return Err(Error::msg("unsupported metadata mode")),
    }
    config.module_version(ModuleVersionStrategy::Custom(format!(
        "shellsim-wasmtime-49-{}",
        env!("SHELLSIM_WASMTIME_BUILD_ID")
    )))?;
    let engine = Engine::new(&config)?;
    let read_started = Instant::now();
    let source = std::fs::read(source_path)?;
    let read_seconds = read_started.elapsed().as_secs_f64();
    let compile_started = Instant::now();
    let module = Module::new(&engine, &source)?;
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    let image = module.image_range();
    println!(
        "{}",
        serde_json::json!({
            "compiler": compiler, "optimization": optimization, "parallel": parallel,
            "metadata": metadata, "source_bytes": source.len(),
            "source_sha256": format!("{:x}", Sha256::digest(&source)),
            "read_seconds": read_seconds, "compile_seconds": compile_seconds,
            "image_bytes": image.end as usize - image.start as usize,
            "debug_info": engine.get_debug_info(),
            "debug_symbols": engine.get_debug_symbols(),
            "address_map": engine.get_generate_address_map(),
        })
    );
    Ok(())
}
