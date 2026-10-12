"""Build the upstream SQLite amalgamation and its separately selected guest CLI."""

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    configure_args = []
    if ctx.variant == "library":
        targets = """
add_library(sqlite3 SHARED sqlite3.c)
add_library(sqlite3_static STATIC sqlite3.c)
set_target_properties(sqlite3_static PROPERTIES OUTPUT_NAME sqlite3)
foreach(target sqlite3 sqlite3_static)
  target_compile_definitions(${target} PRIVATE SQLITE_THREADSAFE=1)
endforeach()
install(TARGETS sqlite3 sqlite3_static LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES sqlite3.h sqlite3ext.h DESTINATION include)
"""
    elif ctx.variant == "cli":
        configure_args = [
            "-DSQLITE_LIBRARY=" + str(ctx.require_native().dependency_sysroot / "usr/local/lib/libsqlite3.so")
        ]
        targets = """
add_executable(sqlite3 shell.c)
target_compile_definitions(sqlite3 PRIVATE
  SQLITE_THREADSAFE=1 SQLITE_WASI=1 SQLITE_NOHAVE_SYSTEM=1
  _WASI_EMULATED_SIGNAL=1 _WASI_EMULATED_PROCESS_CLOCKS=1 _WASI_EMULATED_GETPID=1)
target_link_libraries(sqlite3 PRIVATE "${SQLITE_LIBRARY}"
  wasi-emulated-signal wasi-emulated-process-clocks wasi-emulated-getpid)
install(TARGETS sqlite3 RUNTIME DESTINATION bin)
"""
    else:
        raise ValueError("unsupported SQLite output variant")
    (ctx.source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\nproject(sqlite C)\n" + targets)
    return cmake(ctx, configure_args=configure_args, jobs=2)
