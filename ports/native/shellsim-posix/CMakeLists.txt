cmake_minimum_required(VERSION 3.20)
project(shellsim_posix LANGUAGES C)
add_library(shellsim-posix STATIC process.c exec.c tempfile.c accounts.c python/cpython/threaded_posix.c)
target_include_directories(shellsim-posix PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_definitions(shellsim-posix PUBLIC _WASI_EMULATED_SIGNAL=1)
target_compile_options(shellsim-posix PRIVATE -Wall -Wextra -Werror
  -include ${CMAKE_CURRENT_SOURCE_DIR}/include/process_abi.h)
install(TARGETS shellsim-posix ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(FILES shellsim-posix.pc DESTINATION lib/pkgconfig)
install(FILES LICENSE DESTINATION share/licenses/shellsim-posix)
