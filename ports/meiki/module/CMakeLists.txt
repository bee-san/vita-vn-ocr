cmake_minimum_required(VERSION 3.16)
set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "Vita toolchain")
project(meiki_module C CXX)
include("$ENV{VITASDK}/share/vita.cmake" REQUIRED)
set(MNN_SOURCE "" CACHE PATH "Pinned MNN source")
set(MNN_LIBRARY "" CACHE FILEPATH "MNN 3.2.0 static Vita library with exceptions enabled")
if(NOT EXISTS "${MNN_LIBRARY}" OR NOT EXISTS "${MNN_SOURCE}/include/MNN/Interpreter.hpp")
  message(FATAL_ERROR "Set MNN_SOURCE and MNN_LIBRARY to the pinned cross-build")
endif()
execute_process(COMMAND "${CMAKE_C_COMPILER}" -print-file-name=crtbegin.o OUTPUT_VARIABLE CRT_BEGIN OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${CMAKE_C_COMPILER}" -print-file-name=crtend.o OUTPUT_VARIABLE CRT_END OUTPUT_STRIP_TRAILING_WHITESPACE)
add_executable(meiki_module "${CRT_BEGIN}" bridge.c metadata.c engine.cpp ../runtime.cpp ../detect_runtime.cpp ../arena.c "${CRT_END}")
target_include_directories(meiki_module PRIVATE .. "${MNN_SOURCE}/include")
target_compile_features(meiki_module PRIVATE c_std_99 cxx_std_17)
target_compile_options(meiki_module PRIVATE -O2 -mfpu=neon -mfloat-abi=hard -ffp-contract=off -fno-builtin -ffunction-sections -fdata-sections -Wall -Wextra)
target_link_options(meiki_module PRIVATE -nostartfiles -Wl,-q -Wl,-e,module_start -Wl,--gc-sections
  -Wl,-u,module_stop
  -Wl,-u,pthread_once -Wl,-u,pthread_cancel
  -Wl,-u,pthread_cond_wait -Wl,-u,pthread_cond_destroy -Wl,-u,pthread_cond_broadcast
  -Wl,--wrap=MNNMemoryAllocAlign -Wl,--wrap=MNNMemoryCallocAlign -Wl,--wrap=MNNMemoryFreeAlign)
target_link_libraries(meiki_module "${MNN_LIBRARY}" pthread m
  SceSysmem_stub SceKernelThreadMgr_stub SceIofilemgr_stub SceLibKernel_stub SceProcessmgr_stub)
vita_create_self(meiki-engine.suprx meiki_module UNSAFE CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/exports.yml")
