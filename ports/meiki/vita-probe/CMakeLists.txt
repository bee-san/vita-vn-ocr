cmake_minimum_required(VERSION 3.16)
set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "Vita toolchain")
project(meiki_vita_probe C CXX)
include("$ENV{VITASDK}/share/vita.cmake" REQUIRED)
set(MNN_SOURCE "" CACHE PATH "Pinned MNN source")
set(MNN_LIBRARY "" CACHE FILEPATH "MNN 3.2.0 static Vita library")
set(MEIKI_BUNDLE_DIR "" CACHE PATH "Private, verified manual-test bundle")
if(NOT EXISTS "${MNN_LIBRARY}" OR NOT EXISTS "${MNN_SOURCE}/include/MNN/Interpreter.hpp")
  message(FATAL_ERROR "Set MNN_SOURCE and MNN_LIBRARY to the pinned cross-build")
endif()
if(NOT EXISTS "${MEIKI_BUNDLE_DIR}/cases.tsv" OR NOT EXISTS "${MEIKI_BUNDLE_DIR}/meiki-stream-int8.mnn" OR NOT EXISTS "${MEIKI_BUNDLE_DIR}/font.bin")
  message(FATAL_ERROR "Set MEIKI_BUNDLE_DIR to the prepared manual-test bundle")
endif()
set(SCREEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../vita/app")
add_executable(meiki_probe main.cpp diagnostics.cpp ../runtime.cpp ../arena.c ../decode.c "${SCREEN_DIR}/screen.c")
target_include_directories(meiki_probe PRIVATE .. "${SCREEN_DIR}" "${MNN_SOURCE}/include")
target_compile_definitions(meiki_probe PRIVATE MEIKI_TRACE=1)
target_compile_options(meiki_probe PRIVATE -O2 -mfpu=neon -mfloat-abi=hard -ffp-contract=off -ffunction-sections -fdata-sections)
target_link_options(meiki_probe PRIVATE -Wl,--gc-sections
  # libstdc++ only references these pthread functions weakly. Static archives
  # do not otherwise extract them, breaking MNN's std::call_once on Vita.
  -Wl,-u,pthread_once -Wl,-u,pthread_cancel
  -Wl,-u,pthread_cond_wait -Wl,-u,pthread_cond_destroy -Wl,-u,pthread_cond_broadcast
  -Wl,--wrap=malloc -Wl,--wrap=__cxa_throw
  -Wl,--wrap=MNNMemoryAllocAlign -Wl,--wrap=MNNMemoryCallocAlign -Wl,--wrap=MNNMemoryFreeAlign)
target_link_libraries(meiki_probe "${MNN_LIBRARY}" pthread m
  SceSysmem_stub SceKernelThreadMgr_stub SceIofilemgr_stub SceLibKernel_stub SceProcessmgr_stub ScePower_stub SceDisplay_stub SceCtrl_stub)
vita_create_self(eboot.bin meiki_probe)
set(BUNDLE_FILES "")
file(GLOB DATA_FILES "${MEIKI_BUNDLE_DIR}/*.f32")
foreach(DATA_FILE IN LISTS DATA_FILES)
  get_filename_component(DATA_NAME "${DATA_FILE}" NAME)
  list(APPEND BUNDLE_FILES FILE "${DATA_FILE}" "data/${DATA_NAME}")
endforeach()
foreach(DATA_NAME cases.tsv meiki-stream-int8.mnn font.bin font-OFL.txt)
  list(APPEND BUNDLE_FILES FILE "${MEIKI_BUNDLE_DIR}/${DATA_NAME}" "data/${DATA_NAME}")
endforeach()
vita_create_vpk(meiki-probe.vpk MKOCR0001 eboot.bin NAME "Meiki OCR probe" VERSION "01.02" ${BUNDLE_FILES})
