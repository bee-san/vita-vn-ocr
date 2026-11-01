# vita/CMakeLists.txt - PS Vita (VitaSDK) build of runtime/ and the vocr_test homebrew app.
#
#   vita/scripts/build.sh            the usual entry point (needs $VITASDK: scripts/fetch_vitasdk.sh)
#   cmake -S vita -B build/vita -DCMAKE_TOOLCHAIN_FILE=$VITASDK/share/vita.toolchain.cmake
#   cmake --build build/vita
#
# Builds, with arm-vita-eabi-gcc:
#   libvocr.a       runtime/vocr.c + runtime/vocr_lm.c + runtime/vocr_lines.c (+ vocr_lines_rows.c, the vertical
#                   column finder vocr_vertical.c, the counter post-processing vocr_post.c and the edge continuation
#                   vocr_lines_widen.c, at -O2 with the strict warnings), NEON, the runtime's own -Werror flags
#   vocr_test       the test app (app/main.c, app/screen.c) -> eboot.bin -> vocr_test.vpk
#   vocr_par        the runtime's host drivers (runtime/tools/vocr_cli.c, runtime/vocr_textspot6_test.c) as an app
#                   (par/main.c) -> par_eboot.bin -> vocr_par.vpk, for vita/vita3k_parity.py's check in Vita3K
# Nothing built here is committed: the VPK is a build product (vita/README.md).
cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VITASDK})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "toolchain file")
  else()
    message(FATAL_ERROR "VITASDK is not set: run vita/scripts/fetch_vitasdk.sh and export what it prints")
  endif()
endif()

project(vocr_vita C)
include("${VITASDK}/share/vita.cmake" REQUIRED)

set(VITA_APP_NAME "vita-vn-ocr test")
set(VITA_TITLEID "VVOC00001")
set(VITA_VERSION "01.00")

# VOCR_RUNTIME_DIR: the runtime/ to build (default this checkout's; vita/vita3k_parity.py builds another one, e.g.
# main's, for its before/after check)
set(VOCR_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../runtime CACHE PATH "runtime/ directory to build")
set(RT ${VOCR_RUNTIME_DIR})
# runtime/README.md "Porting notes": vocr.c alone, C99, -O2, no FP contraction; Cortex-A9 NEON.
set(VOCR_ARCH -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard)
set(VOCR_WARN -std=c99 -ffp-contract=off -Wall -Wextra -Werror -Wpedantic -Wshadow
    -Wstrict-prototypes -Wmissing-prototypes -Wcast-align -Wpointer-arith)

add_library(vocr STATIC ${RT}/vocr.c ${RT}/vocr_lm.c ${RT}/vocr_lines.c ${RT}/vocr_lines_rows.c ${RT}/vocr_vertical.c
  ${RT}/vocr_post.c ${RT}/vocr_lines_widen.c)
target_compile_options(vocr PRIVATE ${VOCR_ARCH} -O2 ${VOCR_WARN})
# runtime/vocr_lines.mk compiles the line finder with these on top
set_source_files_properties(${RT}/vocr_lines.c ${RT}/vocr_lines_rows.c ${RT}/vocr_vertical.c ${RT}/vocr_post.c
  ${RT}/vocr_lines_widen.c PROPERTIES COMPILE_OPTIONS "-Wconversion;-Wsign-conversion")

execute_process(COMMAND git rev-parse --short=12 HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE VOCR_GIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE git_rc)
if(NOT git_rc EQUAL 0 OR VOCR_GIT STREQUAL "")
  set(VOCR_GIT "unknown")
else()
  execute_process(COMMAND git diff --quiet HEAD -- ../runtime . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE dirty)
  if(NOT dirty EQUAL 0)
    set(VOCR_GIT "${VOCR_GIT}-dirty")
  endif()
endif()

add_executable(vocr_test app/main.c app/screen.c app/cases.c)
target_include_directories(vocr_test PRIVATE ${RT})
target_compile_options(vocr_test PRIVATE ${VOCR_ARCH} -O2 -std=gnu99 -Wall -Wextra -Werror)
target_compile_definitions(vocr_test PRIVATE VOCR_GIT="${VOCR_GIT}")
target_link_libraries(vocr_test
  vocr m
  SceDisplay_stub SceCtrl_stub ScePower_stub SceSysmem_stub SceKernelThreadMgr_stub
  SceIofilemgr_stub SceLibKernel_stub SceProcessmgr_stub
  ScePerf_stub_weak)   # weak: the app still starts where ScePerf is missing

vita_create_self(eboot.bin vocr_test)

# LiveArea images are generated (no image files in git, tests/test_repo_policy.py)
find_program(PYTHON3 NAMES python3 python REQUIRED NO_CMAKE_FIND_ROOT_PATH)
set(LA ${CMAKE_CURRENT_BINARY_DIR}/livearea)
add_custom_command(OUTPUT ${LA}/icon0.png ${LA}/bg.png ${LA}/startup.png
                   COMMAND ${PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/mkicon.py ${LA}
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mkicon.py VERBATIM)
add_custom_target(livearea DEPENDS ${LA}/icon0.png ${LA}/bg.png ${LA}/startup.png)

vita_create_vpk(vocr_test.vpk ${VITA_TITLEID} eboot.bin
  VERSION ${VITA_VERSION}
  NAME "${VITA_APP_NAME}"
  FILE ${LA}/icon0.png sce_sys/icon0.png
  FILE ${LA}/bg.png sce_sys/livearea/contents/bg.png
  FILE ${LA}/startup.png sce_sys/livearea/contents/startup.png
  FILE ${CMAKE_CURRENT_SOURCE_DIR}/sce_sys/livearea/contents/template.xml sce_sys/livearea/contents/template.xml)
add_dependencies(vocr_test.vpk-vpk livearea)

# vita/par (vita/vita3k_parity.py): the runtime's own host drivers run on the Vita's machine code, so that their outputs
# can be compared byte for byte with the host's. The recognizer, its decoder and post-processing, the line finder's
# rows and the text spotter as Rubi builds it (-DVOCR_TEXTSPOT6_NO_MS, a trunk blob only), at -O2 with the runtime's
# flags; the drivers' main functions renamed.
add_library(vocr_par_rt STATIC ${RT}/vocr.c ${RT}/vocr_lm.c ${RT}/vocr_post.c ${RT}/vocr_lines.c ${RT}/vocr_lines_rows.c
  ${RT}/vocr_textspot6.c)
target_compile_options(vocr_par_rt PRIVATE ${VOCR_ARCH} -O2 ${VOCR_WARN})
target_compile_definitions(vocr_par_rt PUBLIC VOCR_TEXTSPOT6_NO_MS=1)
# the drivers are always this checkout's (they read the runtime only through its headers, which a VOCR_RUNTIME_DIR to
# compare must share), so that both builds of a comparison run the same driver code
set(VOCR_DRIVERS ${CMAKE_CURRENT_SOURCE_DIR}/../runtime)
add_executable(vocr_par par/main.c ${VOCR_DRIVERS}/tools/vocr_cli.c ${VOCR_DRIVERS}/vocr_textspot6_test.c)
target_include_directories(vocr_par PRIVATE ${RT})
target_compile_options(vocr_par PRIVATE ${VOCR_ARCH} -O2 -std=gnu99 -Wall -Wextra -Werror)
set_source_files_properties(${VOCR_DRIVERS}/tools/vocr_cli.c PROPERTIES COMPILE_DEFINITIONS "main=vocr_cli_main")
set_source_files_properties(${VOCR_DRIVERS}/vocr_textspot6_test.c PROPERTIES COMPILE_DEFINITIONS "main=ts6_main")
target_link_libraries(vocr_par vocr_par_rt m SceSysmem_stub SceKernelThreadMgr_stub SceIofilemgr_stub SceLibKernel_stub
  SceProcessmgr_stub)
# vita.cmake's vita_create_self and vita_create_vpk append their flags to these variables: without a reset the second
# call would pass the first one's again
set(VITA_ELF_CREATE_FLAGS "")
set(VITA_MAKE_FSELF_FLAGS "")
set(VITA_MKSFOEX_FLAGS "")
set(VITA_PACK_VPK_FLAGS "")
vita_create_self(par_eboot.bin vocr_par)
vita_create_vpk(vocr_par.vpk VVOC00002 par_eboot.bin
  VERSION ${VITA_VERSION}
  NAME "vita-vn-ocr parity"
  FILE ${LA}/icon0.png sce_sys/icon0.png
  FILE ${LA}/bg.png sce_sys/livearea/contents/bg.png
  FILE ${LA}/startup.png sce_sys/livearea/contents/startup.png
  FILE ${CMAKE_CURRENT_SOURCE_DIR}/sce_sys/livearea/contents/template.xml sce_sys/livearea/contents/template.xml)
add_dependencies(vocr_par.vpk-vpk livearea)
