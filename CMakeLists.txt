cmake_minimum_required(VERSION 3.20)
project(FusePass LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
# HandleLLVMOptions copies the RTTI and exception flags of the LLVM build. A
# mismatch here shows up later as confusing "undefined typeinfo" link errors.
include(HandleLLVMOptions)

message(STATUS "FusePass compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})")
message(STATUS "FusePass LLVM/MLIR: ${LLVM_PACKAGE_VERSION} from ${MLIR_DIR}")
message(STATUS "FusePass LLVM build: assertions=${LLVM_ENABLE_ASSERTIONS} rtti=${LLVM_ENABLE_RTTI} eh=${LLVM_ENABLE_EH}")

# A plugin must share one copy of MLIR with the mlir-opt that loads it. On ELF
# the plugin resolves MLIR symbols from the executable at load time. A Windows
# DLL has to resolve them at link time, so it can only share MLIR when MLIR
# itself is built as DLLs. The stock MSYS2 mlir-opt.exe links MLIR statically,
# so the plugin is off there and fusepass-opt is the entry point.
get_target_property(FUSEPASS_MLIR_LIB_TYPE MLIRIR TYPE)
if(WIN32 AND NOT FUSEPASS_MLIR_LIB_TYPE STREQUAL "SHARED_LIBRARY")
  set(FUSEPASS_PLUGIN_DEFAULT OFF)
else()
  set(FUSEPASS_PLUGIN_DEFAULT ON)
endif()
option(FUSEPASS_ENABLE_PLUGIN "Build the mlir-opt pass plugin"
       ${FUSEPASS_PLUGIN_DEFAULT})
message(STATUS "FusePass plugin: ${FUSEPASS_ENABLE_PLUGIN} (MLIR libraries are ${FUSEPASS_MLIR_LIB_TYPE})")

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
# LLVM_DEFINITIONS is one space separated string. Passed unsplit, everything
# after the first -D ends up inside the value of the first macro.
separate_arguments(FUSEPASS_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${FUSEPASS_LLVM_DEFINITIONS})

add_subdirectory(include/FusePass)
add_subdirectory(lib)
add_subdirectory(tools/fusepass-opt)
add_subdirectory(test)
