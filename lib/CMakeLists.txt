set(FUSEPASS_PASS_SOURCES
  ElementwiseFusion.cpp
  FoldTransposeIntoMatmul.cpp
  Passes.cpp
  )

# Only the plugin target builds Plugin.cpp, and LLVM's CMake rejects files in a
# directory that no target lists unless they are marked optional.
set(LLVM_OPTIONAL_SOURCES Plugin.cpp)

set(FUSEPASS_MLIR_LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTransformUtils
  )

# Linked into the fusepass-opt driver.
add_mlir_library(FusePassTransforms
  ${FUSEPASS_PASS_SOURCES}

  DEPENDS
  FusePassPassIncGen

  LINK_LIBS PUBLIC
  ${FUSEPASS_MLIR_LIBS}
  )

if(FUSEPASS_ENABLE_PLUGIN)
  # The plugin compiles the pass sources itself rather than linking
  # FusePassTransforms, which would pull a second copy of static MLIR libraries
  # into the module and split the pass registry in two.
  add_llvm_library(FusePassPlugin MODULE BUILDTREE_ONLY
    ${FUSEPASS_PASS_SOURCES}
    Plugin.cpp

    DEPENDS
    FusePassPassIncGen
    )
  if(FUSEPASS_MLIR_LIB_TYPE STREQUAL "SHARED_LIBRARY")
    target_link_libraries(FusePassPlugin PRIVATE ${FUSEPASS_MLIR_LIBS})
  endif()
endif()
