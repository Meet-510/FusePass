//===- Plugin.cpp - mlir-opt --load-pass-plugin entry point ---------------===//

#include "FusePass/Passes.h"

#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

// mlir-opt looks this symbol up by name after loading the shared library.
extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "FusePass", LLVM_VERSION_STRING,
          []() { mlir::fusepass::registerFusePass(); }};
}
