//===- fusepass-opt.cpp - mlir-opt style driver for FusePass --------------===//
//
// Stock mlir-opt cannot load a plugin when it links MLIR statically on Windows,
// so this driver offers the same MlirOptMain command line with the FusePass
// passes registered directly.
//
//===----------------------------------------------------------------------===//

#include "FusePass/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::fusepass::registerFusePass();

  // Only the dialects FusePass tests and examples parse.
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "FusePass elementwise fusion driver\n", registry));
}
