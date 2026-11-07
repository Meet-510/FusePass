//===- Passes.cpp - FusePass registration ---------------------------------===//

#include "FusePass/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::fusepass {

#define GEN_PASS_REGISTRATION
#include "FusePass/Passes.h.inc"

// registerFusePassPasses() comes from the TableGen output above.
void registerFusePass() {
  registerFusePassPasses();
  PassPipelineRegistration<>(
      "fusepass-pipeline", "Run elementwise fusion, then the transpose fold",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createElementwiseFusion());
        pm.addNestedPass<func::FuncOp>(createFoldTransposeIntoMatmul());
      });
}

} // namespace mlir::fusepass
