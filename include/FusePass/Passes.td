//===- Passes.td - FusePass pass declarations --------------*- tablegen -*-===//

#ifndef FUSEPASS_PASSES_TD
#define FUSEPASS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ElementwiseFusion : Pass<"fusepass-elementwise-fusion", "func::FuncOp"> {
  let summary = "Fuse chains of elementwise tensor ops into one linalg.generic";
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "math::MathDialect",
    "tensor::TensorDialect"
  ];
  let statistics = [
    Statistic<"numFusedGroups", "fused-groups",
              "Number of groups rewritten into one linalg.generic">,
    Statistic<"numFusedOps", "fused-ops",
              "Number of elementwise ops moved into a linalg.generic">,
    Statistic<"numRejectedCandidates", "rejected-candidates",
              "Number of elementwise tensor ops left unfused at a group root">
  ];
}

#endif // FUSEPASS_PASSES_TD
