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
}

#endif // FUSEPASS_PASSES_TD
