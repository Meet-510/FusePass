// REQUIRES: stats
// RUN: %fusepass-opt %s --fusepass-pipeline --mlir-pass-statistics --mlir-pass-statistics-display=list -o /dev/null 2>&1 | FileCheck %s

// Statistics only exist on LLVM builds with LLVM_ENABLE_STATS, usually the ones
// with assertions. lit.cfg.py detects that by running the tool.

// CHECK-DAG: (S) 2 fused-groups
// CHECK-DAG: (S) 5 fused-ops
// CHECK-DAG: (S) 3 rejected-candidates
// CHECK-DAG: (S) 1 transposes-folded

// Two groups: 2 ops and 3 ops.
func.func @fusable(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.addf %a, %b : tensor<4xf32>
  %1 = arith.mulf %0, %b : tensor<4xf32>
  %2 = arith.subf %a, %b : tensor<4xf32>
  %3 = math.absf %2 : tensor<4xf32>
  %4 = math.sqrt %3 : tensor<4xf32>
  return %1, %4 : tensor<4xf32>, tensor<4xf32>
}

// Three rejected candidates: the shared value and both of its users.
func.func @two_users(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %x = arith.addf %a, %b : tensor<4xf32>
  %p = arith.mulf %x, %a : tensor<4xf32>
  %q = arith.subf %x, %b : tensor<4xf32>
  return %p, %q : tensor<4xf32>, tensor<4xf32>
}

func.func @fold(%x: tensor<3x2xf32>, %y: tensor<3x4xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %init = tensor.empty() : tensor<2x3xf32>
  %t = linalg.transpose ins(%x : tensor<3x2xf32>) outs(%init : tensor<2x3xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}
