// REQUIRES: asserts
// RUN: %fusepass-opt %s --fusepass-elementwise-fusion --debug-only=fusepass --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s

// LLVM_DEBUG output only exists on LLVM builds with assertions. Threading is off
// because functions run in parallel and their log lines would interleave.

// The driver visits the last op first, so the root comes before its producers.
// CHECK:      [fusepass] candidate: {{.*}}arith.mulf
// CHECK-NEXT:   accepted group of 2 ops:
// CHECK-NEXT:     {{.*}}arith.mulf
// CHECK-NEXT:     {{.*}}arith.addf
func.func @fused(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b : tensor<4xf32>
  %1 = arith.mulf %0, %b : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// The leaf log names the rule that keeps the shared value out of its users.
// CHECK:      [fusepass] candidate: {{.*}}arith.subf
// CHECK-NEXT:   leaf from arith.addf: producer has a user outside the group
// CHECK-NEXT:   rejected: group has only one op
// CHECK-NEXT: [fusepass] candidate: {{.*}}arith.mulf
// CHECK-NEXT:   leaf from arith.addf: producer has a user outside the group
// CHECK-NEXT:   rejected: group has only one op
// CHECK-NEXT: [fusepass] candidate: {{.*}}arith.addf
// CHECK-NEXT:   rejected: group has only one op
func.func @two_users(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %x = arith.addf %a, %b : tensor<4xf32>
  %p = arith.mulf %x, %a : tensor<4xf32>
  %q = arith.subf %x, %b : tensor<4xf32>
  return %p, %q : tensor<4xf32>, tensor<4xf32>
}
