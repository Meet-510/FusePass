// REQUIRES: plugin
// RUN: %fusepass-opt %s --fusepass-elementwise-fusion | FileCheck %s

// With the plugin feature on, %fusepass-opt is mlir-opt --load-pass-plugin. lit
// only turns the feature on after seeing mlir-opt list the plugin's passes.

// CHECK-LABEL: func.func @through_plugin(
// CHECK-NOT:     arith.addf
// CHECK:         linalg.generic
// CHECK:           arith.addf
// CHECK-NEXT:      arith.mulf
func.func @through_plugin(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b : tensor<4xf32>
  %1 = arith.mulf %0, %b : tensor<4xf32>
  return %1 : tensor<4xf32>
}
