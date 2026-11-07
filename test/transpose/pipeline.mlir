// RUN: %fusepass-opt %s --fusepass-pipeline | FileCheck %s

// The pipeline alias runs both passes.
// CHECK-LABEL: func.func @both_passes(
// CHECK-NOT:     linalg.transpose
// CHECK-NOT:     arith.addf
// CHECK:         %[[SUM:.*]] = linalg.generic
// CHECK:           arith.addf
// CHECK-NEXT:      arith.mulf
// CHECK-NOT:     linalg.transpose
// CHECK:         linalg.matmul
// CHECK-SAME:      indexing_maps
// CHECK-SAME:      ins(%{{.*}}, %[[SUM]] : tensor<3x2xf32>, tensor<3x4xf32>)
func.func @both_passes(%x: tensor<3x2xf32>, %y: tensor<3x4xf32>, %z: tensor<3x4xf32>, %init: tensor<2x3xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %s = arith.addf %y, %z : tensor<3x4xf32>
  %m = arith.mulf %s, %z : tensor<3x4xf32>
  %t = linalg.transpose ins(%x : tensor<3x2xf32>) outs(%init : tensor<2x3xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t, %m : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}
