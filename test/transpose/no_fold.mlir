// RUN: %fusepass-opt %s --split-input-file --fusepass-fold-transpose-matmul | FileCheck %s

// Rule: the permutation must be exactly [1, 0]. [0, 1] is an identity transpose.
// CHECK-LABEL: func.func @identity_permutation(
// CHECK:         %[[T:.*]] = linalg.transpose
// CHECK-SAME:      permutation = [0, 1]
// CHECK-NOT:     indexing_maps
// CHECK:         linalg.matmul ins(%[[T]], %{{.*}} : tensor<2x3xf32>, tensor<3x4xf32>)
func.func @identity_permutation(%x: tensor<2x3xf32>, %y: tensor<3x4xf32>, %init: tensor<2x3xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %t = linalg.transpose ins(%x : tensor<2x3xf32>) outs(%init : tensor<2x3xf32>) permutation = [0, 1]
  %r = linalg.matmul ins(%t, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// Rule: rank 2 only. A rank 3 transpose cannot feed linalg.matmul, and the
// batched op it can feed is not matched.
// CHECK-LABEL: func.func @rank_three(
// CHECK:         %[[T:.*]] = linalg.transpose
// CHECK-SAME:      permutation = [0, 2, 1]
// CHECK:         linalg.batch_matmul ins(%[[T]], %{{.*}} : tensor<5x2x3xf32>, tensor<5x3x4xf32>)
func.func @rank_three(%x: tensor<5x3x2xf32>, %y: tensor<5x3x4xf32>, %init: tensor<5x2x3xf32>, %out: tensor<5x2x4xf32>) -> tensor<5x2x4xf32> {
  %t = linalg.transpose ins(%x : tensor<5x3x2xf32>) outs(%init : tensor<5x2x3xf32>) permutation = [0, 2, 1]
  %r = linalg.batch_matmul ins(%t, %y : tensor<5x2x3xf32>, tensor<5x3x4xf32>) outs(%out : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
  return %r : tensor<5x2x4xf32>
}

// -----

// Rule: rank 2 only, so a rank 1 operand that matmul broadcasts has no transpose
// to fold. This is the shape case the rule rejects on a plain matmul operand:
// B's transpose folds, and A keeps its broadcast map.
// CHECK-DAG:   #[[A:.*]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   #[[B:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-LABEL: func.func @rank_one_broadcast_operand(
// CHECK-SAME:    %[[V:[a-z0-9_]+]]: tensor<3xf32>, %[[Y:[a-z0-9_]+]]: tensor<4x3xf32>,
// CHECK-NOT:     linalg.transpose
// CHECK:         linalg.matmul
// CHECK-SAME:      indexing_maps = [#[[A]], #[[B]], #{{.*}}]
// CHECK-SAME:      ins(%[[V]], %[[Y]] : tensor<3xf32>, tensor<4x3xf32>)
func.func @rank_one_broadcast_operand(%v: tensor<3xf32>, %y: tensor<4x3xf32>, %init: tensor<3x4xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %t = linalg.transpose ins(%y : tensor<4x3xf32>) outs(%init : tensor<3x4xf32>) permutation = [1, 0]
  %r = linalg.matmul
         indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
         ins(%v, %t : tensor<3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// Rule: element types must be equal across the matmul.
// CHECK-LABEL: func.func @mixed_element_types(
// CHECK:         %[[T:.*]] = linalg.transpose
// CHECK-NOT:     indexing_maps
// CHECK:         linalg.matmul ins(%[[T]], %{{.*}} : tensor<2x3xf16>, tensor<3x4xf32>)
func.func @mixed_element_types(%x: tensor<3x2xf16>, %y: tensor<3x4xf32>, %init: tensor<2x3xf16>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %t = linalg.transpose ins(%x : tensor<3x2xf16>) outs(%init : tensor<2x3xf16>) permutation = [1, 0]
  %r = linalg.matmul ins(%t, %y : tensor<2x3xf16>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// Only a transpose directly on the operand counts: a cast in between stops it.
// CHECK-LABEL: func.func @cast_between(
// CHECK:         %[[T:.*]] = linalg.transpose
// CHECK:         %[[C:.*]] = tensor.cast %[[T]]
// CHECK-NOT:     indexing_maps
// CHECK:         linalg.matmul ins(%[[C]], %{{.*}} : tensor<?x?xf32>, tensor<3x4xf32>)
func.func @cast_between(%x: tensor<3x2xf32>, %y: tensor<3x4xf32>, %init: tensor<2x3xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %t = linalg.transpose ins(%x : tensor<3x2xf32>) outs(%init : tensor<2x3xf32>) permutation = [1, 0]
  %c = tensor.cast %t : tensor<2x3xf32> to tensor<?x?xf32>
  %r = linalg.matmul ins(%c, %y : tensor<?x?xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// Transposing the result is a different rewrite and is not matched.
// CHECK-LABEL: func.func @transpose_of_result(
// CHECK:         %[[M:.*]] = linalg.matmul
// CHECK-NOT:     indexing_maps
// CHECK:         linalg.transpose ins(%[[M]] : tensor<2x4xf32>)
func.func @transpose_of_result(%x: tensor<2x3xf32>, %y: tensor<3x4xf32>, %out: tensor<2x4xf32>, %init: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %m = linalg.matmul ins(%x, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  %t = linalg.transpose ins(%m : tensor<2x4xf32>) outs(%init : tensor<4x2xf32>) permutation = [1, 0]
  return %t : tensor<4x2xf32>
}
