// RUN: %fusepass-opt %s --split-input-file --fusepass-fold-transpose-matmul | FileCheck %s

// The maps of plain matmul are A: (d0, d2), B: (d2, d1), C: (d0, d1).

// Fold A: its map swaps to (d2, d0) and its now dead transpose and init go away.
// CHECK-DAG:   #[[A:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG:   #[[B:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:   #[[C:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @fold_a(
// CHECK-SAME:    %[[X:.*]]: tensor<3x2xf32>, %[[Y:.*]]: tensor<3x4xf32>, %[[OUT:.*]]: tensor<2x4xf32>)
// CHECK-NOT:     linalg.transpose
// CHECK-NOT:     tensor.empty
// CHECK:         %[[R:.*]] = linalg.matmul
// CHECK-SAME:      indexing_maps = [#[[A]], #[[B]], #[[C]]]
// CHECK-SAME:      ins(%[[X]], %[[Y]] : tensor<3x2xf32>, tensor<3x4xf32>)
// CHECK-SAME:      outs(%[[OUT]] : tensor<2x4xf32>)
// CHECK-NOT:     linalg.transpose
// CHECK:         return %[[R]]
func.func @fold_a(%x: tensor<3x2xf32>, %y: tensor<3x4xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %init = tensor.empty() : tensor<2x3xf32>
  %t = linalg.transpose ins(%x : tensor<3x2xf32>) outs(%init : tensor<2x3xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// Fold B: its map swaps to (d1, d2).
// CHECK-DAG:   #[[A:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:   #[[B:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:   #[[C:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @fold_b(
// CHECK-SAME:    %[[X:.*]]: tensor<2x3xf32>, %[[Y:.*]]: tensor<4x3xf32>, %[[OUT:.*]]: tensor<2x4xf32>)
// CHECK-NOT:     linalg.transpose
// CHECK:         %[[R:.*]] = linalg.matmul
// CHECK-SAME:      indexing_maps = [#[[A]], #[[B]], #[[C]]]
// CHECK-SAME:      ins(%[[X]], %[[Y]] : tensor<2x3xf32>, tensor<4x3xf32>)
// CHECK-NOT:     linalg.transpose
// CHECK:         return %[[R]]
func.func @fold_b(%x: tensor<2x3xf32>, %y: tensor<4x3xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %init = tensor.empty() : tensor<3x4xf32>
  %t = linalg.transpose ins(%y : tensor<4x3xf32>) outs(%init : tensor<3x4xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%x, %t : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// Fold both operands in one rewrite.
// CHECK-DAG:   #[[A:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG:   #[[B:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:   #[[C:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @fold_both(
// CHECK-SAME:    %[[X:.*]]: tensor<3x2xf32>, %[[Y:.*]]: tensor<4x3xf32>, %[[OUT:.*]]: tensor<2x4xf32>)
// CHECK-NOT:     linalg.transpose
// CHECK:         %[[R:.*]] = linalg.matmul
// CHECK-SAME:      indexing_maps = [#[[A]], #[[B]], #[[C]]]
// CHECK-SAME:      ins(%[[X]], %[[Y]] : tensor<3x2xf32>, tensor<4x3xf32>)
// CHECK-NOT:     linalg.transpose
// CHECK:         return %[[R]]
func.func @fold_both(%x: tensor<3x2xf32>, %y: tensor<4x3xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %init0 = tensor.empty() : tensor<2x3xf32>
  %t0 = linalg.transpose ins(%x : tensor<3x2xf32>) outs(%init0 : tensor<2x3xf32>) permutation = [1, 0]
  %init1 = tensor.empty() : tensor<3x4xf32>
  %t1 = linalg.transpose ins(%y : tensor<4x3xf32>) outs(%init1 : tensor<3x4xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t0, %t1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// One transpose feeding both operands is folded twice and erased once.
// CHECK-DAG:   #[[A:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG:   #[[B:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-LABEL: func.func @same_transpose_both_operands(
// CHECK-SAME:    %[[X:.*]]: tensor<2x2xf32>, %[[OUT:.*]]: tensor<2x2xf32>)
// CHECK-NOT:     linalg.transpose
// CHECK:         linalg.matmul
// CHECK-SAME:      indexing_maps = [#[[A]], #[[B]], #{{.*}}]
// CHECK-SAME:      ins(%[[X]], %[[X]] : tensor<2x2xf32>, tensor<2x2xf32>)
func.func @same_transpose_both_operands(%x: tensor<2x2xf32>, %out: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %init = tensor.empty() : tensor<2x2xf32>
  %t = linalg.transpose ins(%x : tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t, %t : tensor<2x2xf32>, tensor<2x2xf32>) outs(%out : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// -----

// A transpose with another user stays alive for that user. The fold still
// applies to this matmul.
// CHECK-DAG:   #[[A:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-LABEL: func.func @transpose_has_other_user(
// CHECK-SAME:    %[[X:.*]]: tensor<3x2xf32>,
// CHECK:         %[[T:.*]] = linalg.transpose ins(%[[X]] : tensor<3x2xf32>)
// CHECK:         %[[R:.*]] = linalg.matmul
// CHECK-SAME:      indexing_maps = [#[[A]], #{{.*}}, #{{.*}}]
// CHECK-SAME:      ins(%[[X]], %{{.*}} : tensor<3x2xf32>, tensor<3x4xf32>)
// CHECK:         return %[[R]], %[[T]]
func.func @transpose_has_other_user(%x: tensor<3x2xf32>, %y: tensor<3x4xf32>, %out: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x3xf32>) {
  %init = tensor.empty() : tensor<2x3xf32>
  %t = linalg.transpose ins(%x : tensor<3x2xf32>) outs(%init : tensor<2x3xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r, %t : tensor<2x4xf32>, tensor<2x3xf32>
}

// -----

// Two transposes in a row cancel: A goes back to the default map, which the
// printer then omits.
// CHECK-LABEL: func.func @double_transpose(
// CHECK-SAME:    %[[X:.*]]: tensor<2x3xf32>, %[[Y:.*]]: tensor<3x4xf32>,
// CHECK-NOT:     linalg.transpose
// CHECK-NOT:     indexing_maps
// CHECK:         linalg.matmul ins(%[[X]], %[[Y]] : tensor<2x3xf32>, tensor<3x4xf32>)
func.func @double_transpose(%x: tensor<2x3xf32>, %y: tensor<3x4xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %init0 = tensor.empty() : tensor<3x2xf32>
  %t0 = linalg.transpose ins(%x : tensor<2x3xf32>) outs(%init0 : tensor<3x2xf32>) permutation = [1, 0]
  %init1 = tensor.empty() : tensor<2x3xf32>
  %t1 = linalg.transpose ins(%t0 : tensor<3x2xf32>) outs(%init1 : tensor<2x3xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t1, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// Dynamic sizes fold the same way.
// CHECK-DAG:   #[[A:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-LABEL: func.func @dynamic_shapes(
// CHECK-SAME:    %[[X:[a-z0-9_]+]]: tensor<?x?xf32>, %[[Y:[a-z0-9_]+]]: tensor<?x?xf32>,
// CHECK-NOT:     linalg.transpose
// CHECK:         linalg.matmul
// CHECK-SAME:      indexing_maps = [#[[A]], #{{.*}}, #{{.*}}]
// CHECK-SAME:      ins(%[[X]], %[[Y]] : tensor<?x?xf32>, tensor<?x?xf32>)
func.func @dynamic_shapes(%x: tensor<?x?xf32>, %y: tensor<?x?xf32>, %init: tensor<?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %t = linalg.transpose ins(%x : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t, %y : tensor<?x?xf32>, tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// Rewrite attributes on the matmul survive the fold.
// CHECK-LABEL: func.func @cast_attribute_is_kept(
// CHECK:         linalg.matmul
// CHECK-SAME:      cast = #linalg.type_fn<cast_unsigned>
func.func @cast_attribute_is_kept(%x: tensor<3x2xf32>, %y: tensor<3x4xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %init = tensor.empty() : tensor<2x3xf32>
  %t = linalg.transpose ins(%x : tensor<3x2xf32>) outs(%init : tensor<2x3xf32>) permutation = [1, 0]
  %r = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>} ins(%t, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}
