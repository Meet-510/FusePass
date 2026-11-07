// RUN: %fusepass-opt %s --split-input-file --fusepass-elementwise-fusion | FileCheck %s

// Both dimensions are dynamic, so tensor.empty needs a tensor.dim for each.
// CHECK-DAG:   #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @all_dynamic(
// CHECK-SAME:    %[[A:.*]]: tensor<?x?xf32>, %[[B:.*]]: tensor<?x?xf32>)
// CHECK-NOT:     arith.addf
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[D0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:     %[[D1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xf32>
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x?xf32>)
// CHECK:           arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:      arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.addf
// CHECK-NOT:     arith.mulf
// CHECK:         return %[[R]]
func.func @all_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = arith.addf %a, %b : tensor<?x?xf32>
  %1 = arith.mulf %0, %b : tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// Only the dynamic dimension needs a size operand.
// CHECK-LABEL: func.func @mixed_static_dynamic(
// CHECK-SAME:    %[[A:.*]]: tensor<3x?xf32>,
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[D1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<3x?xf32>
// CHECK-NOT:     tensor.dim
// CHECK:         %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<3x?xf32>
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK:         } -> tensor<3x?xf32>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.subf
// CHECK:         return %[[R]]
func.func @mixed_static_dynamic(%a: tensor<3x?xf32>, %b: tensor<3x?xf32>) -> tensor<3x?xf32> {
  %0 = arith.subf %a, %b : tensor<3x?xf32>
  %1 = math.absf %0 : tensor<3x?xf32>
  return %1 : tensor<3x?xf32>
}

// -----

// The two-users rule holds for dynamic shapes as well.
// CHECK-LABEL: func.func @dynamic_two_users(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf %{{.*}}, %{{.*}} : tensor<?xf32>
// CHECK-NEXT:    arith.mulf %[[X]], %{{.*}} : tensor<?xf32>
// CHECK-NEXT:    arith.subf %[[X]], %{{.*}} : tensor<?xf32>
// CHECK-NOT:     linalg.generic
func.func @dynamic_two_users(%a: tensor<?xf32>, %b: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %x = arith.addf %a, %b : tensor<?xf32>
  %p = arith.mulf %x, %a : tensor<?xf32>
  %q = arith.subf %x, %b : tensor<?xf32>
  return %p, %q : tensor<?xf32>, tensor<?xf32>
}
