// RUN: %fusepass-opt %s --split-input-file --fusepass-elementwise-fusion | FileCheck %s

// CHECK-DAG:   #[[MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @two_op_chain(
// CHECK-SAME:    %[[A:.*]]: tensor<4xf32>, %[[B:.*]]: tensor<4xf32>, %[[C:.*]]: tensor<4xf32>)
// CHECK-NOT:     arith.
// CHECK:         %[[INIT:.*]] = tensor.empty() : tensor<4xf32>
// CHECK-NOT:     arith.
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME:      iterator_types = ["parallel"]
// CHECK-SAME:      ins(%[[A]], %[[B]], %[[C]] : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<4xf32>)
// CHECK:         ^bb0(%[[EA:.*]]: f32, %[[EB:.*]]: f32, %[[EC:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[SUM:.*]] = arith.addf %[[EA]], %[[EB]] : f32
// CHECK-NEXT:      %[[MUL:.*]] = arith.mulf %[[SUM]], %[[EC]] : f32
// CHECK-NEXT:      linalg.yield %[[MUL]] : f32
// CHECK:         } -> tensor<4xf32>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.
// CHECK:         return %[[R]] : tensor<4xf32>
func.func @two_op_chain(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b : tensor<4xf32>
  %1 = arith.mulf %0, %c : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// Five ops, covering arith and math, with one leaf reused by two ops.
// CHECK-LABEL: func.func @five_op_chain(
// CHECK-SAME:    %[[A:.*]]: tensor<2x3xf32>, %[[B:.*]]: tensor<2x3xf32>)
// CHECK-NOT:     arith.
// CHECK-NOT:     math.
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<2x3xf32>, tensor<2x3xf32>)
// CHECK:         ^bb0(%[[EA:.*]]: f32, %[[EB:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[V0:.*]] = arith.subf %[[EA]], %[[EB]] : f32
// CHECK-NEXT:      %[[V1:.*]] = math.absf %[[V0]] : f32
// CHECK-NEXT:      %[[V2:.*]] = math.sqrt %[[V1]] : f32
// CHECK-NEXT:      %[[V3:.*]] = arith.divf %[[V2]], %[[EB]] : f32
// CHECK-NEXT:      %[[V4:.*]] = math.exp %[[V3]] : f32
// CHECK-NEXT:      linalg.yield %[[V4]] : f32
// CHECK:         } -> tensor<2x3xf32>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.
// CHECK-NOT:     math.
// CHECK:         return %[[R]]
func.func @five_op_chain(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = arith.subf %a, %b : tensor<2x3xf32>
  %1 = math.absf %0 : tensor<2x3xf32>
  %2 = math.sqrt %1 : tensor<2x3xf32>
  %3 = arith.divf %2, %b : tensor<2x3xf32>
  %4 = math.exp %3 : tensor<2x3xf32>
  return %4 : tensor<2x3xf32>
}

// -----

// (a + b) * (c - d): a tree with two branches.
// CHECK-LABEL: func.func @tree(
// CHECK-NOT:     arith.
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK:         ^bb0(%[[EA:.*]]: f32, %[[EB:.*]]: f32, %[[EC:.*]]: f32, %[[ED:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[L:.*]] = arith.addf %[[EA]], %[[EB]] : f32
// CHECK-NEXT:      %[[M:.*]] = arith.subf %[[EC]], %[[ED]] : f32
// CHECK-NEXT:      %[[N:.*]] = arith.mulf %[[L]], %[[M]] : f32
// CHECK-NEXT:      linalg.yield %[[N]] : f32
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.
// CHECK:         return %[[R]]
func.func @tree(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>, %d: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b : tensor<4xf32>
  %1 = arith.subf %c, %d : tensor<4xf32>
  %2 = arith.mulf %0, %1 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// x * x with x = a + b: two uses, one consumer. Every use is inside the group.
// CHECK-LABEL: func.func @same_consumer_twice(
// CHECK-NOT:     arith.
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>)
// CHECK:         ^bb0(%[[EA:.*]]: f32, %[[EB:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[X:.*]] = arith.addf %[[EA]], %[[EB]] : f32
// CHECK-NEXT:      %[[Y:.*]] = arith.mulf %[[X]], %[[X]] : f32
// CHECK-NEXT:      linalg.yield %[[Y]] : f32
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.
// CHECK:         return %[[R]]
func.func @same_consumer_twice(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %x = arith.addf %a, %b : tensor<4xf32>
  %y = arith.mulf %x, %x : tensor<4xf32>
  return %y : tensor<4xf32>
}

// -----

// A diamond: x has two users, but both end up in the same group. Fuses because
// the collection runs to a fixed point. The %y branch is listed second, so x is
// reached through the %p branch before its other user has joined.
// CHECK-LABEL: func.func @diamond(
// CHECK-NOT:     arith.
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>)
// CHECK:         ^bb0(%[[EA:.*]]: f32, %[[EB:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[X:.*]] = arith.addf %[[EA]], %[[EB]] : f32
// CHECK-NEXT:      %[[P:.*]] = arith.mulf %[[X]], %[[EA]] : f32
// CHECK-NEXT:      %[[Q:.*]] = arith.subf %[[X]], %[[EB]] : f32
// CHECK-NEXT:      %[[Y:.*]] = math.absf %[[Q]] : f32
// CHECK-NEXT:      %[[Z:.*]] = arith.divf %[[P]], %[[Y]] : f32
// CHECK-NEXT:      linalg.yield %[[Z]] : f32
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.
// CHECK-NOT:     math.
// CHECK:         return %[[R]]
func.func @diamond(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %x = arith.addf %a, %b : tensor<4xf32>
  %p = arith.mulf %x, %a : tensor<4xf32>
  %q = arith.subf %x, %b : tensor<4xf32>
  %y = math.absf %q : tensor<4xf32>
  %z = arith.divf %p, %y : tensor<4xf32>
  return %z : tensor<4xf32>
}

// -----

// Two chains that share nothing become two generics.
// CHECK-LABEL: func.func @two_independent_chains(
// CHECK-NOT:     arith.
// CHECK:         %[[R0:.*]] = linalg.generic
// CHECK:           arith.addf
// CHECK-NEXT:      arith.mulf
// CHECK-NOT:     arith.
// CHECK:         %[[R1:.*]] = linalg.generic
// CHECK:           arith.subf
// CHECK-NEXT:      arith.divf
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.
// CHECK:         return %[[R0]], %[[R1]]
func.func @two_independent_chains(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.addf %a, %b : tensor<4xf32>
  %1 = arith.mulf %0, %b : tensor<4xf32>
  %2 = arith.subf %a, %b : tensor<4xf32>
  %3 = arith.divf %2, %b : tensor<4xf32>
  return %1, %3 : tensor<4xf32>, tensor<4xf32>
}

// -----

// Integer element type.
// CHECK-LABEL: func.func @integer_chain(
// CHECK-NOT:     arith.
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK:         ^bb0(%[[EA:.*]]: i32, %[[EB:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:      %[[V0:.*]] = arith.addi %[[EA]], %[[EB]] : i32
// CHECK-NEXT:      %[[V1:.*]] = arith.muli %[[V0]], %[[EB]] : i32
// CHECK-NEXT:      linalg.yield %[[V1]] : i32
// CHECK:         } -> tensor<4xi32>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.
// CHECK:         return %[[R]]
func.func @integer_chain(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = arith.addi %a, %b : tensor<4xi32>
  %1 = arith.muli %0, %b : tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

// f16 element type.
// CHECK-LABEL: func.func @f16_chain(
// CHECK-NOT:     arith.
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK:         ^bb0(%[[EA:.*]]: f16, %[[EB:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:      %[[V0:.*]] = arith.maximumf %[[EA]], %[[EB]] : f16
// CHECK-NEXT:      %[[V1:.*]] = arith.minimumf %[[V0]], %[[EB]] : f16
// CHECK-NEXT:      %[[V2:.*]] = arith.negf %[[V1]] : f16
// CHECK-NEXT:      linalg.yield %[[V2]] : f16
// CHECK:         } -> tensor<4xf16>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.
// CHECK:         return %[[R]]
func.func @f16_chain(%a: tensor<4xf16>, %b: tensor<4xf16>) -> tensor<4xf16> {
  %0 = arith.maximumf %a, %b : tensor<4xf16>
  %1 = arith.minimumf %0, %b : tensor<4xf16>
  %2 = arith.negf %1 : tensor<4xf16>
  return %2 : tensor<4xf16>
}

// -----

// Rewrite flags on the original ops must survive on the scalar clones.
// CHECK-LABEL: func.func @flags_are_kept(
// CHECK:         linalg.generic
// CHECK:           arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f32
// CHECK-NEXT:      arith.mulf %{{.*}}, %{{.*}} fastmath<nnan> : f32
func.func @flags_are_kept(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b fastmath<fast> : tensor<4xf32>
  %1 = arith.mulf %0, %b fastmath<nnan> : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @integer_overflow_flags_are_kept(
// CHECK:         linalg.generic
// CHECK:           arith.addi %{{.*}}, %{{.*}} overflow<nsw> : i32
// CHECK-NEXT:      arith.muli %{{.*}}, %{{.*}} overflow<nuw> : i32
func.func @integer_overflow_flags_are_kept(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = arith.addi %a, %b overflow<nsw> : tensor<4xi32>
  %1 = arith.muli %0, %b overflow<nuw> : tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

// Rank zero tensors have no loops but still fuse.
// CHECK-LABEL: func.func @rank_zero(
// CHECK-NOT:     arith.
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = []
// CHECK:           arith.addf
// CHECK-NEXT:      arith.mulf
// CHECK:         } -> tensor<f32>
// CHECK-NOT:     arith.
// CHECK:         return %[[R]]
func.func @rank_zero(%a: tensor<f32>, %b: tensor<f32>) -> tensor<f32> {
  %0 = arith.addf %a, %b : tensor<f32>
  %1 = arith.mulf %0, %b : tensor<f32>
  return %1 : tensor<f32>
}
