// RUN: %fusepass-opt %s --split-input-file --fusepass-elementwise-fusion | FileCheck %s

// Rule: a value with two different users stays materialized. Neither user may
// absorb it, and each user is then a group of one.
// CHECK-LABEL: func.func @two_users(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK-NEXT:    %[[P:.*]] = arith.mulf %[[X]], %{{.*}} : tensor<4xf32>
// CHECK-NEXT:    %[[Q:.*]] = arith.subf %[[X]], %{{.*}} : tensor<4xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[P]], %[[Q]]
func.func @two_users(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %x = arith.addf %a, %b : tensor<4xf32>
  %p = arith.mulf %x, %a : tensor<4xf32>
  %q = arith.subf %x, %b : tensor<4xf32>
  return %p, %q : tensor<4xf32>, tensor<4xf32>
}

// -----

// A value also used by a non elementwise op, here the return, stays too.
// CHECK-LABEL: func.func @escapes_through_return(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf
// CHECK-NEXT:    %[[Y:.*]] = arith.mulf %[[X]], %{{.*}}
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[X]], %[[Y]]
func.func @escapes_through_return(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %x = arith.addf %a, %b : tensor<4xf32>
  %y = arith.mulf %x, %a : tensor<4xf32>
  return %x, %y : tensor<4xf32>, tensor<4xf32>
}

// -----

// A shared value does not stop the ops above it from fusing with it: x is the
// root of its own group, and p and q stay as single ops.
// CHECK-LABEL: func.func @two_users_with_producer(
// CHECK:         %[[X:.*]] = linalg.generic
// CHECK:           arith.addf
// CHECK-NEXT:      arith.mulf
// CHECK:         } -> tensor<4xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         %[[P:.*]] = arith.subf %[[X]], %{{.*}} : tensor<4xf32>
// CHECK-NEXT:    %[[Q:.*]] = arith.divf %[[X]], %{{.*}} : tensor<4xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[P]], %[[Q]]
func.func @two_users_with_producer(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %s = arith.addf %a, %b : tensor<4xf32>
  %x = arith.mulf %s, %a : tensor<4xf32>
  %p = arith.subf %x, %a : tensor<4xf32>
  %q = arith.divf %x, %b : tensor<4xf32>
  return %p, %q : tensor<4xf32>, tensor<4xf32>
}

// -----

// Rule: a group of one is left alone.
// CHECK-LABEL: func.func @single_op(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[X]]
func.func @single_op(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %x = arith.addf %a, %b : tensor<4xf32>
  return %x : tensor<4xf32>
}

// -----

// A function argument used twice has no producer to fuse.
// CHECK-LABEL: func.func @function_argument(
// CHECK-SAME:    %[[A:.*]]: tensor<4xf32>)
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.mulf %[[A]], %[[A]] : tensor<4xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[X]]
func.func @function_argument(%a: tensor<4xf32>) -> tensor<4xf32> {
  %x = arith.mulf %a, %a : tensor<4xf32>
  return %x : tensor<4xf32>
}

// -----

// Rule: the chain breaks at a non elementwise op.
// CHECK-LABEL: func.func @broken_by_extract_slice(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK-NEXT:    %[[S:.*]] = tensor.extract_slice %[[X]]
// CHECK-NEXT:    %[[Y:.*]] = arith.mulf %[[S]], %{{.*}} : tensor<2xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[Y]]
func.func @broken_by_extract_slice(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<2xf32>) -> tensor<2xf32> {
  %x = arith.addf %a, %b : tensor<4xf32>
  %s = tensor.extract_slice %x[0] [2] [1] : tensor<4xf32> to tensor<2xf32>
  %y = arith.mulf %s, %c : tensor<2xf32>
  return %y : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @broken_by_matmul(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf %{{.*}}, %{{.*}} : tensor<2x2xf32>
// CHECK-NEXT:    %[[M:.*]] = linalg.matmul ins(%[[X]], %{{.*}} : tensor<2x2xf32>, tensor<2x2xf32>)
// CHECK-NEXT:    %[[Y:.*]] = arith.mulf %[[M]], %{{.*}} : tensor<2x2xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[Y]]
func.func @broken_by_matmul(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %init: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %x = arith.addf %a, %b : tensor<2x2xf32>
  %m = linalg.matmul ins(%x, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
  %y = arith.mulf %m, %a : tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// -----

// Rule: a type changing op is not fusable, even though it is elementwise.
// CHECK-LABEL: func.func @type_changing_op(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf %{{.*}}, %{{.*}} : tensor<4xf16>
// CHECK-NEXT:    %[[E:.*]] = arith.extf %[[X]] : tensor<4xf16> to tensor<4xf32>
// CHECK-NEXT:    %[[Y:.*]] = arith.mulf %[[E]], %{{.*}} : tensor<4xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[Y]]
func.func @type_changing_op(%a: tensor<4xf16>, %b: tensor<4xf16>, %c: tensor<4xf32>) -> tensor<4xf32> {
  %x = arith.addf %a, %b : tensor<4xf16>
  %e = arith.extf %x : tensor<4xf16> to tensor<4xf32>
  %y = arith.mulf %e, %c : tensor<4xf32>
  return %y : tensor<4xf32>
}

// -----

// Rule: every op in a group has the same tensor type. A cast separates two
// otherwise fusable ops.
// CHECK-LABEL: func.func @shape_changes_through_cast(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK-NEXT:    %[[C:.*]] = tensor.cast %[[X]] : tensor<4xf32> to tensor<?xf32>
// CHECK-NEXT:    %[[Y:.*]] = arith.mulf %[[C]], %{{.*}} : tensor<?xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[Y]]
func.func @shape_changes_through_cast(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<?xf32>) -> tensor<?xf32> {
  %x = arith.addf %a, %b : tensor<4xf32>
  %cast = tensor.cast %x : tensor<4xf32> to tensor<?xf32>
  %y = arith.mulf %cast, %c : tensor<?xf32>
  return %y : tensor<?xf32>
}

// -----

// Rule: no fusion across region boundaries. The producer is outside the scf.if.
// CHECK-LABEL: func.func @producer_in_other_region(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK:         scf.if
// CHECK-NEXT:      %[[Y:.*]] = arith.mulf %[[X]], %{{.*}} : tensor<4xf32>
// CHECK-NEXT:      scf.yield %[[Y]]
// CHECK:         else
// CHECK-NEXT:      scf.yield
// CHECK-NOT:     linalg.generic
func.func @producer_in_other_region(%cond: i1, %a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %x = arith.addf %a, %b : tensor<4xf32>
  %r = scf.if %cond -> tensor<4xf32> {
    %y = arith.mulf %x, %a : tensor<4xf32>
    scf.yield %y : tensor<4xf32>
  } else {
    scf.yield %a : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// -----

// Inside one region a chain still fuses, and the generic stays in that region.
// CHECK-LABEL: func.func @fuses_inside_region(
// CHECK-NOT:     linalg.generic
// CHECK:         scf.if
// CHECK-NOT:       arith.
// CHECK:           %[[R:.*]] = linalg.generic
// CHECK:             arith.addf
// CHECK-NEXT:        arith.mulf
// CHECK:           scf.yield %[[R]]
// CHECK:         else
// CHECK-NOT:     linalg.generic
func.func @fuses_inside_region(%cond: i1, %a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %r = scf.if %cond -> tensor<4xf32> {
    %x = arith.addf %a, %b : tensor<4xf32>
    %y = arith.mulf %x, %a : tensor<4xf32>
    scf.yield %y : tensor<4xf32>
  } else {
    scf.yield %a : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// -----

// Rule: ranked tensors only.
// CHECK-LABEL: func.func @unranked_tensor(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.addf %{{.*}}, %{{.*}} : tensor<*xf32>
// CHECK-NEXT:    %[[Y:.*]] = arith.mulf %[[X]], %{{.*}} : tensor<*xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[Y]]
func.func @unranked_tensor(%a: tensor<*xf32>, %b: tensor<*xf32>) -> tensor<*xf32> {
  %x = arith.addf %a, %b : tensor<*xf32>
  %y = arith.mulf %x, %a : tensor<*xf32>
  return %y : tensor<*xf32>
}

// -----

// Rule: only allowlisted ops fuse. arith.remf is elementwise but not listed.
// CHECK-LABEL: func.func @not_in_allowlist(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.remf %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK-NEXT:    %[[Y:.*]] = arith.mulf %[[X]], %{{.*}} : tensor<4xf32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[Y]]
func.func @not_in_allowlist(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %x = arith.remf %a, %b : tensor<4xf32>
  %y = arith.mulf %x, %a : tensor<4xf32>
  return %y : tensor<4xf32>
}

// -----

// Rule: integer division can trap, so it is not Pure and must not fuse.
// CHECK-LABEL: func.func @integer_division(
// CHECK-NOT:     linalg.generic
// CHECK:         %[[X:.*]] = arith.divsi %{{.*}}, %{{.*}} : tensor<4xi32>
// CHECK-NEXT:    %[[Y:.*]] = arith.addi %[[X]], %{{.*}} : tensor<4xi32>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[Y]]
func.func @integer_division(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %x = arith.divsi %a, %b : tensor<4xi32>
  %y = arith.addi %x, %a : tensor<4xi32>
  return %y : tensor<4xi32>
}

// -----

// Scalars are not this pass's business, even in a chain.
// CHECK-LABEL: func.func @scalar_ops(
// CHECK-NOT:     linalg.generic
// CHECK:         arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:    arith.mulf %{{.*}}, %{{.*}} : f32
func.func @scalar_ops(%a: f32, %b: f32) -> f32 {
  %x = arith.addf %a, %b : f32
  %y = arith.mulf %x, %a : f32
  return %y : f32
}
