// RUN: %fusepass-opt %s --split-input-file --fusepass-elementwise-fusion | FileCheck %s

// A splat constant becomes a scalar constant in the body, not another input,
// and the tensor constant disappears once nothing uses it.
// CHECK-LABEL: func.func @splat_moves_into_body(
// CHECK-SAME:    %[[A:.*]]: tensor<4xf32>, %[[B:.*]]: tensor<4xf32>)
// CHECK-NOT:     arith.constant
// CHECK-NOT:     dense<
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<4xf32>, tensor<4xf32>)
// CHECK:         ^bb0(%[[EA:.*]]: f32, %[[EB:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:      %[[SUM:.*]] = arith.addf %[[EA]], %[[EB]] : f32
// CHECK-NEXT:      %[[MUL:.*]] = arith.mulf %[[SUM]], %[[TWO]] : f32
// CHECK-NEXT:      linalg.yield %[[MUL]] : f32
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.
// CHECK:         return %[[R]]
func.func @splat_moves_into_body(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %c = arith.constant dense<2.0> : tensor<4xf32>
  %0 = arith.addf %a, %b : tensor<4xf32>
  %1 = arith.mulf %0, %c : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// Integer splats work too, and each constant value appears once in the body.
// CHECK-LABEL: func.func @integer_splats(
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}} : tensor<4xi32>)
// CHECK:         ^bb0(%[[EA:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:      %[[TEN:.*]] = arith.constant 10 : i32
// CHECK-NEXT:      %[[V0:.*]] = arith.addi %[[EA]], %[[TEN]] : i32
// CHECK-NEXT:      %[[V1:.*]] = arith.muli %[[V0]], %[[TEN]] : i32
// CHECK-NEXT:      linalg.yield %[[V1]] : i32
// CHECK-NOT:     arith.addi
func.func @integer_splats(%a: tensor<4xi32>) -> tensor<4xi32> {
  %c = arith.constant dense<10> : tensor<4xi32>
  %0 = arith.addi %a, %c : tensor<4xi32>
  %1 = arith.muli %0, %c : tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

// A constant with a user outside the group is copied into the body and stays.
// CHECK-LABEL: func.func @splat_used_elsewhere(
// CHECK:         %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<4xf32>
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      ins(%{{.*}} : tensor<4xf32>)
// CHECK:           arith.constant 2.000000e+00 : f32
// CHECK-NEXT:      arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:      arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:      linalg.yield
// CHECK-NOT:     arith.
// CHECK:         return %[[R]], %[[CST]]
func.func @splat_used_elsewhere(%a: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %c = arith.constant dense<2.0> : tensor<4xf32>
  %0 = arith.addf %a, %c : tensor<4xf32>
  %1 = arith.mulf %0, %a : tensor<4xf32>
  return %1, %c : tensor<4xf32>, tensor<4xf32>
}

// -----

// Only constants, so the generic has no inputs and takes its shape from outs.
// CHECK-LABEL: func.func @all_constant_operands(
// CHECK-NOT:     dense<
// CHECK:         %[[INIT:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-NOT:       ins(
// CHECK-SAME:      outs(%[[INIT]] : tensor<4xf32>)
// CHECK:           arith.addf
// CHECK-NEXT:      arith.mulf
// CHECK:         return %[[R]]
func.func @all_constant_operands() -> tensor<4xf32> {
  %c1 = arith.constant dense<1.0> : tensor<4xf32>
  %c2 = arith.constant dense<2.0> : tensor<4xf32>
  %0 = arith.addf %c1, %c2 : tensor<4xf32>
  %1 = arith.mulf %0, %c2 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// A non splat constant cannot become one scalar, so it stays a regular input.
// CHECK-LABEL: func.func @non_splat_constant(
// CHECK:         %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %[[CST]] : tensor<2xf32>, tensor<2xf32>)
func.func @non_splat_constant(%a: tensor<2xf32>) -> tensor<2xf32> {
  %c = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = arith.addf %a, %c : tensor<2xf32>
  %1 = arith.mulf %0, %a : tensor<2xf32>
  return %1 : tensor<2xf32>
}

// -----

// One op with a splat is still a group of one: the constant does not count.
// CHECK-LABEL: func.func @single_op_with_splat(
// CHECK-NOT:     linalg.generic
// CHECK:         arith.constant dense<2.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    arith.addf %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK-NOT:     linalg.generic
func.func @single_op_with_splat(%a: tensor<4xf32>) -> tensor<4xf32> {
  %c = arith.constant dense<2.0> : tensor<4xf32>
  %0 = arith.addf %a, %c : tensor<4xf32>
  return %0 : tensor<4xf32>
}
