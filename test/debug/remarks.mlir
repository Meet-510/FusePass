// RUN: %fusepass-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(fusepass-elementwise-fusion{emit-remarks=true}))" -o %t

// Remarks are the debug channel that also works on release builds of LLVM.
// --verify-diagnostics fails on any remark not listed here, and on any listed
// remark that does not appear.

// The root reports the group. Its producer is not a root and stays quiet.
func.func @fused_chain(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b : tensor<4xf32>
  // expected-remark @below {{fused group of 2 ops}}
  %1 = arith.mulf %0, %b : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// The value with two users is its own root, and a group of one. So are both of
// its users, and each one reports why it stayed.
func.func @two_users(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // expected-remark @below {{not fused: group has only one op}}
  %x = arith.addf %a, %b : tensor<4xf32>
  // expected-remark @below {{not fused: group has only one op}}
  %p = arith.mulf %x, %a : tensor<4xf32>
  // expected-remark @below {{not fused: group has only one op}}
  %q = arith.subf %x, %b : tensor<4xf32>
  return %p, %q : tensor<4xf32>, tensor<4xf32>
}

// -----

func.func @outside_allowlist(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // expected-remark @below {{not fused: op is not in the allowlist}}
  %x = arith.remf %a, %b : tensor<4xf32>
  return %x : tensor<4xf32>
}

// -----

func.func @unranked(%a: tensor<*xf32>, %b: tensor<*xf32>) -> tensor<*xf32> {
  // expected-remark @below {{not fused: result is not a ranked tensor}}
  %x = arith.addf %a, %b : tensor<*xf32>
  return %x : tensor<*xf32>
}

// -----

func.func @type_changing(%a: tensor<4xf16>) -> tensor<4xf32> {
  // expected-remark @below {{not fused: op is not in the allowlist}}
  %x = arith.extf %a : tensor<4xf16> to tensor<4xf32>
  return %x : tensor<4xf32>
}

// -----

// Scalars are not candidates, so they get no remark at all.
func.func @scalars(%a: f32, %b: f32) -> f32 {
  %x = arith.addf %a, %b : f32
  %y = arith.mulf %x, %a : f32
  return %y : f32
}
