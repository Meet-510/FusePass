// RUN: %fusepass-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(fusepass-fold-transpose-matmul{emit-remarks=true}))" -o /dev/null

func.func @fold_one(%x: tensor<3x2xf32>, %y: tensor<3x4xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %init = tensor.empty() : tensor<2x3xf32>
  %t = linalg.transpose ins(%x : tensor<3x2xf32>) outs(%init : tensor<2x3xf32>) permutation = [1, 0]
  // expected-remark @below {{folded 1 transposed operand(s)}}
  %r = linalg.matmul ins(%t, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

func.func @fold_two(%x: tensor<2x2xf32>, %out: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %init = tensor.empty() : tensor<2x2xf32>
  %t = linalg.transpose ins(%x : tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) permutation = [1, 0]
  // expected-remark @below {{folded 2 transposed operand(s)}}
  %r = linalg.matmul ins(%t, %t : tensor<2x2xf32>, tensor<2x2xf32>) outs(%out : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// -----

// A matmul with nothing to fold stays quiet.
func.func @nothing_to_fold(%x: tensor<2x3xf32>, %y: tensor<3x4xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %r = linalg.matmul ins(%x, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}
