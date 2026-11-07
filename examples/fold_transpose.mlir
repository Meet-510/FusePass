// Run with:
//   fusepass-opt examples/fold_transpose.mlir --fusepass-fold-transpose-matmul

// The transpose of %x folds into the matmul's indexing maps and disappears.
func.func @transpose_a_times_b(%x: tensor<3x2xf32>, %y: tensor<3x4xf32>, %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %init = tensor.empty() : tensor<2x3xf32>
  %t = linalg.transpose ins(%x : tensor<3x2xf32>) outs(%init : tensor<2x3xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t, %y : tensor<2x3xf32>, tensor<3x4xf32>) outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}
