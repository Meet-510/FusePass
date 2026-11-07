// RUN: not %fusepass-opt %s --fusepass-fold-transpose-matmul 2>&1 | FileCheck %s

// The fold relies on this upstream behaviour: a transpose cannot hide the static
// sizes of its source behind dynamic ones. If it could, reading the tensor<4x3>
// directly would give the matmul a contraction size of 4 against B's 5.
// linalg.transpose refuses such a type, so this input never reaches the pass.
// Should a later MLIR accept it, this test fails and the verifier guard in
// verifiesAfterFold needs a real test.

// CHECK: error: 'linalg.transpose' op dim(result, 0) = {{.*}} doesn't match dim(input, permutation[0]) = 3
func.func @hidden_static_size(%x: tensor<4x3xf32>, %y: tensor<5x7xf32>, %init: tensor<?x?xf32>, %out: tensor<?x7xf32>) -> tensor<?x7xf32> {
  %t = linalg.transpose ins(%x : tensor<4x3xf32>) outs(%init : tensor<?x?xf32>) permutation = [1, 0]
  %r = linalg.matmul ins(%t, %y : tensor<?x?xf32>, tensor<5x7xf32>) outs(%out : tensor<?x7xf32>) -> tensor<?x7xf32>
  return %r : tensor<?x7xf32>
}
