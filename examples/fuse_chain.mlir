// Run with:
//   fusepass-opt examples/fuse_chain.mlir --fusepass-elementwise-fusion

// |a + b| * 2 fuses, with the splat constant moved into the loop body.
func.func @scale_sum(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %two = arith.constant dense<2.0> : tensor<4xf32>
  %sum = arith.addf %a, %b : tensor<4xf32>
  %abs = math.absf %sum : tensor<4xf32>
  %out = arith.mulf %abs, %two : tensor<4xf32>
  return %out : tensor<4xf32>
}

// Dynamic shapes read their sizes from an input through tensor.dim.
func.func @dynamic(%a: tensor<?x4xf32>, %b: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %sub = arith.subf %a, %b : tensor<?x4xf32>
  %out = arith.mulf %sub, %sub : tensor<?x4xf32>
  return %out : tensor<?x4xf32>
}

// %x has two users, so it stays materialized and nothing fuses across it.
func.func @two_users(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %x = arith.addf %a, %b : tensor<4xf32>
  %p = arith.mulf %x, %a : tensor<4xf32>
  %q = arith.subf %x, %b : tensor<4xf32>
  return %p, %q : tensor<4xf32>, tensor<4xf32>
}
