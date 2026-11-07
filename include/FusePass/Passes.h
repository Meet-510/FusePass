//===- Passes.h - FusePass pass entry points --------------------*- C++ -*-===//

#ifndef FUSEPASS_PASSES_H
#define FUSEPASS_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir::fusepass {

#define GEN_PASS_DECL
#include "FusePass/Passes.h.inc"

/// Registers every FusePass pass and pipeline with the global registry. Both
/// the plugin and the fusepass-opt driver call this.
void registerFusePass();

} // namespace mlir::fusepass

#endif // FUSEPASS_PASSES_H
