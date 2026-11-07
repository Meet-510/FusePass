//===- FoldTransposeIntoMatmul.cpp - Fold transposes into matmul ----------===//
//
// Rewrites
//
//   %t = linalg.transpose ins(%a) outs(%init) permutation = [1, 0]
//   %r = linalg.matmul ins(%t, %b) outs(%c)
//
// so that the matmul reads %a directly through swapped indexing maps. MLIR 23
// has no linalg.matmul_transpose_a, so indexing_maps is the only spelling, and
// it can express a transpose on both operands at once.
//
// The shape rule is in docs/DESIGN.md and enforced by getFoldableTranspose and
// verifiesAfterFold below.
//
//===----------------------------------------------------------------------===//

#include "FusePass/Passes.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fusepass"

namespace mlir::fusepass {

#define GEN_PASS_DEF_FOLDTRANSPOSEINTOMATMUL
#include "FusePass/Passes.h.inc"

namespace {

/// Number of input operands of a matmul, A and B.
constexpr unsigned kNumMatmulInputs = 2;

/// Returns the transpose defining `operand` if it swaps the two dimensions of a
/// rank 2 tensor, and a null op otherwise.
linalg::TransposeOp getFoldableTranspose(Value operand) {
  auto transpose = operand.getDefiningOp<linalg::TransposeOp>();
  if (!transpose)
    return {};
  auto sourceType = dyn_cast<RankedTensorType>(transpose.getInput().getType());
  if (!sourceType || sourceType.getRank() != 2)
    return {};
  if (transpose.getPermutation() != ArrayRef<int64_t>{1, 0})
    return {};
  return transpose;
}

/// Matmul allows mixed element types through its cast. Folding one operand
/// would not change that, but this pass only promises the plain case.
bool hasUniformElementType(linalg::MatmulOp matmul) {
  Type elementType = getElementTypeOrSelf(matmul.getDpsInits()[0]);
  return llvm::all_of(matmul.getDpsInputs(), [&](Value input) {
    return getElementTypeOrSelf(input) == elementType;
  });
}

/// Swaps the two results of a rank 2 operand's map, which reads the operand's
/// transpose: (d0, d2) becomes (d2, d0).
AffineMap swapResults(AffineMap map) {
  assert(map.getNumResults() == 2 && "expected the map of a rank 2 operand");
  return AffineMap::get(map.getNumDims(), /*symbolCount=*/0,
                        {map.getResult(1), map.getResult(0)}, map.getContext());
}

/// Tries the fold on a detached clone, because a pattern must not touch the IR
/// when it goes on to return failure. In MLIR 23 this guard cannot fail:
/// linalg.transpose verifies that its result is exactly the swapped source
/// shape, so each loop sees the same sizes before and after the fold
/// (test/transpose/transpose_keeps_exact_shape.mlir pins that down). The guard
/// stays so that a relaxed verifier upstream cannot turn this into a bad rewrite.
bool verifiesAfterFold(linalg::MatmulOp matmul, ArrayRef<Value> newInputs,
                       ArrayAttr newMaps) {
  OwningOpRef<linalg::MatmulOp> candidate(matmul.clone());
  for (auto [index, input] : llvm::enumerate(newInputs))
    candidate->getOperation()->setOperand(index, input);
  candidate->setIndexingMapsAttr(newMaps);

  // A failed trial is an expected outcome, not an error for the user.
  ScopedDiagnosticHandler silence(matmul.getContext(), [](Diagnostic &diag) {
    LLVM_DEBUG(llvm::dbgs() << "  verifier: " << diag.str() << "\n");
    return success();
  });
  return succeeded(verify(candidate->getOperation(), /*verifyRecursively=*/false));
}

struct FoldTransposeIntoMatmulPattern : OpRewritePattern<linalg::MatmulOp> {
  FoldTransposeIntoMatmulPattern(MLIRContext *context,
                                 Pass::Statistic &numFolded)
      : OpRewritePattern(context), numFolded(numFolded) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "[fusepass] matmul: " << matmul << "\n");

    if (!matmul.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(matmul, "not on tensors");
    if (!hasUniformElementType(matmul))
      return rewriter.notifyMatchFailure(matmul, "element types differ");

    SmallVector<Value> newInputs(matmul.getDpsInputs());
    SmallVector<Attribute> newMaps(matmul.getIndexingMaps().getValue());
    llvm::SmallSetVector<Operation *, 2> transposes;
    for (unsigned i = 0; i < kNumMatmulInputs; ++i) {
      linalg::TransposeOp transpose = getFoldableTranspose(newInputs[i]);
      if (!transpose)
        continue;
      newInputs[i] = transpose.getInput();
      newMaps[i] = AffineMapAttr::get(
          swapResults(cast<AffineMapAttr>(newMaps[i]).getValue()));
      transposes.insert(transpose);
    }
    if (transposes.empty())
      return rewriter.notifyMatchFailure(matmul, "no foldable transpose");

    ArrayAttr newMapsAttr = rewriter.getArrayAttr(newMaps);
    if (!verifiesAfterFold(matmul, newInputs, newMapsAttr)) {
      LLVM_DEBUG(llvm::dbgs() << "  rejected: shapes do not allow the fold\n");
      return rewriter.notifyMatchFailure(matmul, "shapes do not allow the fold");
    }

    unsigned foldedOperands = 0;
    rewriter.modifyOpInPlace(matmul, [&] {
      for (unsigned i = 0; i < kNumMatmulInputs; ++i) {
        if (matmul->getOperand(i) == newInputs[i])
          continue;
        matmul->setOperand(i, newInputs[i]);
        ++foldedOperands;
      }
      matmul.setIndexingMapsAttr(newMapsAttr);
    });
    numFolded += foldedOperands;
    LLVM_DEBUG(llvm::dbgs() << "  folded " << foldedOperands << " operand(s)\n");

    // A transpose with other users stays for them.
    for (Operation *transpose : transposes)
      if (transpose->use_empty())
        rewriter.eraseOp(transpose);
    return success();
  }

private:
  Pass::Statistic &numFolded;
};

struct FoldTransposeIntoMatmulPass
    : impl::FoldTransposeIntoMatmulBase<FoldTransposeIntoMatmulPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldTransposeIntoMatmulPattern>(&getContext(),
                                                 numTransposesFolded);

    // Same policy as the fusion pass: apply this pattern and nothing else.
    GreedyRewriteConfig config;
    config.enableFolding(false).enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::fusepass
