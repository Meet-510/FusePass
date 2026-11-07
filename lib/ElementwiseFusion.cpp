//===- ElementwiseFusion.cpp - Fuse elementwise tensor ops ----------------===//
//
// Fuses groups of elementwise ops on ranked tensors into one linalg.generic.
// The legality rules are listed in docs/DESIGN.md and enforced by
// whyNotFusable and canJoinGroup below.
//
//===----------------------------------------------------------------------===//

#include "FusePass/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::fusepass {

#define GEN_PASS_DEF_ELEMENTWISEFUSION
#include "FusePass/Passes.h.inc"

namespace {

using Group = llvm::SmallSetVector<Operation *, 8>;

/// Ops whose scalar form is the same op with scalar operand and result types.
/// Division on integers is left out because it is not Pure (it can trap).
bool isAllowlisted(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp, math::ExpOp,
             math::SqrtOp, math::AbsFOp, arith::AddIOp, arith::SubIOp,
             arith::MulIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
             arith::MaxSIOp, arith::MinSIOp>(op);
}

/// Checks the per-op legality rules. Returns the reason an op cannot be part of
/// any group, or nullptr if it can.
const char *whyNotFusable(Operation *op) {
  if (!isAllowlisted(op))
    return "op is not in the allowlist";
  if (!isPure(op))
    return "op is not side effect free";
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type)
    return "result is not a ranked tensor";
  if (type.getEncoding())
    return "result tensor has an encoding";
  if (llvm::any_of(op->getOperandTypes(),
                   [&](Type operand) { return operand != type; }))
    return "operand types differ from the result type";
  return nullptr;
}

/// Tests whether `producer` may be pulled into the group rooted at `root`.
bool canJoinGroup(Operation *producer, Operation *root, const Group &group) {
  // Rule: never fuse across a block or region boundary.
  if (producer->getBlock() != root->getBlock())
    return false;
  if (whyNotFusable(producer))
    return false;
  // Rule: every op in a group has the same result tensor type.
  if (producer->getResult(0).getType() != root->getResult(0).getType())
    return false;
  // Rule: every user of the producer must be in the group. Otherwise erasing
  // the producer would leave the outside user without a value.
  return llvm::all_of(producer->getUsers(), [&](Operation *user) {
    return group.contains(user);
  });
}

/// Walks backwards from `root`, adding producers until a fixed point. One pass
/// is not enough: a value with two users can only join after both users have.
Group collectGroup(Operation *root) {
  Group group;
  group.insert(root);
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < group.size(); ++i) {
      for (Value operand : group[i]->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (producer && !group.contains(producer) &&
            canJoinGroup(producer, root, group)) {
          group.insert(producer);
          changed = true;
        }
      }
    }
  }
  return group;
}

/// Returns true if some transitive user's group would contain `op`. Such an op
/// is not a root: it gets fused when the pattern fires on that user. Asking
/// this at every op makes groups independent of the driver's visiting order.
bool isAbsorbedByConsumer(Operation *op) {
  SmallVector<Operation *> worklist(op->getUsers());
  llvm::DenseSet<Operation *> visited;
  while (!worklist.empty()) {
    Operation *user = worklist.pop_back_val();
    if (!visited.insert(user).second)
      continue;
    if (user->getBlock() != op->getBlock() || whyNotFusable(user))
      continue;
    if (collectGroup(user).contains(op))
      return true;
    llvm::append_range(worklist, user->getUsers());
  }
  return false;
}

/// Replaces the group with one linalg.generic at the root's position.
void rewriteGroup(const Group &group, PatternRewriter &rewriter) {
  Operation *root = group[0];
  Location loc = root->getLoc();
  auto resultType = cast<RankedTensorType>(root->getResult(0).getType());

  // Clones run in block order so that each one finds its operands mapped.
  SmallVector<Operation *> ordered(group.begin(), group.end());
  llvm::sort(ordered, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });

  // Operands defined outside the group become inputs, once per distinct value.
  llvm::SmallSetVector<Value, 8> inputs;
  for (Operation *op : ordered)
    for (Value operand : op->getOperands())
      if (!group.contains(operand.getDefiningOp()))
        inputs.insert(operand);

  int64_t rank = resultType.getRank();
  SmallVector<AffineMap> maps(
      inputs.size() + 1,
      AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType());

  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{resultType}, inputs.getArrayRef(),
      ValueRange{init}, maps, iterators,
      [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
        IRMapping mapping;
        for (auto [input, arg] : llvm::zip(inputs, args))
          mapping.map(input, arg);
        // clone() keeps every attribute, such as fastmath and overflow flags.
        for (Operation *op : ordered) {
          Operation *scalar = builder.clone(*op, mapping);
          scalar->getResult(0).setType(resultType.getElementType());
        }
        linalg::YieldOp::create(builder, bodyLoc,
                                mapping.lookup(root->getResult(0)));
      });

  rewriter.replaceOp(root, generic.getResults());
  // Users come after producers, so reverse order erases each op after its
  // last user is gone.
  for (Operation *op : llvm::reverse(ordered))
    if (op != root)
      rewriter.eraseOp(op);
}

/// Anchors on every elementwise op and rewrites the ones that are group roots.
struct FuseElementwiseOps : OpTraitRewritePattern<OpTrait::Elementwise> {
  using Base::Base;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (const char *reason = whyNotFusable(op))
      return rewriter.notifyMatchFailure(op, reason);
    if (isAbsorbedByConsumer(op))
      return rewriter.notifyMatchFailure(op, "op fuses into a consumer's group");
    if (!cast<RankedTensorType>(op->getResult(0).getType()).hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes are unsupported");

    Group group = collectGroup(op);
    // Rule: a group of one gains nothing from a linalg.generic.
    if (group.size() < 2)
      return rewriter.notifyMatchFailure(op, "group has only one op");

    rewriteGroup(group, rewriter);
    return success();
  }
};

struct ElementwiseFusionPass
    : impl::ElementwiseFusionBase<ElementwiseFusionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseElementwiseOps>(&getContext());

    // This pass only applies its own pattern. Folding would rewrite the input
    // (x * 1.0 becomes x) before fusion could see it.
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
