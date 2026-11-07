//===- ElementwiseFusion.cpp - Fuse elementwise tensor ops ----------------===//
//
// Fuses groups of elementwise ops on ranked tensors into one linalg.generic.
// The legality rules are listed in docs/DESIGN.md and enforced by
// whyNotFusable and canJoinGroup below.
//
//===----------------------------------------------------------------------===//

#include "FusePass/Passes.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fusepass"

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

/// Returns the reason `producer` may not join the group rooted at `root`, or
/// nullptr if it may.
const char *whyNotJoinable(Operation *producer, Operation *root,
                           const Group &group) {
  // Rule: never fuse across a block or region boundary.
  if (producer->getBlock() != root->getBlock())
    return "producer is in a different block";
  if (const char *reason = whyNotFusable(producer))
    return reason;
  // Rule: every op in a group has the same result tensor type.
  if (producer->getResult(0).getType() != root->getResult(0).getType())
    return "producer has a different result type";
  // Rule: every user of the producer must be in the group. Otherwise erasing
  // the producer would leave the outside user without a value.
  if (!llvm::all_of(producer->getUsers(),
                    [&](Operation *user) { return group.contains(user); }))
    return "producer has a user outside the group";
  return nullptr;
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
            !whyNotJoinable(producer, root, group)) {
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

/// Logs why each producer of the group stayed out, which is what you want to
/// see when a chain did not fuse as far as expected.
void logGroupBoundary(const Group &group) {
  LLVM_DEBUG({
    Operation *root = group[0];
    for (Operation *member : group) {
      for (Value operand : member->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (!producer || group.contains(producer))
          continue;
        const char *reason = whyNotJoinable(producer, root, group);
        llvm::dbgs() << "  leaf from " << producer->getName() << ": "
                     << (reason ? reason : "joinable") << "\n";
      }
    }
  });
}

/// Returns the scalar value of a splat constant tensor, or a null attribute if
/// `value` is anything else.
TypedAttr getSplatConstant(Value value) {
  DenseElementsAttr attr;
  if (!value.getDefiningOp<arith::ConstantOp>() ||
      !matchPattern(value, m_Constant(&attr)) || !attr.isSplat())
    return {};
  return attr.getSplatValue<TypedAttr>();
}

/// Creates the tensor.empty for the result, reading dynamic sizes from a leaf.
/// Every leaf has the result's type, so any of them has the right sizes.
Value createInit(PatternRewriter &rewriter, Location loc, RankedTensorType type,
                 ValueRange inputs) {
  SmallVector<Value> dynamicSizes;
  for (int64_t dim = 0, rank = type.getRank(); dim < rank; ++dim)
    if (type.isDynamicDim(dim))
      dynamicSizes.push_back(
          tensor::DimOp::create(rewriter, loc, inputs.front(), dim));
  return tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                 type.getElementType(), dynamicSizes);
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
  // Splat constants become scalar constants in the body instead.
  llvm::SmallSetVector<Value, 8> inputs;
  llvm::SmallSetVector<Value, 4> splats;
  for (Operation *op : ordered) {
    for (Value operand : op->getOperands()) {
      if (group.contains(operand.getDefiningOp()))
        continue;
      if (getSplatConstant(operand))
        splats.insert(operand);
      else
        inputs.insert(operand);
    }
  }

  int64_t rank = resultType.getRank();
  SmallVector<AffineMap> maps(
      inputs.size() + 1,
      AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  Value init = createInit(rewriter, loc, resultType, inputs.getArrayRef());

  auto generic = linalg::GenericOp::create(
      rewriter, loc, TypeRange{resultType}, inputs.getArrayRef(),
      ValueRange{init}, maps, iterators,
      [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
        IRMapping mapping;
        for (auto [input, arg] : llvm::zip(inputs, args))
          mapping.map(input, arg);
        for (Value splat : splats)
          mapping.map(splat, arith::ConstantOp::create(
                                 builder, bodyLoc, getSplatConstant(splat)));
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
  // last user is gone. The driver removes constants that end up unused.
  for (Operation *op : llvm::reverse(ordered))
    if (op != root)
      rewriter.eraseOp(op);
}

/// What the pass wants to hear about its pattern's decisions.
struct FusionReporting {
  Pass::Statistic &fusedGroups;
  Pass::Statistic &fusedOps;
  Pass::Statistic &rejectedCandidates;
  bool emitRemarks;
};

/// Anchors on every elementwise op and rewrites the ones that are group roots.
struct FuseElementwiseOps : OpTraitRewritePattern<OpTrait::Elementwise> {
  FuseElementwiseOps(MLIRContext *context, FusionReporting report)
      : OpTraitRewritePattern(context), report(report) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Scalar and vector arith ops carry the same trait but are not candidates.
    if (llvm::none_of(op->getResultTypes(), llvm::IsaPred<TensorType>))
      return rewriter.notifyMatchFailure(op, "no result is a tensor");

    LLVM_DEBUG(llvm::dbgs() << "[fusepass] candidate: " << *op << "\n");

    if (const char *reason = whyNotFusable(op))
      return reject(op, reason, rewriter);

    // Not a rejection: this op is accepted when its consumer's group is built.
    if (isAbsorbedByConsumer(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  not a root, a consumer's group absorbs it\n");
      return rewriter.notifyMatchFailure(op,
                                         "op fuses into a consumer's group");
    }

    Group group = collectGroup(op);
    logGroupBoundary(group);
    // Rule: a group of one gains nothing from a linalg.generic.
    if (group.size() < 2)
      return reject(op, "group has only one op", rewriter);

    LLVM_DEBUG({
      llvm::dbgs() << "  accepted group of " << group.size() << " ops:\n";
      for (Operation *member : group)
        llvm::dbgs() << "    " << *member << "\n";
    });
    if (report.emitRemarks)
      op->emitRemark() << "fused group of " << group.size() << " ops";
    rewriteGroup(group, rewriter);
    ++report.fusedGroups;
    report.fusedOps += group.size();
    return success();
  }

private:
  LogicalResult reject(Operation *op, const char *reason,
                       PatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "  rejected: " << reason << "\n");
    // The driver revisits ops whose neighbours changed. Report each op once.
    if (rejected.insert(op).second) {
      ++report.rejectedCandidates;
      if (report.emitRemarks)
        op->emitRemark() << "not fused: " << reason;
    }
    return rewriter.notifyMatchFailure(op, reason);
  }

  FusionReporting report;
  // One pattern instance serves one pass run on one function, and patterns are
  // applied through a const method.
  mutable llvm::DenseSet<Operation *> rejected;
};

struct ElementwiseFusionPass
    : impl::ElementwiseFusionBase<ElementwiseFusionPass> {
  using ElementwiseFusionBase::ElementwiseFusionBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseElementwiseOps>(
        &getContext(), FusionReporting{numFusedGroups, numFusedOps,
                                       numRejectedCandidates, emitRemarks});

    // This pass only applies its own pattern. Folding would rewrite the input
    // (x * 1.0 becomes x) before fusion could see it.
    GreedyRewriteConfig config;
    config.enableFolding(false).enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::fusepass
