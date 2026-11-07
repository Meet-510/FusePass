//===- ElementwiseFusion.cpp - Fuse elementwise tensor ops ----------------===//

#include "FusePass/Passes.h"

namespace mlir::fusepass {

#define GEN_PASS_DEF_ELEMENTWISEFUSION
#include "FusePass/Passes.h.inc"

namespace {

struct ElementwiseFusionPass
    : impl::ElementwiseFusionBase<ElementwiseFusionPass> {
  void runOnOperation() override {}
};

} // namespace
} // namespace mlir::fusepass
