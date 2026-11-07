//===- Passes.cpp - FusePass registration ---------------------------------===//

#include "FusePass/Passes.h"

namespace mlir::fusepass {

#define GEN_PASS_REGISTRATION
#include "FusePass/Passes.h.inc"

// registerFusePassPasses() comes from the TableGen output above.
void registerFusePass() { registerFusePassPasses(); }

} // namespace mlir::fusepass
