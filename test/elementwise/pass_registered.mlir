// RUN: %fusepass-opt %s --fusepass-elementwise-fusion | FileCheck %s

// The pass is registered and keeps an empty function intact.
// CHECK-LABEL: func.func @empty
func.func @empty() {
  // CHECK-NEXT: return
  return
}
