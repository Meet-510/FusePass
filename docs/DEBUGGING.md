# Debugging FusePass

Every output excerpt below is from a real run during development. Nothing here is
reconstructed.

## Which tool works on which LLVM build

This was checked against the MSYS2 MLIR 23.1.5 release packages rather than
assumed. Two of the four tools need more than a normal release build.

| Tool | Release LLVM | LLVM with assertions |
|---|---|---|
| `emit-remarks=true` pass option | Yes, tested | Yes |
| `--mlir-print-ir-after-all` and friends | Yes | Yes |
| `--mlir-pass-statistics` | **Empty report**, see below | Yes |
| `--debug-only=fusepass` | **Rejected as an unknown argument** | Yes |

```text
$ fusepass-opt examples/fold_transpose.mlir --fusepass-fold-transpose-matmul --debug-only=fusepass
fusepass-opt.exe: Unknown command line argument '--debug-only=fusepass'.
```

Statistics are the less obvious one. On a release build the report prints only its
header, because `llvm/ADT/Statistic.h` swaps in a no-op counter unless
`!defined(NDEBUG) || LLVM_FORCE_ENABLE_STATS`, and `mlir/lib/Pass/PassStatistics.cpp`
only collects values under the same condition. This is why the passes have the
`emit-remarks` option: it goes through the normal diagnostics engine, so it works
everywhere and `--verify-diagnostics` can test it.

```text
$ fusepass-opt examples/fuse_chain.mlir \
    --pass-pipeline='builtin.module(func.func(fusepass-elementwise-fusion{emit-remarks=true}))'
```

lit detects what the build supports and skips what it cannot run:

| lit feature | Set when | Tests that need it |
|---|---|---|
| `asserts` | `llvm-config --assertion-mode` prints `ON` | `test/debug/debug_output.mlir` |
| `stats` | a trial run reports `fused-groups` | `test/debug/statistics.mlir` |
| `plugin` | `mlir-opt --load-pass-plugin=... --help` lists the passes | `test/plugin/load_plugin.mlir` |

## How the debug text was obtained here

Development happened on Windows 11 with Smart App Control on. It blocked the
LLVM assertions build (`WinError 4551: An Application Control policy has blocked
this file`, on `llvm-min-tblgen.exe` loading a freshly built `libLLVMTableGen.dll`).
So on this machine `--debug-only=fusepass` itself **has not been run**.

To see the pass's real debug text anyway, the experiments below used a throwaway
copy of the source with `LLVM_DEBUG` expanded unconditionally. The log statements
are the same ones the pass ships with; only the gate differs. To repeat it:

```bash
cp -r . /tmp/fusepass-trace && cd /tmp/fusepass-trace
sed -i 's/LLVM_DEBUG(/FORCED_DEBUG(/' lib/ElementwiseFusion.cpp lib/FoldTransposeIntoMatmul.cpp
sed -i 's/^#define DEBUG_TYPE "fusepass"$/#define DEBUG_TYPE "fusepass"\n#define FORCED_DEBUG(...) do { __VA_ARGS__; } while (false)/' \
  lib/ElementwiseFusion.cpp lib/FoldTransposeIntoMatmul.cpp
```

Then configure and build that copy as usual. On an LLVM with assertions, use
`--debug-only=fusepass` on the normal build instead.

## Worked example 1: a broken legality rule, diagnosed from the debug text

This is the mutation check. It shows that the suite really does guard the rule
the project is built around: a value with two users must not fuse.

**The mutation.** In `whyNotJoinable` in `lib/ElementwiseFusion.cpp`, delete the
check that every user is in the group:

```diff
-  if (!llvm::all_of(producer->getUsers(), [&](Operation *user) {
-        return group.contains(user);
-      }))
-    return "producer has a user outside the group";
+  (void)group; // MUTATION: the all-uses rule is gone.
```

**The suite catches it.** Three tests that passed before now fail, and the rest of
the suite is unaffected:

```text
FAIL: FusePass :: elementwise/no_fuse.mlir
FAIL: FusePass :: debug/remarks.mlir
FAIL: FusePass :: elementwise/dynamic_shapes.mlir
Total Discovered Tests: 14
  Unsupported: 3 (21.43%)
  Passed     : 8 (57.14%)
  Failed     : 3 (21.43%)
```

Those are the three files with a value that has two users. `no_fuse.mlir` holds
`@two_users`, the case named on the resume. FileCheck only reports that its input
is empty, because the tool crashed first:

```text
# error: command failed with exit status: 0xc0000005
# | FileCheck error: '<stdin>' is empty.
```

**The remark already contradicts the rule.** `%q`, on line 4, has a shared operand,
yet the pass claims a group:

```text
two_users.mlir:4:8: remark: fused group of 2 ops
```

**The debug text shows the mechanism.** Running only that function:

```text
[fusepass] candidate: %2 = arith.subf %0, %arg1 : tensor<4xf32>
  accepted group of 2 ops:
    %2 = arith.subf %0, %arg1 : tensor<4xf32>
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
[fusepass] candidate: %0 = "arith.mulf"(<<NULL VALUE>>, %arg0) <{fastmath = #arith.fastmath<none>}> : (<<NULL TYPE>>, tensor<4xf32>) -> tensor<4xf32>
Exception Code: 0xC0000005
 #0 mlir::fusepass::(anonymous namespace)::whyNotFusable(mlir::Operation*)
 #1 mlir::fusepass::(anonymous namespace)::FuseElementwiseOps::matchAndRewrite(...)
```

Reading it from the top:

1. The group contains `arith.addf`, which also feeds `arith.mulf`. The rule should
   have stopped this, so the bug is in group collection, not in body construction.
2. The rewrite erases every group member. The next candidate, `arith.mulf`, now has
   `<<NULL VALUE>>` as its first operand: it still pointed at the erased `addf`.
3. The crash in `whyNotFusable` is only the symptom. It reads the operand types of
   that op and dereferences the null type.

**The fix.** Restore the four lines. With the rule in place, the same debug text
names the rule that kept the shared value out:

```text
[fusepass] candidate: %2 = arith.subf %0, %arg1 : tensor<4xf32>
  leaf from arith.addf: producer has a user outside the group
  rejected: group has only one op
[fusepass] candidate: %1 = arith.mulf %0, %arg0 : tensor<4xf32>
  leaf from arith.addf: producer has a user outside the group
  rejected: group has only one op
[fusepass] candidate: %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  rejected: group has only one op
```

## Worked example 2: the debug output disproved my own test

The first version of `test/debug/debug_output.mlir` expected `arith.addf` to be
logged before `arith.mulf`. Checked against the real debug text, it failed:

```text
test/debug/debug_output.mlir:6:11: error: CHECK: expected string not found in input
<stdin>:1:1: note: scanning from here
[fusepass] candidate: [fusepass] candidate: %2% = 1 = arith.subf arith.mulf% 0%,0 ,%arg1 %arg1 : tensor<:4 xtensor<f324>x
```

That output shows two separate problems:

1. **Garbled lines.** Two functions were logging at once, because the pass manager
   runs function passes in parallel. The fix is `--mlir-disable-threading` on the
   RUN line.
2. **Wrong order.** Once the lines were readable, `arith.mulf` came first. The
   greedy driver defaults to bottom-up traversal, so it reaches the root before
   its producers. The fix is correcting the expected order.

The second point raised a design question. Bottom-up order means the "not a root"
check rarely fires, so was it dead code? I tested that rather than guessing:

| Experiment | Result |
|---|---|
| `config.setUseTopDownTraversal(true)` | 11 of 11 runnable tests pass: grouping does not depend on visiting order |
| Top-down, with `isAbsorbedByConsumer` disabled | `debug/remarks.mlir` and `elementwise/fuse_chain.mlir` fail: the check is needed |

With top-down order the debug text shows the check doing its job:

```text
[fusepass] candidate: %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  not a root, a consumer's group absorbs it
[fusepass] candidate: %1 = arith.mulf %0, %arg1 : tensor<4xf32>
  accepted group of 2 ops:
```

## Other real failures along the way

| Symptom | Diagnosis | Fix |
|---|---|---|
| `'linalg.transpose' op dim(result, 0) = -9223372036854775808 doesn't match dim(input, permutation[0]) = 3` | I assumed a transpose could hide static sizes behind `tensor<?x?xf32>`. Its verifier demands exact sizes (`-9223372036854775808` is how `?` prints here) | Replaced the test with `transpose_keeps_exact_shape.mlir`, which pins this down, and corrected the comment on `verifiesAfterFold` |
| `CHECK-SAME ... with "X" equal to "arg0: tensor<?x?xf32>, %arg1"` | `%[[X:.*]]` is greedy and swallowed the next argument, since both have the same type | Capture with `[a-z0-9_]+` |
| `CHECK-NOT: excluded string found ... %2 = arith.addf %in, %cst_0 : f32` | My `CHECK-NOT: arith.addf` also matched the scalar op inside the body | Check the body with `CHECK-NEXT`, then forbid `arith.` afterwards |
| `error: expected remark "folded 1 transposed operand(s)" was not produced` | One test file ran both passes, but each pass only emits its own remarks | One file per pass |
| `debug/remarks.mlir` passes with the build directory on `C:\Users\...` but fails with `build/` inside `D:\Intel projects\FusePass` | Found by following BUILD_AND_TEST.md in a fresh directory. lit does not quote `%t`, so a space in the build path split `-o %t` in two | These tests never read the output, so use `-o /dev/null` |
| `--mlir-pass-statistics` prints an empty report | See the first section | `stats` lit feature and `emit-remarks` |
| `'hello-plugin' does not refer to a registered pass or pass pipeline` | On Windows the stock `mlir-opt.exe` links MLIR statically, so a plugin DLL registers into its own copy of the registry | `fusepass-opt` driver; lit probes the plugin before using it |
| `undefined symbol: vtable for mlir::Pass` linking against `libMLIR.dll` | That DLL does not export the C++ pass API | Same |
| `error: redefinition of 'registerFusePassPasses'` | `mlir-tblgen -name FusePass` already generates that function | Named the wrapper `registerFusePass` |
| `'_FILE_OFFSET_BITS' macro redefined` | `LLVM_DEFINITIONS` is one space separated string, so `add_definitions` glued the flags together | `separate_arguments` |
| `no matching constructor for initialization of ElementwiseFusionPass` | Once a pass declares options, TableGen generates a constructor taking them | `using ElementwiseFusionBase::ElementwiseFusionBase;` |
