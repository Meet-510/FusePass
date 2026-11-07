# FusePass design

## Fusion legality rules

An op can join a group only if all of these hold. Each rule is enforced in
`lib/ElementwiseFusion.cpp` and has a test.

| # | Rule | Code | Test |
|---|---|---|---|
| 1 | Every use of a producer's result is inside the group. A value with two different users stays materialized. Two uses by the same consumer (`x * x`) are fine | `whyNotJoinable` | `no_fuse.mlir` `@two_users`, `fuse_chain.mlir` `@same_consumer_twice` and `@diamond` |
| 2 | Same block as the root | `whyNotJoinable` | `no_fuse.mlir` `@producer_in_other_region`, `@fuses_inside_region` |
| 3 | Same result tensor type as the root, and operands of that same type. Shape and element type both count. Ranked only, no encoding | `whyNotFusable`, `whyNotJoinable` | `no_fuse.mlir` `@unranked_tensor`, `@shape_changes_through_cast` |
| 4 | On the allowlist and `Pure` | `whyNotFusable` | `no_fuse.mlir` `@not_in_allowlist`, `@integer_division` |
| 5 | A group of one op is left alone | `FuseElementwiseOps` | `no_fuse.mlir` `@single_op`, `splat_constants.mlir` `@single_op_with_splat` |
| 6 | Anything else ends the chain, such as `tensor.extract_slice`, `linalg.matmul`, or a type changing op like `arith.extf`. Its result is a leaf | Follows from 3 and 4 | `no_fuse.mlir` `@broken_by_extract_slice`, `@broken_by_matmul`, `@type_changing_op` |

Function arguments have no defining op, so they are always leaves
(`@function_argument`).

### The allowlist

Each of these ops has a scalar form that is the same op with scalar types, which is
how the body is built:

- **Float:** `arith.addf`, `subf`, `mulf`, `divf`, `negf`, `maximumf`, `minimumf`,
  `math.exp`, `sqrt`, `absf`
- **Integer:** `arith.addi`, `subi`, `muli`, `andi`, `ori`, `xori`, `maxsi`, `minsi`

Integer division is left out on purpose. It can trap, so it is not `Pure`.

## Why rewrite patterns and the greedy driver

The rewrite is local: start at an op, look at its operands, and replace a small
piece of the function. That is what a `RewritePattern` expresses. Dialect
conversion is the wrong tool here. It is built around a target that declares some
ops illegal and a type converter, while here the same `arith.addf` is legal in one
place and fused in another, depending on its use list.

The driver is configured with `enableFolding(false)` and `enableConstantCSE(false)`.
This pass applies its own pattern and nothing else. With folding on, `x * 1.0`
would become `x` before fusion saw it, and tests would be checking upstream folders
instead of this pass. The driver still removes unused ops, which is not optional,
so a chain whose result is unused simply disappears.

## Why anchor at the root

A group has to be built exactly once, from one place. The pattern fires on every
elementwise op, so each op must be able to answer: "am I the root of my group?"

The definition is: an op is a root unless some transitive user's group would
contain it. `isAbsorbedByConsumer` computes that by walking forward through users
and collecting each one's group.

This makes groups a partition. If `x` were in the groups of two roots `r1` and `r2`,
every forward path from `x` stays inside each group until it reaches that group's
root, so one root would lie on the other's path. It would then be in the other's
group, and so not a root.

The partition does not depend on visiting order, and that was tested rather than
assumed. The driver defaults to bottom-up, which reaches roots first, so the check
rarely fires:

| Experiment | Result |
|---|---|
| Default traversal | 11 of 11 runnable tests pass |
| `setUseTopDownTraversal(true)` | 11 of 11 pass |
| Top-down with `isAbsorbedByConsumer` disabled | 2 tests fail |

Collection walks backwards and repeats until nothing changes. One sweep would be
wrong for a diamond, because a value with two users can only join once both users
have, and the walk may reach it through one user first.

The cost is roughly quadratic in the size of the chain, since each op may collect a
group. That is fine for small tensor programs, and a real compiler would cache it.

## How a group becomes a generic

`rewriteGroup` works in this order:

1. Sort the members into block order, so each clone finds its operands mapped.
2. Sort operands defined outside the group into inputs, deduplicated, and splat
   constants.
3. Create `tensor.empty` at the root's position, with one `tensor.dim` per dynamic
   size. Rule 3 means every leaf has the result's type, so the first input has the
   right sizes. A splat cannot have a dynamic type, so a group with dynamic shapes
   always has an input.
4. Create the `linalg.generic` with identity maps and all parallel iterators.
5. In the body, map inputs to block arguments and splats to scalar `arith.constant`
   ops, then clone each member and change its result type to the element type.
   Cloning is what keeps `fastmath` and `overflow` flags, and tests check that.
6. Replace the root and erase the rest in reverse order, so each op is erased after
   its last user.

Why `linalg.generic` as the target: it states the iteration and the body directly,
which is the point of fusion, and it is what later Linalg passes consume. Upstream
bufferization and tiling work on it without knowing FusePass exists.

## The transpose rule

"When the shapes allow it" means all of these, enforced in
`lib/FoldTransposeIntoMatmul.cpp`:

1. The operand is the direct result of a `linalg.transpose` on a rank 2 ranked
   tensor (`getFoldableTranspose`).
2. The permutation is exactly `[1, 0]` (`getFoldableTranspose`).
3. A, B and the output have the same element type (`hasUniformElementType`). Mixed
   types are legal for matmul, but this pass only promises the plain case.
4. The op still verifies after the change (`verifiesAfterFold`), tried on a detached
   clone because a pattern must not touch the IR if it then returns failure.

**Rule 4 cannot fail in MLIR 23**, and I learned that from a failing test. I had
written a test where a transpose hid static sizes behind `tensor<?x?xf32>`, and
`linalg.transpose` rejected it, because its verifier requires the result to be
exactly the swapped source shape. So each loop sees the same sizes before and after
the fold. `test/transpose/transpose_keeps_exact_shape.mlir` pins this: if upstream
relaxes it, that test fails, and the guard needs a real test. The shape cases that
actually reject a fold are therefore rank and permutation (`no_fold.mlir`
`@rank_three`, `@identity_permutation`).

**Spelling.** MLIR 23 has no `linalg.matmul_transpose_a`, confirmed in the installed
headers. `indexing_maps` is the only spelling, and it handles both operands at once.
The rewrite swaps the two results of that operand's map, turning `(d0, d2)` into
`(d2, d0)`. Because it swaps whatever map is there, two transposes in a row cancel
back to the default map (`@double_transpose`).

**Users of the transpose.** The fold is valid for this matmul regardless of who else
reads the transpose. It stays alive if it has other users and is erased if dead,
along with a `tensor.empty` that only it used.

## Why a plugin, and why also a driver

A plugin means no fork of `mlir-opt`: the passes load into the stock tool. That is
the main entry point where it works.

It works when `mlir-opt` and the plugin share one copy of MLIR. This was probed on
the real tools:

| Platform | How `mlir-opt` gets MLIR | Result |
|---|---|---|
| Windows, MSYS2 | Static, inside `mlir-opt.exe` (140 MB) | Tried: the DLL loads but its passes land in its own registry: `'hello-plugin' does not refer to a registered pass`. Linking `libMLIR.dll` instead fails, because it does not export `mlir::Pass` |
| Ubuntu, apt.llvm.org | `libMLIR.so.23.1`, read from the ELF `NEEDED` list | Expected to work. Not run yet; CI has a step that will fail if it does not |

Upstream's own standalone example also skips its plugin with `if(NOT WIN32)`.

So `fusepass-opt` exists: a few lines on `MlirOptMain`, with the same command line.
lit picks the entry point by checking whether `mlir-opt` really lists the plugin's
passes, because a plugin can load without error and still be invisible.

## Comparison with upstream

Upstream has `--convert-elementwise-to-linalg` followed by
`--linalg-fuse-elementwise-ops`. I ran both on the same input with `mlir-opt` 23.1.5:

| Input | Upstream | FusePass |
|---|---|---|
| `(a + b) * dense<2.0>` | One generic, constant `%cst` outside the body | One generic, constant inside the body |
| `@two_users` | Three generics, the shared value not fused | Unchanged `arith` ops |
| A single `arith.addf` | One generic | Unchanged |

Both decline to fuse the two-user value on this input.

**What upstream does better, and it is a lot:**

- It works on any `linalg.generic`, including non-identity maps, so it fuses through
  reshapes, and with ops that came from elsewhere. FusePass only sees allowlisted
  `arith` and `math` ops, and only identity maps.
- It has a control function so that a client decides the policy.
- It is maintained, tested at scale, and used by real compilers.

**What FusePass does differently:**

- One step, from `arith` on tensors to one fused op, with no generic in between.
- It leaves code alone when nothing fuses. Upstream's first pass converts every op
  whether or not it will fuse.
- A short legality story you can read in one file: six rules with named tests.

**Why write it anyway.** To learn the machinery by owning all of it: use lists, the
greedy driver, pattern legality, TableGen pass declarations and lit. For production
code, use the upstream passes.
