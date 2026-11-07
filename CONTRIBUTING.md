# Contributing

1. Build with one of the routes in [docs/BUILD_AND_TEST.md](docs/BUILD_AND_TEST.md).
2. Run `ninja -C build check-fusepass` before committing.
3. Format C++ with `clang-format -i` using the repository's `.clang-format`.
4. Use conventional commit messages: `feat:`, `fix:`, `test:`, `docs:`, `ci:`.

**Tests.** Every behaviour change needs a FileCheck test under `test/`.

- A test for a fused case asserts one `linalg.generic` and that none of the original
  ops remain.
- A test for a case that does not fuse asserts that the original ops are still there.
- Capture SSA names with `[a-z0-9_]+` if arguments can have the same type. `.*` is
  greedy and will swallow the next argument.
- Tests needing `--debug-only` need `REQUIRES: asserts`. Tests needing statistics need
  `REQUIRES: stats`.

**New allowlist ops.** Only add an op whose scalar form is the same op with scalar
types, and that is `Pure`. Update the list in [docs/DESIGN.md](docs/DESIGN.md).

**MLIR API.** It changes between releases, so check signatures against the headers of
the version you build with.
