# Building and testing FusePass

FusePass needs LLVM and MLIR 23, Clang, CMake, Ninja and lit. It never copies LLVM
source into this repository. Pick a route:

| Route | Debug output | Status |
|---|---|---|
| [A. MSYS2 on Windows](#a-msys2-on-windows) | Remarks and IR dumps only | **Tested.** Steps 4-5 and "Running things" were repeated in a fresh build directory. Steps 1-3 were done once, during setup |
| [B. Source build with assertions](#b-source-build-with-assertions) | Adds `--debug-only` and statistics | **Not completed.** Windows Smart App Control blocked it. Linux steps are untested |
| [C. apt.llvm.org on Ubuntu](#c-aptllvmorg-on-ubuntu) | Remarks and IR dumps only | The steps the CI workflow runs. **Not run until the repository is pushed** |

## A. MSYS2 on Windows

No admin rights or reboot needed. This gives clang and MLIR 23.1.5 in one folder, and
you can remove it by deleting that folder.

**1. Download and unpack MSYS2** from `repo.msys2.org`. The installer is 41 MB, and
about 4 GB is used after step 2. In PowerShell:

```powershell
curl.exe -LO https://repo.msys2.org/distrib/x86_64/msys2-base-x86_64-20260927.sfx.exe
.\msys2-base-x86_64-20260927.sfx.exe -y -o"$env:USERPROFILE"
```

**2. Open the CLANG64 shell** by running `%USERPROFILE%\msys64\clang64.exe`, then update
it. The first command closes the window when it finishes, so reopen it and run the
second:

```bash
pacman -Syuu --noconfirm
```

```bash
pacman -Syuu --noconfirm
```

Then install the tools:

```bash
pacman -S --needed --noconfirm git mingw-w64-clang-x86_64-{clang,lld,llvm,llvm-libs,llvm-tools,mlir,cmake,ninja,python,python-pip}
```

**3. Install lit** in a virtual environment:

```bash
python -m venv ~/lit-venv && ~/lit-venv/bin/python -m pip install lit
```

**4. Configure.** From the repository root:

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DMLIR_DIR=/clang64/lib/cmake/mlir -DFUSEPASS_LIT="$HOME/lit-venv/bin/lit"
```

Configure prints what it found:

```text
-- FusePass compiler: Clang 23.1.5 (C:/Users/you/msys64/clang64/bin/clang++.exe)
-- FusePass LLVM/MLIR: 23.1.5 from C:/Users/you/msys64/clang64/lib/cmake/mlir
-- FusePass LLVM build: assertions=OFF rtti=ON eh=OFF
-- FusePass plugin: OFF (MLIR libraries are STATIC_LIBRARY)
```

**5. Build and test:**

```bash
ninja -C build check-fusepass
```

Expected result: 11 pass and 3 skip. The three need the plugin, statistics or
`--debug-only`, which this build does not have.

**Smart App Control.** If it is on, Windows may refuse programs built on your machine
with `WinError 4551: An Application Control policy has blocked this file`. See
[common errors](#common-errors).

## B. Source build with assertions

`--debug-only` and `--mlir-pass-statistics` need `LLVM_ENABLE_ASSERTIONS=ON`, and
packaged LLVM builds usually have it off. Check with:

```bash
llvm-config --assertion-mode
```

**This route has not been completed.** On the development machine, configure worked
(`mlir-opt` needs 3,988 build steps), then Smart App Control blocked the first
freshly built tool. The commands below are that configuration, and nothing beyond
configure has been run. They are written for Linux or WSL2, which Smart App Control
does not govern.

Pinned release: `llvmorg-23.1.5`. The sparse checkout has to include `libc`, because
LLVM's CMake requires part of it.

```bash
git clone --depth 1 --branch llvmorg-23.1.5 --filter=blob:none --sparse https://github.com/llvm/llvm-project.git ~/llvm-src
```

```bash
git -C ~/llvm-src sparse-checkout set llvm mlir cmake third-party libc
```

Set `LLVM_PARALLEL_COMPILE_JOBS` for your RAM: allow about 2 GB per job. Keep link jobs
at 1 on small machines.

```bash
cmake -G Ninja -S ~/llvm-src/llvm -B ~/llvm-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_PARALLEL_COMPILE_JOBS=4 \
  -DLLVM_PARALLEL_LINK_JOBS=1
```

```bash
ninja -C ~/llvm-build mlir-opt FileCheck not count mlir-tblgen llvm-config
```

Then configure FusePass against that build tree. You do not have to install it:

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DMLIR_DIR=$HOME/llvm-build/lib/cmake/mlir -DFUSEPASS_LIT="$(command -v lit)"
```

The configure output should then say `assertions=ON`, and `check-fusepass` should run
all 14 tests. That result has not been seen yet.

## C. apt.llvm.org on Ubuntu

These are the steps in `.github/workflows/ci.yml`, for Ubuntu 24.04. The package names
were checked against the real repository index. There is no `mlir-23-dev`: the names
are `libmlir-23-dev` and `mlir-23-tools`.

```bash
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
```

```bash
sudo add-apt-repository -y "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-23 main"
```

```bash
sudo apt-get update && sudo apt-get install -y ninja-build zlib1g-dev libzstd-dev libz3-dev clang-23 llvm-23-dev llvm-23-tools libmlir-23-dev mlir-23-tools
```

```bash
pip install lit
```

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-23 -DCMAKE_CXX_COMPILER=clang++-23 -DFUSEPASS_LIT="$(command -v lit)" -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir
```

```bash
cmake --build build --target check-fusepass
```

apt.llvm.org ships a rolling 23.x, 23.1.6 at the time of checking, rather than exactly
23.1.5.

## Running things

These work once a route above has been built. The examples use the driver. Where the
plugin works, use `mlir-opt --load-pass-plugin=build/lib/FusePassPlugin.so` in its place.

Add `build/bin` to your `PATH` first:

```bash
export PATH="$PWD/build/bin:$PATH"
```

**The whole suite:**

```bash
ninja -C build check-fusepass
```

**One test.** Point lit at the test's path under the build directory, not the source
tree. The file does not exist there, and lit maps it back:

```bash
lit -v build/test/elementwise/fuse_chain.mlir
```

**A pass by hand:**

```bash
fusepass-opt examples/fuse_chain.mlir --fusepass-elementwise-fusion
```

```bash
fusepass-opt examples/fold_transpose.mlir --fusepass-fold-transpose-matmul
```

**Both passes:**

```bash
fusepass-opt examples/fold_transpose.mlir --fusepass-pipeline
```

**Remarks,** which work on every build:

```bash
fusepass-opt examples/fuse_chain.mlir -o /dev/null --mlir-disable-threading --pass-pipeline='builtin.module(func.func(fusepass-elementwise-fusion{emit-remarks=true}))'
```

**IR dumps around each pass:**

```bash
fusepass-opt examples/fold_transpose.mlir --fusepass-fold-transpose-matmul --mlir-print-ir-after-all -o /dev/null
```

```bash
fusepass-opt examples/fold_transpose.mlir --fusepass-fold-transpose-matmul --mlir-print-ir-before-all -o /dev/null
```

**Debug output and statistics.** These need route B. Threading is off so that lines from
different functions do not interleave:

```bash
fusepass-opt examples/fuse_chain.mlir --fusepass-elementwise-fusion --debug-only=fusepass --mlir-disable-threading -o /dev/null
```

```bash
fusepass-opt examples/fuse_chain.mlir --fusepass-pipeline --mlir-pass-statistics -o /dev/null
```

**Formatting:**

```bash
clang-format -i lib/*.cpp include/FusePass/Passes.h tools/fusepass-opt/*.cpp
```

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `WinError 4551: An Application Control policy has blocked this file`, or `Permission denied`, or exit code `3236495362` from a tool | Smart App Control is refusing an unsigned program. `Get-MpComputerStatus` shows `SmartAppControlState`, and the Code Integrity event log names the file | This is your security setting to decide. Use WSL2, which it does not govern, or read Microsoft's guidance on turning it off, including whether you can turn it back on |
| `Unknown command line argument '--debug-only=fusepass'` | LLVM has no assertions | Use route B, or `emit-remarks` |
| `--mlir-pass-statistics` prints only the header | Release LLVM: `LLVM_ENABLE_STATS` is off | Use route B, or `emit-remarks` |
| `'FileCheck': command not found` in lit | `FileCheck` missing from the install (on apt it is in `llvm-23-tools`), or the environment lost `PATHEXT` on Windows, which lit needs to find `.exe` files | Install the package, or `export PATHEXT=".COM;.EXE;.BAT;.CMD"` |
| `'fusepass-elementwise-fusion' does not refer to a registered pass` with `--load-pass-plugin` | The plugin loaded but `mlir-opt` has its own copy of MLIR. See [DESIGN.md](DESIGN.md) | Use `fusepass-opt` |
| `Failed to load passes from '...'. Request ignored.` | Wrong plugin path, or a plugin built for a different LLVM. The plugin API checks the version | Check the path, and rebuild against the `mlir-opt` you are running |
| `plugin unavailable, using the fusepass-opt driver` in lit output | Not an error. lit tried the plugin and fell back | See above if you expected the plugin |
| `lit not found: check-fusepass will fail` at configure | No `lit` on `PATH` | `pip install lit`, or pass `-DFUSEPASS_LIT=/path/to/lit` |
| `undefined reference to typeinfo for mlir::...` at link | RTTI mismatch between this project and LLVM. FusePass includes `HandleLLVMOptions` to copy LLVM's flags, so this means you overrode them. Not hit during development | Remove `-frtti` or `-fno-rtti` from your own flags, and use a clean build directory |
| `Could not find a package configuration file provided by "MLIR"` | CMake cannot find MLIR | Set `-DMLIR_DIR=` to the directory holding `MLIRConfig.cmake` |
| `LLVM libc is not found` when configuring LLVM | The sparse checkout left out `libc` | Add `libc` to `sparse-checkout set` |
