"""lit configuration for FusePass."""

import os
import shutil
import subprocess

import lit.formats

config.name = "FusePass"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.fusepass_obj_root, "test")

# Tools live next to each other in the LLVM install. The build tree comes first
# so that fusepass-opt and any DLLs it needs are found.
tool_path = [
    os.path.join(config.fusepass_obj_root, "bin"),
    config.llvm_tools_dir,
    config.mlir_tools_dir,
]
config.environment["PATH"] = os.pathsep.join(tool_path + [config.environment["PATH"]])


def find_tool(name):
    return shutil.which(name, path=config.environment["PATH"])


def plugin_loads(mlir_opt, plugin):
    """Returns true when mlir_opt really sees the passes from the plugin.

    Loading alone proves nothing: a DLL with its own copy of MLIR loads without
    complaint and registers its passes in a registry mlir-opt never reads.
    """
    if not (mlir_opt and plugin and os.path.exists(plugin)):
        return False
    probe = subprocess.run(
        [mlir_opt, "--load-pass-plugin=" + plugin, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=dict(os.environ, PATH=config.environment["PATH"]),
    )
    return "fusepass-elementwise-fusion" in probe.stdout


# Pick the entry point. Every RUN line uses %fusepass-opt, so each test runs
# through mlir-opt with the plugin wherever the platform allows it.
mlir_opt = find_tool("mlir-opt")
if plugin_loads(mlir_opt, config.fusepass_plugin):
    config.available_features.add("plugin")
    opt_args = [mlir_opt, "--load-pass-plugin=" + config.fusepass_plugin]
    lit_config.note("using mlir-opt with the FusePass plugin: " + mlir_opt)
else:
    opt_args = [find_tool("fusepass-opt")]
    lit_config.note("plugin unavailable, using the fusepass-opt driver")
config.substitutions.append(
    ("%fusepass-opt", " ".join('"{}"'.format(a) for a in opt_args).replace("\\", "/"))
)


def statistics_work():
    """Returns true when --mlir-pass-statistics reports real values.

    MLIR only records them under LLVM_ENABLE_STATS, which release builds of LLVM
    turn off unless they set LLVM_FORCE_ENABLE_STATS. Asking the tool covers
    both cases.
    """
    report = subprocess.run(
        opt_args
        + [
            os.path.join(config.fusepass_src_root, "examples", "fuse_chain.mlir"),
            "--fusepass-elementwise-fusion",
            "--mlir-pass-statistics",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=dict(os.environ, PATH=config.environment["PATH"]),
    )
    return "fused-groups" in report.stdout


stats_recorded = statistics_work()
if stats_recorded:
    config.available_features.add("stats")
lit_config.note("pass statistics recorded: " + str(stats_recorded))

# Tests that need --debug-only are marked REQUIRES: asserts because LLVM_DEBUG
# output only exists in LLVM builds with assertions.
llvm_config = find_tool("llvm-config")
if llvm_config:
    mode = subprocess.check_output(
        [llvm_config, "--assertion-mode"],
        universal_newlines=True,
        env=dict(os.environ, PATH=config.environment["PATH"]),
    ).strip()
    if mode == "ON":
        config.available_features.add("asserts")
    lit_config.note("llvm-config --assertion-mode: " + mode)

# lit substitutes %s without quoting, and a checkout path may contain spaces.
# Config substitutions run before the built-in ones, so this wraps the built-in
# %s in quotes and keeps the RUN lines conventional.
config.substitutions.insert(0, ("%s\\b", '"%s"'))
