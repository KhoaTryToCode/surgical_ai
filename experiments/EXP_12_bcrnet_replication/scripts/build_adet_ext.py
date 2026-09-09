"""
Build & Installation Script for AdelaiDet C++/CUDA Extension (_C)
================================================================
Compiles the Multi-Scale Deformable Attention CUDA / C++ operations
into `adet._C` with automatic modern PyTorch (2.4+) compatibility patching.

Features:
- Non-destructive staged build: source files in `repos/BCRNet/adet/layers/csrc/`
  are copied to an isolated `.build_csrc/` staging directory before patching,
  keeping `repos/BCRNet` 100% clean and immutable.
- Patches deprecated `value.type()` to modern `value.scalar_type()` and
  `value.type().is_cuda()` to `value.is_cuda()` for PyTorch 2.1 - 2.6+ / CUDA 12+.

Usage:
  python experiments/EXP_12_bcrnet_replication/scripts/build_adet_ext.py
"""

import os
import sys
import shutil
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def patch_modern_pytorch_compatibility(staging_dir: str):
    """
    Patches modern PyTorch 2.x breaking changes in staged C++/CUDA files:
    1. AT_DISPATCH_FLOATING_TYPES(value.type(), ...) -> (value.scalar_type(), ...)
    2. tensor.type().is_cuda() -> tensor.is_cuda()
    """
    for root, _, files in os.walk(staging_dir):
        for fname in files:
            if fname.endswith((".cu", ".cpp", ".h", ".cuh")):
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                modified = content
                # Patch 1: tensor.type().is_cuda() -> tensor.is_cuda()
                modified = modified.replace(".type().is_cuda()", ".is_cuda()")
                # Patch 2: AT_DISPATCH_FLOATING_TYPES(value.type() -> AT_DISPATCH_FLOATING_TYPES(value.scalar_type()
                modified = modified.replace("AT_DISPATCH_FLOATING_TYPES(value.type(),", "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),")

                if modified != content:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(modified)
                    print(f"   🔧 Applied PyTorch 2.4+ compatibility patch to: {fname}")


def build_extension():
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    csrc_orig = os.path.join(ws_root, "repos/BCRNet/adet/layers/csrc")
    staging_dir = os.path.join(os.path.dirname(__file__), "../.build_csrc")
    bcrnet_dir = os.path.join(ws_root, "repos/BCRNet")

    if not os.path.exists(csrc_orig):
        raise FileNotFoundError(f"csrc directory not found at: {csrc_orig}. Ensure repos/BCRNet is cloned.")

    print(f"📦 Staging C++/CUDA files from {csrc_orig} -> {staging_dir}")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    shutil.copytree(csrc_orig, staging_dir)

    # Patch staged files for PyTorch 2.4+ (leaving repos/BCRNet pristine)
    patch_modern_pytorch_compatibility(staging_dir)

    print(f"⚙️ Compiling adet._C extension...")
    print(f"   CUDA Available: {torch.cuda.is_available()}")

    sources = [
        os.path.join(staging_dir, "vision.cpp"),
        os.path.join(staging_dir, "DeformAttn/ms_deform_attn_cpu.cpp"),
    ]

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    if torch.cuda.is_available():
        sources.append(os.path.join(staging_dir, "cuda_version.cu"))
        sources.append(os.path.join(staging_dir, "DeformAttn/ms_deform_attn_cuda.cu"))
        define_macros.append(("WITH_CUDA", None))
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        ext_cls = CUDAExtension
    else:
        print("⚠️ Compiling in CPU-only mode (CUDA not available).")
        ext_cls = CppExtension

    include_dirs = [staging_dir, os.path.join(staging_dir, "DeformAttn")]

    ext = ext_cls(
        name="adet._C",
        sources=sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )

    from setuptools import setup
    old_argv = sys.argv
    sys.argv = [
        "setup.py",
        "build_ext",
        f"--build-lib={bcrnet_dir}",
        "--inplace",
    ]
    try:
        setup(
            name="adet_extension",
            ext_modules=[ext],
            cmdclass={"build_ext": BuildExtension},
        )
        print("✅ Successfully built adet._C extension in repos/BCRNet!")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    build_extension()
