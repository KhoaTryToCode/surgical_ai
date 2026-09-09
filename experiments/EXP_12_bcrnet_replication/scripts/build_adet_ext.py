"""
Build & Installation Script for AdelaiDet C++/CUDA Extension (_C)
================================================================
Compiles the Multi-Scale Deformable Attention CUDA / C++ operations
located in `repos/BCRNet/adet/layers/csrc/` into `adet._C`.

Usage:
  python experiments/EXP_12_bcrnet_replication/scripts/build_adet_ext.py
"""
import os
import sys
import glob
import shutil
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

def build_extension():
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    csrc_dir = os.path.join(ws_root, "repos/BCRNet/adet/layers/csrc")
    output_dir = os.path.join(ws_root, "repos/BCRNet/adet")

    if not os.path.exists(csrc_dir):
        raise FileNotFoundError(f"csrc directory not found at: {csrc_dir}")

    print(f"📦 Compiling adet._C extension from: {csrc_dir}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")

    sources = [
        os.path.join(csrc_dir, "vision.cpp"),
        os.path.join(csrc_dir, "DeformAttn/ms_deform_attn_cpu.cpp"),
    ]

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    if torch.cuda.is_available():
        sources.append(os.path.join(csrc_dir, "cuda_version.cu"))
        sources.append(os.path.join(csrc_dir, "DeformAttn/ms_deform_attn_cuda.cu"))
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

    include_dirs = [csrc_dir, os.path.join(csrc_dir, "DeformAttn")]

    ext = ext_cls(
        name="adet._C",
        sources=sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )

    from setuptools import setup
    # Setup distribution arguments to compile in-place inside repos/BCRNet
    old_argv = sys.argv
    sys.argv = [
        "setup.py",
        "build_ext",
        f"--build-lib={os.path.join(ws_root, 'repos/BCRNet')}",
        "--inplace",
    ]
    try:
        setup(
            name="adet_extension",
            ext_modules=[ext],
            cmdclass={"build_ext": BuildExtension},
        )
        print("✅ Successfully built adet._C extension!")
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    build_extension()
