"""Build SM120 FlashAttention as a torch extension."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sm120_fmha",
    ext_modules=[
        CUDAExtension(
            "sm120_fmha",
            ["sm120_launch.cu", "sm120_fmha_fwd.cu"],
            extra_compile_args={
                "nvcc": [
                    "-gencode=arch=compute_120f,code=sm_120f",
                    "-O3", "-std=c++17",
                    "--expt-relaxed-constexpr",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
