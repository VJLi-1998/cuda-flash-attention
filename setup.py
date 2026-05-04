import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["TORCH_CUDA_ARCH_LIST"] = ""  # bypass CUDA version check

HERE = os.path.abspath(os.path.dirname(__file__))

def get_compute_capability():
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        return f"{major}{minor}"
    except Exception:
        return "89"

compute_cap = get_compute_capability()

try:
    import torch.utils.cpp_extension
    torch.utils.cpp_extension._check_cuda_version = lambda *a, **k: None
except Exception:
    pass

setup(
    name="flash_attn",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="flash_attn._C",
            sources=[
                "flash_attn/csrc/flash_attn.cu",
                "flash_attn/csrc/bindings.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-Wno-deprecated-declarations"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    f"-arch=compute_{compute_cap}",
                    f"-code=sm_{compute_cap}",
                    "-maxrregcount=64",
                    "--use_fast_math",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
