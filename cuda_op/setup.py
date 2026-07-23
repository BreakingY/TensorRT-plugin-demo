from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension



setup(name="centerpool_cuda",
    ext_modules=[
        CUDAExtension(name="centerpool_cuda",sources=["centerpool.cpp","centerpool_kernel.cu"])
        ],
    cmdclass={"build_ext":BuildExtension}
)
