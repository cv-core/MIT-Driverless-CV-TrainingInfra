# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension

# also requires torch and torchvision as well but assuming they are already installed

requirements = ["tqdm", "requests", "imgaug", "numpy", "requests", "pillow", "tensorboardX",
                "google-cloud-storage", "retrying", "optuna", "pymysql"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu

    extra_compile_args = {"cxx": []}
    define_macros = []

    extension = CUDAExtension
    sources += source_cuda
    define_macros += [("WITH_CUDA", None)]
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "_C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="mit-dut-yolov3",
    version="0.1",
    author="MIT Driverless",
    url="https://github.com/DUT-Racing/DUT18D_PerceptionCV/vectorized_yolov3",
    description="",
    # packages=find_packages(exclude=("configs", "tests",)),
    install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
