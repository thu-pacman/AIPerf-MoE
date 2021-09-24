# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import subprocess
import os
import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDA_HOME

# Setting this param to a list has a problem of generating
# different compilation commands (with diferent order of architectures)
# and leading to recompilation of fused kernels.
# set it to empty string to avoid recompilation
# and assign arch flags explicity in extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
        raise RuntimeError("Cuda extensions are being compiled with a version of Cuda that does " +
                           "not match the version used to compile Pytorch binaries.  " +
                           "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda) +
                           "In some cases, a minor-version mismatch will not cause later errors:  " +
                           "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
                           "You can try commenting out this check (at your own risk).")

def create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

cmdclass = {}
ext_modules = []

check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ['-DVERSION_GE_1_1']
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ['-DVERSION_GE_1_3']
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ['-DVERSION_GE_1_5']
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

srcpath = pathlib.Path(__file__).parent.absolute()
buildpath = srcpath / 'build'
cc_flag = []
_, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
if int(bare_metal_major) >= 11:
    cc_flag.append('-gencode')
    cc_flag.append('arch=compute_80,code=sm_80')

create_build_dir(buildpath)

amp_C = cpp_extension.load(name='amp_C',
                        sources=[srcpath / 'amp_C_frontend.cpp',
                                srcpath / 'multi_tensor_sgd_kernel.cu',
                                srcpath / 'multi_tensor_scale_kernel.cu',
                                srcpath / 'multi_tensor_axpby_kernel.cu',
                                srcpath / 'multi_tensor_l2norm_kernel.cu',
                                srcpath / 'multi_tensor_l2norm_scale_kernel.cu',
                                srcpath / 'multi_tensor_lamb_stage_1.cu',
                                srcpath / 'multi_tensor_lamb_stage_2.cu',
                                srcpath / 'multi_tensor_adam.cu',
                                srcpath / 'multi_tensor_adagrad.cu',
                                srcpath / 'multi_tensor_novograd.cu',
                                srcpath / 'multi_tensor_lamb.cu'],
                        build_directory=buildpath,
                        extra_cflags=['-O3'] + version_dependent_macros,
                        extra_cuda_cflags=['-lineinfo',
                                            '-O3',
                                            # '--resource-usage',
                                            '--use_fast_math'] + version_dependent_macros)

fused_layer_norm_cuda = cpp_extension.load(name='fused_layer_norm_cuda',
                                           sources=[srcpath / 'layer_norm_cuda.cpp',
                                                    srcpath / 'layer_norm_cuda_kernel.cu'],
                                           build_directory=buildpath,
                                           extra_cflags=['-O3'] + version_dependent_macros,
                                           extra_cuda_cflags=['-maxrregcount=50',
                                                            '-O3',
                                                            '--use_fast_math'] + version_dependent_macros)