"""Local compiler configuration for native Devito OpenMP GPU offload."""
import os
from pathlib import Path


def configure_compiler(operator):
    default = Path(__file__).resolve().parents[1] / 'profile_output/devito_gpu_20261029/cuda_toolchain'
    # Frozen run snapshots use the path recorded in their launch environment.
    toolkit = Path(os.environ.get('ENO_CUDA_PATH', default)).resolve()
    required = ['bin/ptxas', 'bin/nvlink', 'nvvm/libdevice/libdevice.10.bc']
    if any(not (toolkit / name).is_file() for name in required):
        raise RuntimeError('native Devito GPU compiler tools missing; set ENO_CUDA_PATH')
    os.environ['PATH'] = str(toolkit / 'bin') + ':/usr/lib/llvm-20/bin:' + os.environ['PATH']
    os.environ['OMP_TARGET_OFFLOAD'] = 'MANDATORY'
    operator._compiler.cflags += ['--cuda-path=' + str(toolkit), '-Wno-unknown-cuda-version']
    operator._compiler.ldflags += ['--cuda-path=' + str(toolkit), '-Wl,-rpath,/usr/lib/llvm-20/lib']
