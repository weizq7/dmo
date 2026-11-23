"""Portable build/cache and Python bindings for the generation CUDA kernels.

Compilation and graph capture happen before benchmark timing. The libraries
use the CUDA runtime ABI, not the PyTorch C++ ABI.
"""

from __future__ import annotations

import copy
import ctypes as C
import fcntl
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig

import torch
from torch import nn

SOURCES = Path(__file__).with_name("csrc")


def require_cuda_environment():
    for key in ("CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_DEVICE_MAX_COPY_CONNECTIONS"):
        if os.environ.get(key) != "32":
            raise RuntimeError(f"Set {key}=32 before CUDA initialization")
    if torch.__version__ != "2.8.0+cu128":
        raise RuntimeError(
            "Generation requires the verified torch==2.8.0 CUDA 12.8 build"
        )


@lru_cache(maxsize=1)
def compiler_config():
    """Locate nvcc and the CUDA runtime shipped with the active Python env."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda_home) / "bin/nvcc") if cuda_home else shutil.which("nvcc")
    if not nvcc and Path("/usr/local/cuda/bin/nvcc").is_file():
        nvcc = "/usr/local/cuda/bin/nvcc"
    if not nvcc or not Path(nvcc).is_file():
        raise RuntimeError("CUDA Toolkit 12.6 or 12.8 with nvcc is required; set CUDA_HOME")
    version = subprocess.check_output([nvcc, "--version"], text=True)
    release = re.search(r"release (\d+\.\d+),", version)
    if not release or release.group(1) not in {"12.6", "12.8"}:
        raise RuntimeError(
            "Use CUDA Toolkit 12.6 or 12.8; nvidia-smi's CUDA Version is not the toolkit version"
        )
    runtime = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/lib"
    if not (runtime / "libcudart.so.12").is_file():
        raise RuntimeError(
            "Missing nvidia-cuda-runtime-cu12 from the CUDA PyTorch installation"
        )
    return nvcc, runtime, version


@lru_cache(maxsize=None)
def build_library(name):
    if name not in {"exit_control", "depth_scheduler", "rms_norm", "rotary"}:
        raise ValueError(f"Unknown CUDA library: {name}")
    nvcc, runtime, version = compiler_config()
    source = SOURCES / f"{name}.cu"
    flags = ["-O3", "--fmad=false", "--ftz=false", "-std=c++17", "-arch=sm_80"]
    contents = source.read_bytes()
    if name == "depth_scheduler":
        contents += (SOURCES / "exit_control.cu").read_bytes()
    key = hashlib.sha256(
        contents + repr(flags).encode() + version.encode() + str(runtime).encode()
    ).hexdigest()[:20]
    cache = Path(
        os.environ.get("DMO_KERNEL_CACHE", Path.home() / ".cache/dmo/kernels")
    ).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{name}_{key}.so"
    with (cache / "compile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.is_file():
            temporary = target.with_suffix(f".{os.getpid()}.tmp.so")
            try:
                subprocess.run(
                    [
                        nvcc,
                        "-shared",
                        "-Xcompiler=-fPIC",
                        *flags,
                        "--cudart",
                        "shared",
                        str(source),
                        "-I" + str(runtime.parent / "include"),
                        "-L" + str(runtime),
                        "-Xlinker",
                        "-rpath",
                        "-Xlinker",
                        str(runtime),
                        "-o",
                        str(temporary),
                    ],
                    check=True,
                )
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return target


def _graph_library(name):
    lib = C.CDLL(str(build_library(name)))
    u, i, up = C.c_uint64, C.c_int, C.POINTER(C.c_uint64)
    lib.ce_error.restype = C.c_char_p
    lib.ce_launch.argtypes = [u, u]
    lib.ce_launch.restype = i
    lib.ce_destroy.argtypes = [u]
    lib.ce_destroy.restype = None
    if name == "exit_control":
        lib.ce_make_fg.argtypes = [i, i, i, i, up, up, up, u, u]
        lib.ce_make_fg.restype = u
        lib.ce_copy_kv.argtypes = [u, u, u, u, i, i, i]
        lib.ce_copy_kv.restype = i
    else:
        lib.sd_make_foreground.argtypes = [i, i, i, up, up, up, up, u, u]
        lib.sd_make_foreground.restype = u
        lib.sd_make_worker.argtypes = [i, i, i, i, up, up, u]
        lib.sd_make_worker.restype = u
        lib.sd_create_stream.argtypes = []
        lib.sd_create_stream.restype = u
        lib.sd_destroy_stream.argtypes = [u]
        lib.sd_destroy_stream.restype = i
    return lib


@lru_cache(maxsize=1)
def load_control_library():
    return _graph_library("exit_control")


@lru_cache(maxsize=1)
def load_depth_library():
    return _graph_library("depth_scheduler")


@lru_cache(maxsize=1)
def load_norm_library():
    lib = C.CDLL(str(build_library("rms_norm")))
    lib.launch_full_rmsnorm.argtypes = [
        C.c_uint64,
        C.c_uint64,
        C.c_uint64,
        C.c_int,
        C.c_int,
        C.c_float,
        C.c_uint64,
    ]
    lib.launch_full_rmsnorm.restype = C.c_int
    return lib


@lru_cache(maxsize=1)
def load_rope_library():
    lib = C.CDLL(str(build_library("rotary")))
    lib.launch_exact_rope.argtypes = [C.c_uint64] * 4 + [C.c_int] * 3 + [C.c_uint64]
    lib.launch_exact_rope.restype = C.c_int
    return lib


def pointers(values):
    return (C.c_uint64 * len(values))(
        *[x.data_ptr() if hasattr(x, "data_ptr") else int(x) for x in values]
    )


def check_result(lib, result):
    if not result:
        raise RuntimeError("CUDA graph construction: " + lib.ce_error().decode())
    return result


def check_call(lib, code):
    if code:
        raise RuntimeError("CUDA graph operation: " + lib.ce_error().decode())


class FusedRMSNorm(nn.Module):
    """Single-token RMSNorm preserving PyTorch 2.8's FP32 reduction order."""

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.library = load_norm_library()

    def forward(self, x):
        weight = self.original.weight
        if (
            torch.is_grad_enabled()
            or not x.is_cuda
            or x.ndim != 3
            or x.shape[:2] != (1, 1)
            or x.dtype not in (torch.float32, torch.bfloat16)
            or not x.is_contiguous()
            or weight.dtype != torch.float32
            or not weight.is_contiguous()
            or weight.device != x.device
            or x.shape[-1] < 32
            or x.shape[-1] > 4096
            or x.shape[-1] % 32
            or x.shape[-1] != weight.numel()
            or not torch.is_autocast_enabled("cuda")
        ):
            raise ValueError(
                "Fused RMSNorm requires one contiguous CUDA token, width 32..4096, and BF16 autocast"
            )
        out = torch.empty_like(x, dtype=torch.float32)
        error = self.library.launch_full_rmsnorm(
            x.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            x.shape[-1],
            int(x.dtype == torch.bfloat16),
            self.original.eps,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if error:
            raise RuntimeError(f"Fused RMSNorm launch failed: {error}")
        return out


def exact_rope(x, cos, sin, lib=None):
    if (
        torch.is_grad_enabled()
        or not x.is_cuda
        or x.ndim != 4
        or x.shape[0] != 1
        or x.shape[2] != 1
        or x.shape[-1] % 2
        or not x.is_contiguous()
        or x.dtype not in (torch.float32, torch.bfloat16)
        or cos.dtype != torch.float32
        or sin.dtype != torch.float32
        or cos.shape != (1, x.shape[-1] // 2)
        or sin.shape != cos.shape
        or cos.device != x.device
        or sin.device != x.device
        or not cos.is_contiguous()
        or not sin.is_contiguous()
    ):
        raise ValueError(
            "Fused RoPE requires one contiguous CUDA token and FP32 half-width coefficients"
        )
    lib = lib or load_rope_library()
    out = torch.empty_like(x, dtype=torch.float32)
    error = lib.launch_exact_rope(
        x.data_ptr(),
        cos.data_ptr(),
        sin.data_ptr(),
        out.data_ptr(),
        x.numel(),
        x.shape[-1],
        int(x.dtype == torch.bfloat16),
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if error:
        raise RuntimeError(f"Fused RoPE launch failed: {error}")
    return out


def bf16_linear_view(module):
    """Copy containers and Linear operands once, leaving the source model intact."""
    view = copy.copy(module)
    view._modules = {
        name: bf16_linear_view(child) for name, child in module._modules.items()
    }
    if isinstance(module, nn.Linear):
        view._parameters = dict(module._parameters)
        for name in ("weight", "bias"):
            value = getattr(module, name)
            if value is not None:
                view._parameters[name] = nn.Parameter(
                    value.detach().to(torch.bfloat16), requires_grad=False
                )
    return view


def make_decode_view(model):
    """Both foreground and background capture the same immutable weight view."""
    view = copy.copy(model)
    view._modules = dict(model._modules)
    layers = copy.copy(model.layers)
    layers._modules = {}
    for key, block in model.layers._modules.items():
        block_view = copy.copy(block)
        block_view._modules = dict(block._modules)
        for name in (
            "pre_attn_norm",
            "post_attn_norm",
            "pre_ffn_norm",
            "post_ffn_norm",
        ):
            block_view._modules[name] = FusedRMSNorm(getattr(block, name))
        layers._modules[key] = block_view
    view._modules["layers"] = bf16_linear_view(layers)
    if model.use_grt:
        for name in ("grt_input_proj", "grt_gate"):
            view._modules[name] = bf16_linear_view(getattr(model, name))
    return view
