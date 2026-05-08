#!/usr/bin/env bash
# Install flash-attn for the r1sgg conda env (matches current PyTorch).
#
# AutoDL: 若存在 /etc/network_turbo，默认会先 source（学术加速，含 GitHub）。
#   跳过：USE_NETWORK_TURBO=0 ./scripts/install_flash_attn_proxy.sh
#
# v2.8.3 的 GitHub Release 仅提供 torch2.4–2.8 等预编译 wheel；Torch 2.9+ 无对应 wheel 时会自动改为源码编译（较慢，需 CUDA 与编译环境）。
#
# Override:
#   FLASH_ATTN_WHEEL_PATH=/path/to/xxx.whl ./scripts/install_flash_attn_proxy.sh
#   FLASH_ATTN_WHEEL_URL=https://github.com/.../xxx.whl ./scripts/install_flash_attn_proxy.sh
#   MAX_JOBS=8  # 源码编译并行度
#   FLASH_ATTN_PIP_INDEX=https://pypi.org/simple  # 国内镜像常不含 flash-attn，源码安装请用官方 PyPI
#   TORCH_CUDA_ARCH_LIST=8.9   # 若未设置，脚本会从本机 GPU 推断；4090/L4=8.9，A100=8.0，H100=9.0
# 源码安装使用 --no-build-isolation：setup 阶段需要已安装的 torch。
set -euo pipefail

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-r1sgg}"
# 阿里云等镜像往往没有 flash-attn 包元数据；从源码装时必须走官方 PyPI。
FLASH_ATTN_PIP_INDEX="${FLASH_ATTN_PIP_INDEX:-https://pypi.org/simple}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if [[ "${USE_NETWORK_TURBO:-1}" != "0" ]] && [[ -f /etc/network_turbo ]]; then
  echo "Sourcing AutoDL /etc/network_turbo (GitHub / Hugging Face acceleration)"
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

# flash-attn 源码编译必须包含当前 GPU 的 compute capability，否则会报
# cudaErrorNoKernelImageForDevice / no kernel image is available for execution on the device
configure_torch_cuda_arch_for_flash_attn() {
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (already set)"
    export TORCH_CUDA_ARCH_LIST
    return
  fi
  TORCH_CUDA_ARCH_LIST="$(
    python - <<'PY'
import torch

if torch.cuda.is_available():
    caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    uniq = sorted({f"{a}.{b}" for a, b in caps})
    print(";".join(uniq))
else:
    # 无 GPU 时放宽（编译更慢）；有 GPU 时上面已覆盖本机架构
    print("7.5;8.0;8.6;8.9;9.0")
PY
  )"
  export TORCH_CUDA_ARCH_LIST
  echo "Set TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} for flash-attn build (export explicitly to override)"
}

install_flash_attn_from_source() {
  configure_torch_cuda_arch_for_flash_attn
  pip install --no-cache-dir ninja packaging wheel -i "${FLASH_ATTN_PIP_INDEX}"
  MAX_JOBS="${MAX_JOBS:-4}" pip install --no-cache-dir --no-build-isolation "flash-attn==2.8.3" -i "${FLASH_ATTN_PIP_INDEX}"
}

if [[ -n "${FLASH_ATTN_WHEEL_PATH:-}" ]]; then
  echo "Installing flash-attn from local wheel: ${FLASH_ATTN_WHEEL_PATH}"
  pip uninstall -y flash-attn 2>/dev/null || true
  pip install --no-cache-dir "${FLASH_ATTN_WHEEL_PATH}"
elif [[ -n "${FLASH_ATTN_WHEEL_URL:-}" ]]; then
  echo "Installing flash-attn from FLASH_ATTN_WHEEL_URL"
  pip uninstall -y flash-attn 2>/dev/null || true
  pip install --no-cache-dir "${FLASH_ATTN_WHEEL_URL}"
else
  WHEEL_URL="$(
    python - <<'PY'
import sys
import torch

def main() -> None:
    v = torch.__version__.split("+")[0]
    parts = v.split(".")
    if len(parts) < 2:
        print("", end="")
        return
    major, minor = int(parts[0]), int(parts[1])
    if major != 2 or minor < 4 or minor > 8:
        print("", end="")
        return
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    tv = f"{major}.{minor}"
    # v2.8.3 release assets: cu12 + torch2.4..2.8 + linux_x86_64
    url = (
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        f"flash_attn-2.8.3+cu12torch{tv}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
    )
    print(url, end="")


main()
PY
  )"

  pip uninstall -y flash-attn 2>/dev/null || true

  if [[ -n "${WHEEL_URL}" ]]; then
    echo "Trying prebuilt wheel for your Torch/Python ABI:"
    echo "  ${WHEEL_URL}"
    if pip install --no-cache-dir "${WHEEL_URL}"; then
      true
    else
      echo "Wheel install failed; falling back to building flash-attn from source (can take 15–40+ minutes)."
      install_flash_attn_from_source
    fi
  else
    echo "No official v2.8.3 prebuilt wheel for this PyTorch/CUDA combo; building flash-attn==2.8.3 from source."
    echo "Using pip index: ${FLASH_ATTN_PIP_INDEX} (国内默认镜像常无 flash-attn，勿删)"
    install_flash_attn_from_source
  fi
fi

python - <<'PY'
import importlib.util

import torch

import flash_attn  # noqa: F401

ok = importlib.util.find_spec("flash_attn") is not None
print("torch:", torch.__version__)
print("flash_attn importable:", ok)
raise SystemExit(0 if ok else 1)
PY
