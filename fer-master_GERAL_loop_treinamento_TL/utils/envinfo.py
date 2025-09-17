# utils/envinfo.py
import os
import sys
import json
import platform
import subprocess
from typing import Dict, Any

import torch


def safe_run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=2)
        return out.strip()
    except Exception:
        return ""


def coletar_info_ambiente(hps: Dict[str, Any]) -> Dict[str, Any]:
    # Python
    py = sys.version.splitlines()[0]
    sysinfo = {
        "python": py,
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # Torch / CUDA / CUDNN
    torchinfo = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }

    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        torchinfo["gpus"].append({
            "index": i,
            "name": prop.name,
            "total_memory_mb": int(prop.total_memory // (1024 ** 2)),
            "capability": f"{prop.major}.{prop.minor}",
        })

    # NVIDIA-SMI
    smi = safe_run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    torchinfo["nvidia_smi"] = smi.splitlines() if smi else []

    return {
        "system": sysinfo,
        "torch": torchinfo
    }


def gravar_info_ambiente(hps: Dict[str, Any]) -> None:
    out_dir = hps.get("model_save_dir") or os.path.join(os.getcwd(), "checkpoints", hps.get("name", ""))
    os.makedirs(out_dir, exist_ok=True)

    info = coletar_info_ambiente(hps)

    # JSON
    try:
        with open(os.path.join(out_dir, "env.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    try:
        lines = []
        s, t = info["system"], info["torch"]
        lines.append("[System]")
        lines.append(f"Python: {s['python']}")
        lines.append(f"OS: {s['os_system']} {s['os_release']} ({s['machine']})")
        lines.append(f"Kernel/Version: {s['os_version']}")
        lines.append("")
        lines.append("[Torch/CUDA]")
        lines.append(f"torch: {t['torch_version']}")
        lines.append(f"cuda_available: {t['cuda_available']}")
        lines.append(f"torch.version.cuda: {t['torch_cuda_version']}")
        lines.append(f"cudnn_version: {t['cudnn_version']}")
        lines.append(f"gpu_count: {t['gpu_count']}")
        for g in t["gpus"]:
            lines.append(f"  GPU{g['index']}: {g['name']} ({g['total_memory_mb']} MB, cc {g['capability']})")
        if t["nvidia_smi"]:
            lines.append("nvidia-smi:")
            for line in t["nvidia_smi"]:
                lines.append(f"  {line}")
        with open(os.path.join(out_dir, "env.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass


def salvar_info_modelo(net: torch.nn.Module, hps: Dict[str, Any]) -> None:
    out_dir = hps.get("model_save_dir") or os.path.join(os.getcwd(), "checkpoints", hps.get("name", ""))
    os.makedirs(out_dir, exist_ok=True)

    try:
        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        info = {
            "arch": net.__class__.__name__,
            "network": hps.get("network", ""),
            "parameters_total": int(total),
            "parameters_trainable": int(trainable),
        }
        with open(os.path.join(out_dir, "modelinfo.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, "modelinfo.txt"), "w", encoding="utf-8") as f:
            f.write(f"arch: {info['arch']}\n")
            f.write(f"network: {info['network']}\n")
            f.write(f"parameters_total: {info['parameters_total']}\n")
            f.write(f"parameters_trainable: {info['parameters_trainable']}\n")
    except Exception:
        pass
