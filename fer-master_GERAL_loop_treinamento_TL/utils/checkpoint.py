import os
import shutil
from typing import Optional, Dict, Any

import torch

def diretorio_checkpoint(hps: Dict[str, Any]) -> str:
    return hps['model_save_dir']

def caminho_checkpoint(hps: Dict[str, Any], epoch_or_name) -> str:
    d = diretorio_checkpoint(hps)
    if isinstance(epoch_or_name, int):
        return os.path.join(d, f"epoch_{epoch_or_name}")
    if isinstance(epoch_or_name, str):
        if epoch_or_name in {"best", "last"}:
            return os.path.join(d, f"{epoch_or_name}.pth")
        return os.path.join(d, epoch_or_name)
    raise TypeError("epoch_or_name must be int or str ('best'/'last').")

def copia_segura(src: str, dst: str) -> None:
    try:
        # Remove arquivo/link anterior, se existir
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
    except Exception:
        pass
    try:
        # Tenta symlink (rápido). Se não puder, copia o arquivo.
        os.symlink(src, dst)
    except Exception:
        shutil.copyfile(src, dst)


def save(
    logger,
    net: torch.nn.Module,
    hps: Dict[str, Any],
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    is_best: bool = False,
    make_last_alias: bool = True,
) -> str:
    out_path = caminho_checkpoint(hps, epoch)
    payload = {
        "logs": logger.get_logs() if hasattr(logger, "get_logs") else None,
        "params": net.state_dict(),
    }
    # Acrescenta estados, se fornecidos
    if optimizer is not None:
        try:
            payload["optimizer_state"] = optimizer.state_dict()
        except Exception:
            pass
    if scheduler is not None:
        try:
            # Alguns schedulers (ReduceLROnPlateau) têm .state_dict()
            payload["scheduler_state"] = scheduler.state_dict()
        except Exception:
            pass

    torch.save(payload, out_path)

    if is_best:
        copia_segura(out_path, caminho_checkpoint(hps, "best"))
    if make_last_alias:
        copia_segura(out_path, caminho_checkpoint(hps, "last"))

    return out_path


def restore(
    net: torch.nn.Module,
    logger,
    hps: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    restore_epoch = hps.get("restore_epoch", None)
    if restore_epoch is None:
        print("[restore] 'restore_epoch' não definido. Treino iniciará do zero.")
        return

    path = caminho_checkpoint(hps, restore_epoch)
    if not os.path.exists(path):
        print(f"[restore] Checkpoint não encontrado: {path}. Iniciando do zero.")
        # Mantém comportamento antigo: força start_epoch=0
        hps["start_epoch"] = 0
        return

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[restore] Falha ao carregar checkpoint '{path}': {e}")
        hps["start_epoch"] = 0
        return

    logs = checkpoint.get("logs", None)
    if logs is not None and hasattr(logger, "restore_logs"):
        try:
            logger.restore_logs(logs)
        except Exception as e:
            print(f"[restore] Aviso: não foi possível restaurar logs: {e}")

    try:
        net.load_state_dict(checkpoint["params"])
    except Exception as e:
        print(f"[restore] Erro ao carregar 'params' no modelo: {e}")
        hps["start_epoch"] = 0
        return

    if optimizer is not None and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print(f"[restore] Aviso: não foi possível restaurar optimizer: {e}")

    if scheduler is not None and "scheduler_state" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        except Exception as e:
            print(f"[restore] Aviso: não foi possível restaurar scheduler: {e}")

    print("[restore] Network Restored!")


def load_features(model: torch.nn.Module, params: Dict[str, torch.Tensor]) -> None:
    own_state = model.state_dict()
    loaded = 0
    for name, p in params.items():
        if name in own_state and own_state[name].shape == p.shape:
            own_state[name].copy_(p)
            loaded += 1
    for p in model.parameters():
        p.requires_grad = False
    print(f"[load_features] Camadas carregadas: {loaded}")
