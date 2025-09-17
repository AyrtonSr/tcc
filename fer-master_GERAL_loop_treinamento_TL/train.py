import sys
import warnings
import os
import time
import csv
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from datetime import datetime
from data.fer2013 import get_dataloaders
from utils.checkpoint import save as save_ckpt, restore as restore_ckpt
from utils.hparams import setup_hparams
from utils.loops import train, evaluate
from utils.setup_network import setup_network
from utils.final_metrics import avaliar_e_salvar_relatorios
from utils.envinfo import gravar_info_ambiente, salvar_info_modelo


warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Função que ajsutar qual scheduler foi escolhido
def criar_scheduler(optimizer, hps, steps_per_epoch: int):
    tipo = hps.get("scheduler", "plateau")
    if tipo == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=5, verbose=True)
    if tipo == "cosine":
        T_max = int(hps.get("n_epochs", 10))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    if tipo == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    if tipo == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=float(hps["lr"]), steps_per_epoch=steps_per_epoch, epochs=int(hps["n_epochs"])
        )
    return None

# Função responsavel por salvar informações em csv de cada epoch de cada run (Somente treino e validação)
def gravar_csv_epoca(path):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow([
            "epoch", "acc_tr", "loss_tr", "acc_val", "loss_val",
            "lr", "epoch_time_s", "no_improve", "best_epoch_so_far", "best_metric_so_far"
        ])
    return f, writer

# Função que organiza de fato o treinamento
def run(net, logger, hps):

    # Dataloaders
    csv_path = hps.get("dataset_csv", "/home/ayrton/novaBaseline_FPF/dataset/fer2013.csv")
    trainloader, valloader, testloader = get_dataloaders(path=csv_path, bs=int(hps["bs"]))

    net = net.to(device)
    learning_rate = float(hps["lr"])
    scaler = GradScaler()

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params, lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001
    )
    scheduler = criar_scheduler(optimizer, hps, steps_per_epoch=len(trainloader))

    try:
        restore_ckpt(net, logger, hps, optimizer=optimizer, scheduler=scheduler)
    except Exception as e:
        print(f"[restore] Erro opcional ao restaurar optimizer/scheduler: {e}")

    # Informações de tl para registro em logs, etc
    tl_enabled = bool(hps.get('transfer_learning', False))
    tl_strategy = str(hps.get('tl_strategy', 'partial')).lower()
    params_total = sum(p.numel() for p in net.parameters())
    params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    params_frozen = params_total - params_trainable

    # Logs auxiliares
    model_dir = os.path.join("checkpoints", hps["name"])
    os.makedirs(model_dir, exist_ok=True)
    epoch_csv_path = os.path.join(model_dir, "epoch_log.csv")
    epoch_csv_f, epoch_csv_writer = gravar_csv_epoca(epoch_csv_path)

    best_metric = -float("inf")  # val_acc por padrão
    best_epoch_idx = None
    no_improve = 0

    es_on = bool(hps.get("early_stop", True))
    es_monitor = hps.get("es_monitor", "val_acc")
    es_mode = hps.get("es_mode", "max")
    es_patience = int(hps.get("es_patience", 12))
    es_min_delta = float(hps.get("es_min_delta", 5e-4))
    es_min_epochs = int(hps.get("es_min_epochs", 15))
    es_restore_best = bool(hps.get("es_restore_best", True))

    print("Training", hps["name"], "on", device)
    if tl_enabled:
        print(f"[TL] enabled=True strategy={tl_strategy} | params(trainable/frozen/total): "
              f"{params_trainable}/{params_frozen}/{params_total}")

    t_total_start = time.perf_counter()

    for epoch in range(int(hps["start_epoch"]), int(hps["n_epochs"])):
        t_ep = time.perf_counter()

        #Treino
        acc_tr, loss_tr = train(net, trainloader, nn.CrossEntropyLoss(), optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        #Validação
        acc_v, loss_v = evaluate(net, valloader, nn.CrossEntropyLoss())
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        #Scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(acc_v)
        elif scheduler is not None:
            scheduler.step()

        #Checkpoint do melhor
        cur_metric = acc_v if es_monitor == "val_acc" else -loss_v
        improved = cur_metric > (best_metric + es_min_delta)
        if improved:
            best_metric = cur_metric
            best_epoch_idx = epoch + 1  # checkpoints são 1-indexados
            no_improve = 0
            save_ckpt(
                logger=logger, net=net, hps=hps, epoch=best_epoch_idx,
                optimizer=optimizer, scheduler=scheduler, is_best=True, make_last_alias=True
            )
            logger.save_plt(hps)
        else:
            no_improve += 1
            if ((epoch + 1) % int(hps["save_freq"])) == 0:
                save_ckpt(
                    logger=logger, net=net, hps=hps, epoch=epoch + 1,
                    optimizer=optimizer, scheduler=scheduler, is_best=False, make_last_alias=True
                )
                logger.save_plt(hps)

        #CSV por época
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time_s = time.perf_counter() - t_ep
        epoch_csv_writer.writerow([
            epoch + 1, f"{acc_tr:.6f}", f"{loss_tr:.8f}", f"{acc_v:.6f}", f"{loss_v:.8f}",
            f"{lr_now:.8f}", f"{epoch_time_s:.3f}", no_improve,
            (best_epoch_idx if best_epoch_idx is not None else ""), f"{best_metric:.6f}" if best_epoch_idx else ""
        ])

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.4f %%' % acc_tr,
              'Val Accuracy: %2.4f %%' % acc_v,
              sep='\t\t')

        #Early Stopping
        if es_on and (epoch + 1) >= es_min_epochs and no_improve >= es_patience:
            print(f"[ES] Parando antecipado na época {epoch + 1}. Melhor em {best_epoch_idx} (métrica={best_metric:.4f}).")
            break

        #Última época (Para garantir que ela seja salva)
        if (epoch + 1) == int(hps["n_epochs"]):
            print(f"[INFO] Salvando manualmente a última epoch {epoch + 1}")
            save_ckpt(
                logger=logger, net=net, hps=hps, epoch=epoch + 1,
                optimizer=optimizer, scheduler=scheduler, is_best=False, make_last_alias=True
            )
            logger.save_plt(hps)

    # Fechar CSV
    epoch_csv_f.close()

    # Avaliação final
    acc_test, loss_test = evaluate(net, testloader, nn.CrossEntropyLoss())
    print('Test Accuracy: %2.4f %%' % acc_test, 'Test Loss: %2.6f' % loss_test, sep='\t\t')

    # Restaurar melhor epoch antes de relatórios
    eval_epoch = best_epoch_idx if best_epoch_idx is not None else epoch + 1
    if hps["name"] and hps["network"]:
        avaliar_e_salvar_relatorios(name=hps["name"], network=hps["network"], epoch=eval_epoch)

    # Tempo total + resumo
    t_total_end = time.perf_counter()
    total_s = t_total_end - t_total_start
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join("checkpoints", hps["name"], "training_time.txt"), "a") as f:
        f.write(f"Training ended at: {end_time}\n")

    summary = {
        "name": hps["name"],
        "network": hps["network"],
        "scheduler": hps.get("scheduler", "plateau"),
        "best_epoch": best_epoch_idx,
        "best_metric": (best_metric if best_epoch_idx is not None else None),
        "test_acc": float(acc_test),
        "test_loss": float(loss_test),
        "total_time_s": round(total_s, 3),

        "transfer_learning": tl_enabled,
        "tl_strategy": tl_strategy,
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "params_frozen": int(params_frozen),

        "early_stopped": bool(es_on and (best_epoch_idx is not None) and ((epoch + 1) < int(hps["n_epochs"]))),
        "paths": {
            "best": os.path.join("checkpoints", hps["name"], "best.pth"),
            "last": os.path.join("checkpoints", hps["name"], "last.pth"),
            "epoch_csv": os.path.join("checkpoints", hps["name"], "epoch_log.csv"),
        }
    }
    with open(os.path.join("checkpoints", hps["name"], "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

# registra todas as informações do hparams em txt
def gravar_hparams_txt(hps):
    hparams_path = os.path.join("checkpoints", hps["name"], "hparams_log.txt")
    with open(hparams_path, "w") as f:
        f.write(f"Experiment name: {hps['name']}\n\n")
        for key in sorted(hps.keys()):
            f.write(f"{key}: {hps[key]}\n")


if __name__ == "__main__":
    # Important parameters
    hps = setup_hparams(sys.argv[1:])

    # Redireciona prints p/ arquivo (mantém DualLogger)
    log_file_path = os.path.join("checkpoints", hps["name"], "full_print_log.txt")

    class DualLogger:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, "w", buffering=1)

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = sys.stderr = DualLogger(log_file_path)

    # Pastas + hora de início
    model_save_dir = os.path.join("checkpoints", hps["name"])
    os.makedirs(model_save_dir, exist_ok=True)
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(model_save_dir, "training_time.txt"), "w") as f:
        f.write(f"Training started at: {start_time}\n")

    # hparams log
    gravar_hparams_txt(hps)

    # infos de ambiente (env.json/txt) e do modelo
    gravar_info_ambiente(hps)

    # Network + logger
    logger, net = setup_network(hps)

    # infos do modelo (parâmetros/arch)
    salvar_info_modelo(net, hps)

    # Treino
    run(net, logger, hps)
