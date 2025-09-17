import os
import json
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from data.fer2013 import get_dataloaders

#Reconstrói o modelo (VGG ou ResNet18) só com o dropout ajustado
def construir_modelo(network: str, drop: float = 0.1):
    if network == 'vgg':
        from models.vgg import Vgg
        return Vgg(drop=drop)
    if network == 'resnet18':
        from models.resnet18 import resnet18
        return resnet18(drop=drop)
    raise ValueError(f"Network '{network}' não suportada neste avaliador.")

#Se tiver TenCrop, tira média dos logits (B*ncrops -> B).
@torch.no_grad()
def media_sobre_crops(logits: torch.Tensor, ncrops: int) -> torch.Tensor:
    if ncrops <= 1:
        return logits
    B = logits.size(0) // ncrops
    return logits.view(B, ncrops, -1).mean(dim=1)


def obter_dispositivo():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Função responsável por gerar os relatorios finais
def avaliar_e_salvar_relatorios(name: str, network: str, epoch: int):
    model_dir = os.path.join("checkpoints", name)
    os.makedirs(model_dir, exist_ok=True)

    hps_json = os.path.join(model_dir, "hparams.json")
    dataset_csv = '/home/ayrton/novaBaseline_FPF/dataset/fer2013.csv'
    drop = 0.1
    try:
        if os.path.exists(hps_json):
            with open(hps_json, "r") as f:
                hps = json.load(f)
            dataset_csv = hps.get('dataset_csv', dataset_csv)
            drop = float(hps.get('drop', drop))
    except Exception:
        pass

    device = obter_dispositivo()
    net = construir_modelo(network, drop=drop).to(device)
    ckpt_path = os.path.join(model_dir, f"epoch_{epoch}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(ckpt["params"])
    net.eval()

    _, _, testloader = get_dataloaders(path=dataset_csv, bs=64, augment=False, val_shuffle=False, test_shuffle=False)

    y_true, y_pred, y_conf = [], [], []
    for i, (x, labels) in enumerate(testloader):
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if x.dim() == 5:
            B, ncrops, C, H, W = x.shape
            x_ = x.view(-1, C, H, W)
            logits_all = net(x_)
            logits_avg = media_sobre_crops(logits_all, ncrops)
        else:
            logits_avg = net(x)

        probs = F.softmax(logits_avg, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_conf.append(probs.max(dim=1).values.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_conf = np.concatenate(y_conf, axis=0)

    acc = 100.0 * accuracy_score(y_true, y_pred)
    prec = 100.0 * precision_score(y_true, y_pred, average='micro', zero_division=0)
    rec = 100.0 * recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = 100.0 * f1_score(y_true, y_pred, average='micro', zero_division=0)

    target_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    rep_txt = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
        f.write(rep_txt)

    rep_dict = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    import csv as _csv
    with open(os.path.join(model_dir, "classification_report.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        # cabeçalho
        w.writerow(["label","precision","recall","f1-score","support"])
        for k, v in rep_dict.items():
            if isinstance(v, dict) and all(s in v for s in ("precision","recall","f1-score","support")):
                w.writerow([k, v["precision"], v["recall"], v["f1-score"], v["support"]])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(7)
    plt.xticks(tick_marks, target_names, rotation=45, ha='right')
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    with np.errstate(all='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    plt.title('Confusion Matrix (Row-normalized)')
    plt.colorbar()
    tick_marks = np.arange(7)
    plt.xticks(tick_marks, target_names, rotation=45, ha='right')
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(os.path.join(model_dir, "confusion_matrix_norm.png"), dpi=200)
    plt.close(fig)

    mis_idx = np.where(y_true != y_pred)[0]
    if mis_idx.size > 0:
        import csv as _csv2
        with open(os.path.join(model_dir, "misclassifications.csv"), "w", newline="") as f:
            w = _csv2.writer(f)
            w.writerow(["index","true","pred","confidence"])
            for idx in mis_idx:
                w.writerow([int(idx), int(y_true[idx]), int(y_pred[idx]), float(y_conf[idx])])

    fm_path = os.path.join(model_dir, "final_metrics.csv")
    new_file = not os.path.exists(fm_path)
    import csv as _csv3
    with open(fm_path, "a", newline="") as f:
        w = _csv3.writer(f)
        if new_file:
            w.writerow(["timestamp","epoch","accuracy_micro","precision_micro","recall_micro","f1_micro"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch,
            f"{acc:.6f}",
            f"{prec:.6f}",
            f"{rec:.6f}",
            f"{f1:.6f}",
        ])
