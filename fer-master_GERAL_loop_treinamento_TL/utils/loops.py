import torch
import torch.nn as nn
from torch.cuda.amp import autocast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Se tiver TenCrop (vários recortes por imagem), tira a média dos logits para voltar de (B*ncrops, classes) para (B, classes)
@torch.no_grad()
def media_sobre_crops(logits: torch.Tensor, ncrops: int) -> torch.Tensor:
    if ncrops <= 1:
        return logits
    B = logits.size(0) // ncrops
    return logits.view(B, ncrops, -1).mean(dim=1)


def train(
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    Ncrop: bool = True,
):
    net.train()
    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for x, labels in dataloader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Se TenCrop: reshape para (B*ncrops, C, H, W) e duplica labels
        if Ncrop and x.dim() == 5:
            B, ncrops, C, H, W = x.shape
            x = x.view(-1, C, H, W)
            labels_eff = labels.view(-1, 1).repeat(1, ncrops).view(-1)
        else:
            # Sem TenCrop: mantém rótulos originais
            labels_eff = labels

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = net(x)
            loss = criterion(logits, labels_eff)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Conta acertos/loss no batch
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels_eff).sum().item()
            bs_eff = labels_eff.size(0)

            total_correct += correct
            total_loss += loss.item() * bs_eff
            total_samples += bs_eff

    avg_loss = total_loss / max(1, total_samples)
    acc = 100.0 * total_correct / max(1, total_samples)
    return acc, avg_loss

# Loop de validação/teste (sem backprop)
@torch.no_grad()
def evaluate(
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    Ncrop: bool = True,
):
    net.eval()
    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for x, labels in dataloader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if Ncrop and x.dim() == 5:
            B, ncrops, C, H, W = x.shape
            x_ = x.view(-1, C, H, W)

            with autocast():
                logits_all = net(x_)
                logits_avg = media_sobre_crops(logits_all, ncrops)
                loss = criterion(logits_avg, labels)

            preds = torch.argmax(logits_avg, dim=1)
            bs_eff = labels.size(0)
            correct = (preds == labels).sum().item()

            total_correct += correct
            total_loss += loss.item() * bs_eff
            total_samples += bs_eff

        else:
            with autocast():
                logits = net(x)
                loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            bs_eff = labels.size(0)
            correct = (preds == labels).sum().item()

            total_correct += correct
            total_loss += loss.item() * bs_eff
            total_samples += bs_eff

    avg_loss = total_loss / max(1, total_samples)
    acc = 100.0 * total_correct / max(1, total_samples)
    return acc, avg_loss
