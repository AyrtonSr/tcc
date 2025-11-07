import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib
matplotlib.use("Agg")  # garante ambiente headless
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# =====================================================
# ----------------- CONFIGURAÇÕES GERAIS --------------
# =====================================================
RUNS_BASE = Path("Treinamentos")
RUN_NAME = os.environ.get("RUN_NAME", "treinamento1")  # pode alterar via env var
RUN_DIR = RUNS_BASE / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path("./logs") / RUN_NAME
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 35
LEARNING_RATE = 5e-4
MOMENTUM = 0.9
STEP_SIZE = 6
GAMMA = 0.1
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================================
# --------------------- TRANSFORMS --------------------
# =====================================================
input_size = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# =====================================================
# -------------------- DATA LOADERS -------------------
# =====================================================
image_datasets = {x: datasets.ImageFolder(DATA_DIR / x, data_transforms[x]) for x in ['train', 'val']}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# =====================================================
# ------------------- DISPOSITIVO ---------------------
# =====================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =====================================================
# --------------------- UTILITÁRIOS -------------------
# =====================================================

def imshow(inp, title=None, save_path=None):
    """Desserializa e salva uma imagem em RGB."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    fig = plt.figure()
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def save_grid_from_loader(loader, save_path, title=None, max_images=16):
    """Salva uma grade de imagens de um DataLoader."""
    inputs, classes_idx = next(iter(loader))
    grid = torchvision.utils.make_grid(inputs[:max_images])
    fig = plt.figure(figsize=(10, 10))
    npimg = grid.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    npimg = std * npimg + mean
    npimg = np.clip(npimg, 0, 1)
    plt.imshow(npimg)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_curves(history, save_path_prefix, writer=None):
    """Plota e salva curvas de loss e acurácia (treino/val)."""
    # Loss
    fig1 = plt.figure()
    plt.plot(history['epoch'], history['train_loss'], label='train loss')
    plt.plot(history['epoch'], history['val_loss'], label='val loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    fig1.savefig(f"{save_path_prefix}_loss.png", bbox_inches='tight', dpi=150)
    if writer:
        writer.add_figure('curves/loss', fig1, global_step=max(history['epoch']))
    plt.close(fig1)

    # Acurácia
    fig2 = plt.figure()
    plt.plot(history['epoch'], history['train_acc'], label='train acc')
    plt.plot(history['epoch'], history['val_acc'], label='val acc')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    fig2.savefig(f"{save_path_prefix}_acc.png", bbox_inches='tight', dpi=150)
    if writer:
        writer.add_figure('curves/accuracy', fig2, global_step=max(history['epoch']))
    plt.close(fig2)


def evaluate_and_log(model, dataloader, class_names, save_prefix, writer=None):
    """Gera matriz de confusão e relatório de classificação; salva em disco e TensorBoard."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig_cm = plt.figure(figsize=(1 + 0.8*len(class_names), 1 + 0.8*len(class_names)))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Matriz de Confusão')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_ylabel('Verdadeiro')
    ax.set_xlabel('Predito')

    # números dentro das células
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig_cm.tight_layout()
    cm_path = f"{save_prefix}_confusion_matrix.png"
    fig_cm.savefig(cm_path, bbox_inches='tight', dpi=150)
    if writer:
        writer.add_figure('eval/confusion_matrix', fig_cm)
    plt.close(fig_cm)

    # Relatório de classificação
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_path = f"{save_prefix}_classification_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    if writer:
        writer.add_text('eval/classification_report', f"```{report}```", global_step=0)

    return cm_path, report_path


def save_history_csv(history, save_path):
    import csv
    fields = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "epoch_time_sec", "lr"]
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i],
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_acc'][i],
                history['val_acc'][i],
                history['epoch_time_sec'][i],
                history['lr'][i],
            ])

# =====================================================
# --------------------- MODELO ------------------------
# =====================================================
model_ft = models.densenet161(pretrained=True)
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)

# TensorBoard (um diretório por execução)
writer = SummaryWriter(str(LOGS_DIR))

# Salva uma amostra de treino
save_grid_from_loader(dataloaders['train'], RUN_DIR / 'train_batch.png', title='Amostra treino')

# =====================================================
# -------------------- TREINAMENTO --------------------
# =====================================================

def train_model(model, criterion, optimizer, scheduler, writer=None, num_epochs=25):
    since = time.time()

    best_model_wts = None
    best_acc = 0.0

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time_sec': [],
        'lr': [],
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # LR vigente nesta época (antes do scheduler.step)
        current_lr = optimizer.param_groups[0]['lr']

        # buffers para armazenar as métricas calculadas em cada fase
        train_epoch_loss = None
        train_epoch_acc  = None
        val_epoch_loss   = None
        val_epoch_acc    = None

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            # atualiza scheduler ao fim da fase de treino
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            # logs TensorBoard (se houver writer)
            if writer:
                writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase}/acc',  epoch_acc,  epoch)
                if phase == 'train':
                    writer.add_scalar('train/lr', current_lr, epoch)

            # guarda métricas calculadas
            if phase == 'train':
                train_epoch_loss = epoch_loss
                train_epoch_acc  = epoch_acc
            else:
                val_epoch_loss = epoch_loss
                val_epoch_acc  = epoch_acc
                # checkpoint do melhor modelo pela acurácia de validação
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # fim das duas fases -> registra no history uma vez por época
        epoch_time = time.time() - epoch_start
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['train_acc'].append(train_epoch_acc)
        history['val_acc'].append(val_epoch_acc)
        history['epoch_time_sec'].append(epoch_time)
        history['lr'].append(current_lr)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_epoch_loss:.4f} val_loss={val_epoch_loss:.4f} "
            f"train_acc={train_epoch_acc:.4f} val_acc={val_epoch_acc:.4f} "
            f"time={epoch_time:.1f}s lr={current_lr:.6f}"
        )

    total_time_sec = time.time() - since

    # salva tempos
    times_path = RUN_DIR / 'training_time.json'
    with open(times_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_time_seconds': total_time_sec,
            'total_time_minutes': total_time_sec / 60.0
        }, f, indent=2)

    print(f"Tempo total de treinamento: {total_time_sec/60.0:.2f} minutos (acc_val_melhor={best_acc:.4f})")

    # restaura melhor modelo
    if best_model_wts is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_wts.items()})

    return model, history


# ===================== EXECUÇÃO ======================
model_ft, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, writer, num_epochs=NUM_EPOCHS)

# salva histórico em CSV
csv_path = RUN_DIR / 'metrics.csv'
save_history_csv(history, csv_path)

# plota curvas
plot_curves(history, save_path_prefix=str(RUN_DIR / 'training'), writer=writer)

# visualiza previsões (breve grade salva)
# Gera uma figura com N imagens de validação e suas predições

def visualize_model(model, num_images=6, save_path=None):
    was_training = model.training
    model.eval()
    images_so_far = 0

    cols = 3
    rows = int(np.ceil(num_images / cols))
    fig = plt.figure(figsize=(cols * 4, rows * 4))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                if images_so_far == num_images:
                    break
                ax = plt.subplot(rows, cols, images_so_far + 1)
                ax.axis('off')
                ax.set_title(f'pred: {class_names[preds[j]]}')

                # desserializa
                img = inputs[j].detach().cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                ax.imshow(img)

                images_so_far += 1
            if images_so_far == num_images:
                break

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    if writer:
        writer.add_figure('samples/val_predictions', fig)
    plt.close(fig)
    model.train(mode=was_training)

visualize_model(model_ft, num_images=6, save_path=RUN_DIR / 'val_predictions.png')

# Avaliação final: matriz de confusão e relatório de classificação
cm_path, report_path = evaluate_and_log(model_ft, dataloaders['val'], class_names, save_prefix=str(RUN_DIR / 'eval'), writer=writer)

# fecha writer
writer.flush()
writer.close()

# ========= RESUMO DE SAÍDAS GERADAS =========
summary = {
    "curvas_loss": str(RUN_DIR / 'training_loss.png'),
    "curvas_acc": str(RUN_DIR / 'training_acc.png'),
    "matriz_confusao": cm_path,
    "relatorio_classificacao": report_path,
    "amostra_treino": str(RUN_DIR / 'train_batch.png'),
    "predicoes_validacao": str(RUN_DIR / 'val_predictions.png'),
    "historico_csv": str(csv_path),
    "tensorboard_logdir": str(LOGS_DIR),
}
with open(RUN_DIR / 'outputs_index.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
