import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from data.dataset import CustomDataset


# le o CSV FER2013 e arrumar para um df e mapa de rótulos
def load_data(path: str = '/home/ayrton/novaBaseline_FPF/dataset/fer2013.csv'):
    fer2013 = pd.read_csv(path)
    mapeamento_emocoes = {
        0: 'Angry', 1: 'Disgust', 2: 'Fear',
        3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
    }
    return fer2013, mapeamento_emocoes

#Converte a coluna 'pixels' (string) para imagens 48x48 (uint8)
def prepare_data(df: pd.DataFrame):
    pixels = df['pixels'].tolist()
    X = np.array([np.fromstring(p, sep=' ', dtype=np.float32) for p in pixels], dtype=np.float32)
    X = X.reshape(-1, 48, 48)
    X = np.clip(X, 0, 255).astype(np.uint8)

    y = df['emotion'].astype(np.int64).to_numpy()

    usage = df['Usage'].astype(str).to_numpy()
    mask_train = usage == 'Training'
    mask_val   = usage == 'PrivateTest'
    mask_test  = usage == 'PublicTest'

    xtrain, ytrain = X[mask_train], y[mask_train]
    xval,   yval   = X[mask_val],   y[mask_val]
    xtest,  ytest  = X[mask_test],  y[mask_test]

    return (xtrain, ytrain), (xval, yval), (xtest, ytest)


# Aplica ToTensor -> Normalize (-> RandomErasing opcional) em cada crop do TenCrop e empilha tudo no formato (10, C, H, W) para cada imagem
def empilhar_crops_tensor(normalize, erasing=None):
    def aplicar(crops):
        tensors = []
        for c in crops:
            t = TF.to_tensor(c)     # (C,H,W) em [0,1]
            t = normalize(t)        # Normaliza (0, 255) -> ~ [0,1]
            if erasing is not None:
                t = erasing(t)      # aplicado por-crop
            tensors.append(t)
        return torch.stack(tensors)
    return transforms.Lambda(aplicar)

# Função que monta os transforms de treino e de val/test.
def montar_transforms(augment: bool = True):
    normalizar = transforms.Normalize(mean=(0.0,), std=(255.0,))
    erasing = transforms.RandomErasing(p=0.5)

    # TenCrop + ToTensor + Normalize [+ RandomErasing em treino]
    test_transform = transforms.Compose([
        transforms.TenCrop(40),
        empilhar_crops_tensor(normalizar, erasing=None),
    ])

    if not augment:
        train_transform = test_transform
        return train_transform, test_transform

    # Augment do treino
    try:
        rrc = transforms.RandomResizedCrop(48, scale=(0.8, 1.2))
    except Exception:
        rrc = transforms.RandomResizedCrop(48, scale=(0.8, 1.0))

    # RandomErasing é aplicado POR CROP (depois do TenCrop) -> Quero mudar isso
    train_transform = transforms.Compose([
        rrc,
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.TenCrop(40),
        empilhar_crops_tensor(normalizar, erasing=erasing),   # RandomErasing por crop
    ])

    return train_transform, test_transform


#Retorna (trainloader, valloader, testloader) prontos
def get_dataloaders(
    path: str = '/home/ayrton/novaBaseline_FPF/dataset/fer2013.csv',
    bs: int = 64,
    augment: bool = True,
    val_shuffle: bool = False,
    test_shuffle: bool = False,
):
    df, _ = load_data(path=path)
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = prepare_data(df)

    train_transform, test_transform = montar_transforms(augment=augment)

    train = CustomDataset(xtrain, ytrain, train_transform)
    val   = CustomDataset(xval,   yval,   test_transform)
    test  = CustomDataset(xtest,  ytest,  test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True)
    valloader   = DataLoader(val,   batch_size=64, shuffle=val_shuffle)
    testloader  = DataLoader(test,  batch_size=64, shuffle=test_shuffle)

    return trainloader, valloader, testloader
