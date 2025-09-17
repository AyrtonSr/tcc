import torch
import torch.nn as nn
import torchvision.models as models

#Função responsável por trocar a conv1 de 3 canais (RGB) para 1 canal (grayscale) para funcioanar no fer2013
def adaptar_conv1_para_cinza(conv: nn.Conv2d) -> nn.Conv2d:
    nova_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    )
    with torch.no_grad():
        if conv.weight.shape[1] == 3:
            w = conv.weight.mean(dim=1, keepdim=True)  # (out,1,k,k)
        else:
            w = conv.weight
        nova_conv.weight.copy_(w)
        if conv.bias is not None and nova_conv.bias is not None:
            nova_conv.bias.copy_(conv.bias)
    return nova_conv


class resnet18(nn.Module):
    def __init__(self, drop=0.2, transfer_learning: bool = False, tl_strategy: str = 'partial'):
        super(resnet18, self).__init__()

        #Carrega a ResNet18 (pré-treinada no ImageNet se TL=True)
        if transfer_learning:
            try:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                self.model = models.resnet18(pretrained=True)
        else:
            try:
                self.model = models.resnet18(weights=None)
            except Exception:
                self.model = models.resnet18(pretrained=False)

        # Adapta a primeira conv para trabalhar com imagens em tons de cinza (1 canal)
        self.model.conv1 = adaptar_conv1_para_cinza(self.model.conv1)

        # Troca a camada final: dropout para regularizar + linear para 7 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(num_ftrs, 7)
        )

        # Regras de Transfer Learning:
        # - partial: congela tudo e treina só a layer4 + fc
        # - full: treina tudo
        if transfer_learning:
            if tl_strategy == 'partial':
                for p in self.model.parameters():
                    p.requires_grad = False
                for p in self.model.layer4.parameters():
                    p.requires_grad = True
                for p in self.model.fc.parameters():
                    p.requires_grad = True
            elif tl_strategy == 'full':
                # tudo treinável
                for p in self.model.parameters():
                    p.requires_grad = True

    def forward(self, x):
        return self.model(x)
