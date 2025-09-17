import torch
import torch.nn as nn
import torchvision.models as models

#Função responsável por trocar a conv1 de 3 canais (RGB) para 1 canal (grayscale) para funcioanar no fer2013. Mesma do resnet18
def adaptar_primeira_conv_para_cinza(conv: nn.Conv2d) -> nn.Conv2d:
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
            w = conv.weight.mean(dim=1, keepdim=True)
        else:
            w = conv.weight
        nova_conv.weight.copy_(w)
        if conv.bias is not None and nova_conv.bias is not None:
            nova_conv.bias.copy_(conv.bias)
    return nova_conv


class Vgg(nn.Module):
    def __init__(self, drop: float = 0.2, transfer_learning: bool = False, tl_strategy: str = 'partial'):
        super().__init__()

        # Carrega VGG16 com/sem pesos do ImageNet
        if transfer_learning:
            try:
                self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            except Exception:
                self.model = models.vgg16_bn(pretrained=True)
        else:
            try:
                self.model = models.vgg16_bn(weights=None)
            except Exception:
                self.model = models.vgg16_bn(pretrained=False)

        # Adapta a primeira camada para imagens em grayscale (1 canal)
        self.model.features[0] = adaptar_primeira_conv_para_cinza(self.model.features[0])

        for m in self.model.classifier:
            if isinstance(m, nn.Dropout):
                m.p = drop

        # Troca a última camada para 7 classes
        in_f = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_f, 7)

        # Freezing: parcial = congela features (backbone), treina classifier
        if transfer_learning:
            if tl_strategy == 'partial':
                for p in self.model.features.parameters():
                    p.requires_grad = False
                for p in self.model.classifier.parameters():
                    p.requires_grad = True
            elif tl_strategy == 'full':
                for p in self.model.parameters():
                    p.requires_grad = True

    def forward(self, x):
        return self.model(x)
