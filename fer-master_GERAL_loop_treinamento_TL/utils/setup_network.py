from utils.checkpoint import restore
from utils.logger import Logger
from models import vgg, resnet18

# Cria o modelo de acordo com o nome passado em hps['network'], caso eu queira utilizar novos modelos, não posso esquecer de atualizar aqui
def construir_rede(hps):
    name = hps['network']
    # Instancia VGG com dropout e opções de tl
    if name == 'vgg':
        return vgg.Vgg(
            drop=hps['drop'],
            transfer_learning=bool(hps.get('transfer_learning', False)),
            tl_strategy=str(hps.get('tl_strategy', 'partial')).lower()
        )
    # Instancia ResNet18 com as mesmas opções do vgg
    if name == 'resnet18':
        return resnet18.resnet18(
            drop=hps['drop'],
            transfer_learning=bool(hps.get('transfer_learning', False)),
            tl_strategy=str(hps.get('tl_strategy', 'partial')).lower()
        )
    raise ValueError(f"Network '{name}' not supported by setup_network.")

# Monta a rede e o logger, e já calcula os contadores de parâmetros
def setup_network(hps):
    net = construir_rede(hps)

    logger = Logger()

    tl_enabled = bool(hps.get('transfer_learning', False))
    tl_strategy = str(hps.get('tl_strategy', 'partial')).lower()

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = total - trainable

    logger.tl_enabled = tl_enabled
    logger.tl_strategy = tl_strategy
    logger.params_total = int(total)
    logger.params_trainable = int(trainable)
    logger.params_frozen = int(frozen)

    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net
