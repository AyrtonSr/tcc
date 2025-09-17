import os
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Logger:
    def __init__(self):
        # Listas para guardar métricas de treino e validação (uma entrada por época)
        self.loss_train: List[float] = []
        self.loss_val:   List[float] = []
        self.acc_train:  List[float] = []
        self.acc_val:    List[float] = []

        # Metadados sobre Transfer Learning
        self.tl_enabled: bool = False
        self.tl_strategy: str = "none"   # 'none' | 'partial' | 'full'
        self.params_total: int = 0
        self.params_trainable: int = 0
        self.params_frozen: int = 0

    # Exporta todas as métricas e metadados em um dicionário (vai pro checkpoint)
    def get_logs(self) -> Dict[str, Any]:
        return {
            'loss_train': self.loss_train,
            'loss_val':   self.loss_val,
            'acc_train':  self.acc_train,
            'acc_val':    self.acc_val,

            # meta de TL
            'tl_enabled': self.tl_enabled,
            'tl_strategy': self.tl_strategy,
            'params_total': self.params_total,
            'params_trainable': self.params_trainable,
            'params_frozen': self.params_frozen,
        }

    def restore_logs(self, logs: Dict[str, Any]) -> None:
        # Restaura listas de métricas a partir de um dicionário (checkpoint).
        self.loss_train = list(logs.get('loss_train', []))
        self.loss_val   = list(logs.get('loss_val',   []))
        self.acc_train  = list(logs.get('acc_train',  []))
        self.acc_val    = list(logs.get('acc_val',    []))

        self.tl_enabled = bool(logs.get('tl_enabled', False))
        self.tl_strategy = str(logs.get('tl_strategy', 'none'))
        self.params_total = int(logs.get('params_total', 0))
        self.params_trainable = int(logs.get('params_trainable', 0))
        self.params_frozen = int(logs.get('params_frozen', 0))

    # Cria gráficos de acurácia e loss por época e salva em arquivos .jpg
    def save_plt(self, hps: Dict[str, Any]) -> None:
        out_dir = hps.get('model_save_dir') or os.path.join(os.getcwd(), 'checkpoints', hps.get('name', ''))
        os.makedirs(out_dir, exist_ok=True)

        # acurácia
        fig_acc = plt.figure()
        try:
            plt.plot(self.acc_train, label='Train')
            plt.plot(self.acc_val,   label='Val')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy per epoch')
            plt.legend()
            fig_acc.savefig(os.path.join(out_dir, 'acc.jpg'), dpi=150, bbox_inches='tight')
        finally:
            plt.close(fig_acc)

        # loss
        fig_loss = plt.figure()
        try:
            plt.plot(self.loss_train, label='Train')
            plt.plot(self.loss_val,   label='Val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss per epoch')
            plt.legend()
            fig_loss.savefig(os.path.join(out_dir, 'loss.jpg'), dpi=150, bbox_inches='tight')
        finally:
            plt.close(fig_loss)
