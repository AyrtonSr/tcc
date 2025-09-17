import os
import json
from typing import Iterable, List, Dict, Any

# Padrão (Provavelmente vai ser reescrito com as caracteristicas de cada treinamento)
hps: Dict[str, Any] = {
    'network': '',
    'name': '',
    'n_epochs': 300,
    'model_save_dir': None,
    'restore_epoch': None,
    'start_epoch': 0,
    'lr': 0.01,
    'save_freq': 20,
    'drop': 0.1,
    'bs': 64,

    # Scheduler / Early stop
    'scheduler': 'plateau',             # ['plateau', 'cosine', 'step', 'onecycle']
    'early_stop': True,
    'es_monitor': 'val_acc',            # ['val_acc', 'val_loss']
    'es_mode': 'max',                   # ['max', 'min']
    'es_patience': 20,
    'es_min_delta': 5e-4,
    'es_min_epochs': 15,
    'es_restore_best': True,

    # Dataset
    'dataset_csv': '/home/ayrton/novaBaseline_FPF/dataset/fer2013.csv',
    'val_shuffle': False,
    'test_shuffle': False,

    # Transfer Learning
    'transfer_learning': False,
    'tl_strategy': 'partial',           # ['partial','full']
    'tl_source': 'imagenet',
}

#Garante que os modelos serão requisitados de forma correta
def listar_redes_possiveis() -> List[str]:
    try:
        files = os.listdir('models')
    except FileNotFoundError:
        return []
    nets = []
    for filename in files:
        if not filename.endswith('.py'):
            continue
        name = filename[:-3]
        if name.startswith('_'):
            continue
        nets.append(name)
    return sorted(set(nets))

redes_possiveis = set(listar_redes_possiveis())

# Converte vários formatos para bool (True/False, 1/0, yes/no, y/n) vai que eu esqueço o formato,sla, nunca é demais mesmo kkk
def converter_para_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n', ''}:
        return False
    raise ValueError(f"Cannot parse boolean from '{v}'")

#Função responsável por de fato criar o hparamns final, tanto os que não foram modificados como os que forma
# Também é responsável por criar a pasta chackpoint e salvar o hparams.json nela
def setup_hparams(args: Iterable[str] = None) -> Dict[str, Any]:
    global hps, redes_possiveis

    # Começa obviamente com os valores padrões
    hp = dict(hps)

    # Lw itens do tipo "key=value" vindos da CLI e valida as chaves
    if args is not None:
        try:
            for arg in args:
                if '=' not in arg:
                    raise ValueError(f"Expected key=value, got '{arg}'")
                key, value = arg.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key not in hp:
                    raise ValueError(f"Unknown hparam '{key}'. Allowed: {sorted(hp.keys())}")
                hp[key] = value
        except Exception:
            raise ValueError("Invalid input parameters")

    int_keys = {'n_epochs', 'start_epoch', 'save_freq', 'bs', 'restore_epoch',
                'es_patience', 'es_min_epochs'}
    float_keys = {'lr', 'drop', 'es_min_delta'}
    bool_keys = {'early_stop', 'es_restore_best', 'val_shuffle', 'test_shuffle',
                 'transfer_learning'}

    for k in int_keys:
        if hp.get(k) is None or hp[k] == 'None' or hp[k] == '':
            hp[k] = None if k == 'restore_epoch' else int(hps[k])
        else:
            hp[k] = int(hp[k])

    for k in float_keys:
        hp[k] = float(hp[k])

    for k in bool_keys:
        hp[k] = converter_para_bool(hp[k])

    # Valida se a rede informada existe na pasta models/
    if not redes_possiveis:
        redes_possiveis = set(listar_redes_possiveis())
    if hp['network'] not in redes_possiveis:
        raise ValueError(f"Invalid network '{hp['network']}'. Options: {sorted(redes_possiveis)}")

    # Ve se tem tl
    if hp['transfer_learning']:
        if str(hp.get('tl_strategy', '')).lower() not in {'partial', 'full'}:
            raise ValueError("tl_strategy must be 'partial' or 'full' when transfer_learning=True")

    # Caso queria restaurar vai começar por ele
    if hp['restore_epoch'] is not None:
        hp['start_epoch'] = hp['restore_epoch']

    # Não lembro de ter feito isso, mas caso seja um treino pequeno ele vai salvar mais rapido as epochs, como será q estava a cabeça do palhaço?
    if hp['n_epochs'] < 20:
        hp['save_freq'] = min(5, hp['n_epochs'])

    # Cria a pasta do experimento e salva um hparams.json de conferência
    hp['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hp['name'])
    os.makedirs(hp['model_save_dir'], exist_ok=True)

    try:
        with open(os.path.join(hp['model_save_dir'], 'hparams.json'), 'w', encoding='utf-8') as f:
            json.dump(hp, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return hp
