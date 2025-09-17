import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
from train import run as train_one_experiment
from utils.hparams import setup_hparams
from utils.setup_network import setup_network

DIR_RESULTADOS = os.path.join(os.getcwd(), "results")
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Essa função serve para converter listas tipo ["lr=0.01", "drop=0.5"] em dicionários {"lr": "0.01", "drop": "0.5"}
def converter_base_kv(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise SystemExit(f"--base espera itens no formato key=value; recebido: '{kv}'")
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out

# Caso eu queria fazer um treinamento pelo grid do terminal, essa função vai transformar em dicionario para funcionar
def converter_grid_str(grid_str: str) -> Dict[str, List[str]]:
    grid: Dict[str, List[str]] = {}
    if not grid_str:
        return grid
    parts = [p for p in grid_str.split(";") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise SystemExit(f"Entrada de grid inválida: '{part}'")
        k, rhs = part.split("=", 1)
        k = k.strip()
        rhs = rhs.strip()
        if not (rhs.startswith("[") and rhs.endswith("]")):
            raise SystemExit(f"Valores de grid devem estar em colchetes []. Erro em: '{part}'")
        inner = rhs[1:-1].strip()
        vals = [x.strip().strip("'").strip('"') for x in inner.split(",") if x.strip()]
        if not vals:
            raise SystemExit(f"Nenhum valor para chave '{k}' no grid.")
        grid[k] = vals
    return grid


def expandir_grid(grid: Dict[str, List[str]]) -> List[Dict[str, str]]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    out: List[Dict[str, str]] = []
    for tpl in combos:
        d = {k: v for k, v in zip(keys, tpl)}
        out.append(d)
    return out

# Função responsável por ler o JSON e transformar em runs
def carregar_sweep_json(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        out = []
        for item in obj:
            if not isinstance(item, dict):
                raise SystemExit("Lista de runs no JSON deve conter dicts de overrides.")
            # converte valores para string (hparams cuida de tipos)
            out.append({str(k): str(v) for k, v in item.items()})
        return out
    if isinstance(obj, dict) and "grid" in obj:
        grid = obj["grid"]
        if not isinstance(grid, dict):
            raise SystemExit("'grid' no JSON deve ser um objeto {k:[...]} ")
        # normaliza para lista de strings
        norm = {str(k): [str(x) for x in v] for k, v in grid.items()}
        return expandir_grid(norm)
    raise SystemExit("JSON inválido: use lista de dicts ou {'grid': {...}}")

# Criador de nomes únicos para cada run
def criar_nome(hp: Dict[str, Any], idx: int) -> str:
    """
    <network>_bs<bs>_lr<lr>_drop<drop>_<YYYYmmdd-HHMMSS>_<idx3>
    """
    parts = []
    for key in ("network", "bs", "lr", "drop"):
        val = hp.get(key)
        if val not in (None, "", "None"):
            if key == "network":
                parts.append(str(val))
            else:
                parts.append(f"{key}{val}")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts.append(ts)
    parts.append(f"{idx:03d}")
    return "_".join(parts)

# Transforma dict em lista de strings "k=v" (O hparams só vai funcioar se isso acontecer. IMPORTANTE)
def converter_para_lista_kv(d: Dict[str, Any]) -> List[str]:
    return [f"{k}={d[k]}" for k in sorted(d.keys())]

#Registra o resultado de cada run realizada, ou seja, é aqui que vai estar e resultado de todos os treinamentos realizados por aquele sweep
def gravar_linha_resultados(path: str, row: Dict[str, Any]) -> None:
    import csv
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "network", "scheduler",
            "best_epoch", "best_metric", "test_acc", "test_loss",
            "total_time_s", "status", "message"
        ])
        if new_file:
            w.writeheader()
        w.writerow(row)

# Função principal que vai organizar todos os treinamos, usando a funçaõ train.py
def main():
    ap = argparse.ArgumentParser(
        description="Runner – orquestra vários treinamentos em sequência usando o train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--base", nargs="*", default=[],
                    help="Overrides base no formato key=value (aplicadas em todos os runs)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--grid", type=str, default="",
                       help="Especificação de grid: ex. \"network=[vgg,resnet18]; lr=[0.01,0.005]\"")
    group.add_argument("--sweep_json", type=str, default="",
                       help="Caminho para JSON com lista de dicts OU {'grid': {...}}")
    ap.add_argument("--results_csv", type=str, default=os.path.join(DIR_RESULTADOS, "summary.csv"),
                    help="Arquivo de consolidação de resultados")
    args = ap.parse_args()

    base = converter_base_kv(args.base)

    # Constrói a lista de runs
    if args.grid:
        overrides_list = expandir_grid(converter_grid_str(args.grid))
    elif args.sweep_json:
        overrides_list = carregar_sweep_json(args.sweep_json)
    else:
        overrides_list = [{}]  # um único run com o base

    print(f"[runner] Total de runs: {len(overrides_list)}")

    # Por combinação de hparams
    for i, overrides in enumerate(overrides_list, start=1):
        # Junta os parâmetros base com os específicos dessa execução
        hp_dict = dict(base)
        hp_dict.update(overrides)
        if not hp_dict.get("name"): # Caso não tenha nome, cria um de acordo com o hps
            hp_dict["name"] = criar_nome(hp_dict, i)

        # Monta lista "key=value" para o setup_hparams
        kv_list = converter_para_lista_kv(hp_dict)

        print(f"[runner] === Run {i}/{len(overrides_list)}: {hp_dict['name']} ===")
        print("[runner] hparams:", ", ".join(kv_list))

        # Constrói hps, logger e rede, e roda o treino
        try:
            hps = setup_hparams(kv_list)   # cria pastas e salva configs
            logger, net = setup_network(hps)  # monta rede e logger
            train_one_experiment(net, logger, hps) # Aqui que começa o treinamento
            status, message = "ok", ""
        except SystemExit as e:
            status, message = "error", f"SystemExit: {str(e)}"
            print(f"[runner] ERRO (SystemExit): {e}")
        except Exception as e:
            status, message = "error", repr(e)
            print(f"[runner] ERRO: {e}")

        # Registra os resultados lendo run_summary.json (se existir)
        summary_path = os.path.join("checkpoints", hp_dict["name"], "run_summary.json")
        row = {
            "name": hp_dict["name"],
            "network": hp_dict.get("network", ""),
            "scheduler": hp_dict.get("scheduler", ""),
            "best_epoch": "",
            "best_metric": "",
            "test_acc": "",
            "test_loss": "",
            "total_time_s": "",
            "status": status,
            "message": message,
        }
        try:
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                row.update({
                    "best_epoch": s.get("best_epoch", ""),
                    "best_metric": s.get("best_metric", ""),
                    "test_acc": s.get("test_acc", ""),
                    "test_loss": s.get("test_loss", ""),
                    "total_time_s": s.get("total_time_s", ""),
                })
        finally:
            gravar_linha_resultados(args.results_csv, row)

    print(f"[runner] Finalizado. Consolidação em: {args.results_csv}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()

