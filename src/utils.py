import os
import json
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed=42):
    """
    Фиксируем все источники случайности, чтобы результаты были воспроизводимы.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json_log(path, record):
    """
    Дописывает одну запись в JSON-лог (список объектов).
    Если файл не существует — создаёт его.
    Если директория не существует — создаёт и её.
    """
    # Фикс: dirname может быть пустой строкой если path — просто имя файла
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    record = dict(record)  # копируем чтобы не мутировать оригинал
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    data.append(record)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
