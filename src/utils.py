import os
import json
import random
from datetime import datetime #для лога

import numpy as np
import torch


def set_seed(seed=42):
'''
  делаем рандом воспроизводимым
'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json_log(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    data.append(record)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
