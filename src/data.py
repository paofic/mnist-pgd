from pathlib import Path

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms

'''
    У MNIST один канал, поэтому mean и std записаны как кортежи из одного числа

    x_{norm} = frac{x - mean}{std}
'''

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTStageData:
    def __init__(
        self,
        train_parts,
        cumulative_parts,
        test_dataset,
        mean,
        std,
        eps_raw,
        eps_model,
    ):
        self.train_parts = train_parts
        self.cumulative_parts = cumulative_parts
        self.test_dataset = test_dataset
        self.mean = mean
        self.std = std
        self.eps_raw = eps_raw
        self.eps_model = eps_model


def build_mnist_transform():
    """
    Для этого варианта проекта мы ВСЕГДА используем нормализацию.

    Порядок такой:
    1) переводим картинку в тензор
    2) нормализуем по mean и std для MNIST
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])


def _stratified_three_way_split(labels, seed=42):
    """
    Делит индексы train-выборки на 3 части так,
    чтобы в каждой части были примерно те же пропорции классов.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(labels)

    parts = [[], [], []]
    unique_classes = np.unique(labels) # находим все уникальные классы

    for class_id in unique_classes: # цикл по каждому классу
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices) # перемешиваем чтобы разбиение на 3 части
        # было случайным, а не зависело от исходного порядка в датасете.

        class_splits = np.array_split(class_indices, 3) #делим на 3 части

        for part_id in range(3):
            parts[part_id].extend(class_splits[part_id].tolist())

    for part_id in range(3):
        parts[part_id] = sorted(parts[part_id])

    return parts


def describe_split(labels, split_indices):
    """
    Удобная функция для проверки:
    сколько объектов каждого класса попало в каждую часть.
    """
    labels = np.array(labels)
    result = []

    unique_classes = sorted(np.unique(labels).tolist())

    for part_id, indices in enumerate(split_indices):
        part_labels = labels[indices]

        class_counts = {}
        for cls in unique_classes: # для класса считаем сколько в текущую часть
            class_counts[int(cls)] = int((part_labels == cls).sum())

        result.append(
            {
                "part_id": part_id,
                "num_samples": len(indices),
                "class_counts": class_counts,
            }
        )

    return result


def load_mnist_stage_data(
    root="./data",
    seed=42,
    download=True,
    eps_raw=5.0 / 255.0,
):
    """
    Загружает MNIST и подготавливает данные для поэтапного обучения.

    Возвращает объект с:
    - train_parts[0] -> первая треть train
    - train_parts[1] -> вторая треть train
    - train_parts[2] -> третья треть train

    Также возвращает:
    - cumulative_parts[0] -> только первая треть
    - cumulative_parts[1] -> первая + вторая
    - cumulative_parts[2] -> первая + вторая + третья

    И ещё:
    - test_dataset -> тестовая выборка
    - eps_raw -> исходный epsilon в обычной шкале пикселей
    - eps_model -> epsilon в масштабе модели после Normalize
    """
    root = str(Path(root))
    Path(root).mkdir(parents=True, exist_ok=True)

    transform = build_mnist_transform() # создание конвейера

    '''
        Dataset — это объект, который умеет:
        хранить набор данных;
        по индексу возвращать элемент.
        Для PyTorch-датасета обычно:
        image, label = dataset[i]
        То есть dataset — это “коллекция примеров”, с которой можно работать
        как с источником данных.
    '''

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=download,
    )

    labels = train_dataset.targets.cpu().numpy()

    train_parts_indices = _stratified_three_way_split(labels, seed=seed)

    cumulative_parts_indices = [
        sorted(train_parts_indices[0]),
        sorted(train_parts_indices[0] + train_parts_indices[1]),
        sorted(train_parts_indices[0] + train_parts_indices[1] + train_parts_indices[2]),
    ]

    train_parts = []
    for indices in train_parts_indices:
        train_parts.append(Subset(train_dataset, indices))

    cumulative_parts = []
    for indices in cumulative_parts_indices:
        cumulative_parts.append(Subset(train_dataset, indices))


    eps_model = eps_raw / MNIST_STD

    return MNISTStageData(
        train_parts=train_parts,
        cumulative_parts=cumulative_parts,
        test_dataset=test_dataset,
        mean=MNIST_MEAN,
        std=MNIST_STD,
        eps_raw=eps_raw,
        eps_model=eps_model,
    )


def denormalize(x, mean=MNIST_MEAN, std=MNIST_STD):
    """
    Переводит нормализованное изображение обратно в обычное пространство пикселей.
    """
    return x * std + mean


def renormalize(x, mean=MNIST_MEAN, std=MNIST_STD):
    """
    Снова нормализует изображение.
    """
    return (x - mean) / std
