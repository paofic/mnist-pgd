import torch
import torch.nn as nn

try:
    from src.data import MNIST_MEAN, MNIST_STD
except ImportError:
    from data import MNIST_MEAN, MNIST_STD

try:
    from src.utils import save_json_log
except ImportError:
    from utils import save_json_log


def build_target_labels(labels):
    """
    Строит target labels по ТЗ:
    - если исходный класс не 0 -> target = 0
    - если исходный класс 0 -> target = 1
    """
    target_labels = torch.zeros_like(labels)
    target_labels[labels == 0] = 1
    return target_labels


def get_model_bounds(mean=MNIST_MEAN, std=MNIST_STD):
    """
    Считает допустимые границы значений после нормализации.

    В обычном пиксельном пространстве у нас допустимые значения [0, 1].
    После Normalize(mean, std) эти границы тоже меняются.
    """
    lower = (0.0 - mean[0]) / std[0]
    upper = (1.0 - mean[0]) / std[0]
    return lower, upper


def project_to_linf_ball(x_adv, x_clean, eps):
    """
    Возвращает x_adv обратно в L_infinity-шар радиуса eps вокруг x_clean.
    """
    x_adv = torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps)
    return x_adv


def clamp_to_valid_range(x, mean=MNIST_MEAN, std=MNIST_STD):
    """
    Ограничивает значения так, чтобы картинка оставалась допустимой
    даже после атаки.
    """
    lower, upper = get_model_bounds(mean=mean, std=std)
    x = torch.clamp(x, min=lower, max=upper)
    return x


def pgd_targeted_attack(
    model,
    images,
    labels,
    eps,
    alpha,
    steps,
    device,
    mean=MNIST_MEAN,
    std=MNIST_STD,
    random_start=True,
):
    """
    Делает targeted PGD-атаку для одного батча.

    Параметры:
    - model: обученная модель
    - images: батч нормализованных изображений
    - labels: истинные метки
    - eps: максимальное изменение в пространстве модели
    - alpha: шаг PGD
    - steps: число шагов
    - device: cpu или cuda
    - mean, std: параметры нормализации
    - random_start: начинать ли со случайной точки внутри eps-окрестности

    Возвращает:
    - adv_images: атакованные изображения
    """
    model.eval()

    images = images.to(device)
    labels = labels.to(device)

    target_labels = build_target_labels(labels).to(device)

    x_clean = images.detach().clone()

    if random_start:
        noise = torch.empty_like(x_clean).uniform_(-eps, eps)
        x_adv = x_clean + noise
        x_adv = project_to_linf_ball(x_adv, x_clean, eps)
        x_adv = clamp_to_valid_range(x_adv, mean=mean, std=std)
    else:
        x_adv = x_clean.clone()

    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = criterion(logits, target_labels)

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        loss.backward()

        grad = x_adv.grad.detach()

        with torch.no_grad():
            # Для targeted-атаки мы МИНИМИЗИРУЕМ loss по target-классу.
            # Поэтому идем ПРОТИВ градиента.
            x_adv = x_adv - alpha * grad.sign()

            x_adv = project_to_linf_ball(x_adv, x_clean, eps)
            x_adv = clamp_to_valid_range(x_adv, mean=mean, std=std)

        x_adv = x_adv.detach()

    return x_adv


def evaluate_accuracy(model, loader, device):
    """
    Считает обычную accuracy модели на даталоадере.
    """
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy


def compute_degradation(clean_accuracy, adv_accuracy):
    """
    Считает относительную деградацию по ТЗ.

    Пример:
    clean_accuracy = 0.50
    adv_accuracy   = 0.05

    degradation = (0.50 - 0.05) / 0.50 = 0.90
    """
    if clean_accuracy == 0:
        return 0.0

    degradation = (clean_accuracy - adv_accuracy) / clean_accuracy
    return degradation


def evaluate_targeted_pgd_attack(
    model,
    loader,
    eps,
    alpha,
    steps,
    device,
    mean=MNIST_MEAN,
    std=MNIST_STD,
    random_start=True,
):
    """
    Считает метрики модели на targeted PGD-атаке.

    Возвращает словарь с:
    - clean_accuracy
    - adv_accuracy
    - degradation
    - target_hit_rate
    - eps
    - alpha
    - steps
    """
    model.eval()

    clean_correct = 0
    adv_correct = 0
    target_hits = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        target_labels = build_target_labels(labels).to(device)

        with torch.no_grad():
            clean_logits = model(images)
            clean_predictions = clean_logits.argmax(dim=1)
            clean_correct += (clean_predictions == labels).sum().item()

        adv_images = pgd_targeted_attack(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps,
            device=device,
            mean=mean,
            std=std,
            random_start=random_start,
        )

        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_predictions = adv_logits.argmax(dim=1)

        adv_correct += (adv_predictions == labels).sum().item()
        target_hits += (adv_predictions == target_labels).sum().item()
        total_samples += labels.size(0)

    clean_accuracy = clean_correct / total_samples
    adv_accuracy = adv_correct / total_samples
    degradation = compute_degradation(clean_accuracy, adv_accuracy)
    target_hit_rate = target_hits / total_samples

    results = {
        "clean_accuracy": clean_accuracy,
        "adv_accuracy": adv_accuracy,
        "degradation": degradation,
        "target_hit_rate": target_hit_rate,
        "eps": eps,
        "alpha": alpha,
        "steps": steps,
    }
    return results


def log_attack_results(
    log_path,
    stage_name,
    stage_index,
    train_size,
    seed,
    attack_results,
):
    """
    Логирует результаты атаки в JSON.
    """
    record = {
        "stage_name": stage_name,
        "stage_index": stage_index,
        "train_size": train_size,
        "seed": seed,
        "metric_type": "targeted_pgd_attack",
        "clean_accuracy": attack_results["clean_accuracy"],
        "adv_accuracy": attack_results["adv_accuracy"],
        "degradation": attack_results["degradation"],
        "target_hit_rate": attack_results["target_hit_rate"],
        "eps": attack_results["eps"],
        "alpha": attack_results["alpha"],
        "steps": attack_results["steps"],
    }

    save_json_log(log_path, record)
    return record
