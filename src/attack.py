import torch

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
    По ТЗ:
    - для всех классов, кроме 0 → цель = 0
    - для класса 0 → цель = 1
    """
    target_labels = torch.zeros_like(labels)
    target_labels[labels == 0] = 1
    return target_labels


def get_model_bounds(mean=MNIST_MEAN, std=MNIST_STD):
    """
    Вычисляет допустимый диапазон значений пикселей ПОСЛЕ нормализации.
    Исходные пиксели [0, 1] → после нормализации [(0-mean)/std, (1-mean)/std]
    """
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return lower, upper


def clamp_to_valid_range(x, mean=MNIST_MEAN, std=MNIST_STD):
    """
    Зажимает тензор в допустимый диапазон нормализованных пикселей.
    """
    lower, upper = get_model_bounds(mean=mean, std=std)
    return torch.clamp(x, min=lower, max=upper)


def project_to_linf_ball(x_adv, x_clean, eps):
    """
    Проекция на L_inf шар: x_adv не может отличаться от x_clean более чем на eps
    по любой координате.

    Мы поэлементно обрабатываем 2 тензора [128, 1, 28, 28] - атакованный и нормальный
    """
    return torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps)


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
    images — оригинальные картинки батча
    labels — истинные метки
    eps — максимальное возмущение (в нормализованном масштабе)
    alpha — шаг одной итерации
    steps — количество итераций
    random_start — начинать со случайной точки внутри шара
    """
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()

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

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)

        logits = model(x_adv)

        # Статья Madry: loss = cross-entropy.
        # Targeted: минимизируем CE по целевому классу → шаг "-"
        loss = ce_loss(logits, target_labels)

        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()

        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            x_adv = project_to_linf_ball(x_adv, x_clean, eps)
            x_adv = clamp_to_valid_range(x_adv, mean=mean, std=std)

    return x_adv.detach()

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
    restarts=1,
):
    """
    Оценивает targeted PGD атаку на всём датасете.

    Возвращает:
    - clean_accuracy: точность на чистых примерах
    - adv_accuracy: точность на adversarial примерах (модель всё ещё права?)
    - degradation: ОТНОСИТЕЛЬНАЯ деградация = (clean_acc - adv_acc) / clean_acc
    - target_hit_rate: доля примеров, где модель предсказала именно целевой класс
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

        # Точность на чистых примерах
        with torch.no_grad():
            clean_logits = model(images)
            clean_predictions = clean_logits.argmax(dim=1)
            clean_correct += (clean_predictions == labels).sum().item()

        # Генерируем adversarial примеры (с несколькими перезапусками для надёжности)
        best_adv_images = None
        best_margin = None

        for _ in range(restarts):
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
                row_ids = torch.arange(adv_logits.size(0), device=device)

                target_logits = adv_logits[row_ids, target_labels]
                other_logits = adv_logits.clone()
                other_logits[row_ids, target_labels] = float("-inf")
                max_other_logits = other_logits.max(dim=1).values

                # Margin: чем больше, тем успешнее атака на этот пример
                margin = target_logits - max_other_logits

            if best_adv_images is None:
                best_adv_images = adv_images.detach().clone()
                best_margin = margin.detach().clone()
            else:
                better_mask = margin > best_margin
                best_margin[better_mask] = margin[better_mask]
                best_adv_images[better_mask] = adv_images[better_mask]

        with torch.no_grad():
            adv_logits = model(best_adv_images)
            adv_predictions = adv_logits.argmax(dim=1)

        adv_correct += (adv_predictions == labels).sum().item()
        target_hits += (adv_predictions == target_labels).sum().item()
        total_samples += labels.size(0)

    clean_accuracy = clean_correct / total_samples
    adv_accuracy = adv_correct / total_samples

    # ВАЖНО: деградация считается ОТНОСИТЕЛЬНАЯ по ТЗ
    degradation = (clean_accuracy - adv_accuracy) / clean_accuracy if clean_accuracy > 0 else 0.0
    target_hit_rate = target_hits / total_samples

    return {
        "clean_accuracy": clean_accuracy,
        "adv_accuracy": adv_accuracy,
        "degradation": degradation,
        "target_hit_rate": target_hit_rate,
        "eps": eps,
        "alpha": alpha,
        "steps": steps,
        "restarts": restarts,
    }


def log_attack_results(log_path, stage_name, stage_index, train_size, seed, attack_results):
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
        "restarts": attack_results.get("restarts", 1),
    }
    save_json_log(log_path, record)
    return record
