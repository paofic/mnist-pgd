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
    target_labels = torch.zeros_like(labels)
    target_labels[labels == 0] = 1
    return target_labels


def get_model_bounds(mean=MNIST_MEAN, std=MNIST_STD):
    lower = (0.0 - mean[0]) / std[0]
    upper = (1.0 - mean[0]) / std[0]
    return lower, upper


def clamp_to_valid_range(x, mean=MNIST_MEAN, std=MNIST_STD):
    lower, upper = get_model_bounds(mean=mean, std=std)
    x = torch.clamp(x, min=lower, max=upper)
    return x


def project_to_linf_ball(x_adv, x_clean, eps):
    x_adv = torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps)
    return x_adv


def targeted_margin_loss(logits, target_labels):
    batch_size = logits.size(0)
    device = logits.device

    row_ids = torch.arange(batch_size, device=device)

    target_logits = logits[row_ids, target_labels]

    other_logits = logits.clone()
    other_logits[row_ids, target_labels] = float("-inf")
    max_other_logits = other_logits.max(dim=1).values

    loss = (max_other_logits - target_logits).mean()
    return loss


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

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = targeted_margin_loss(logits, target_labels)

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        loss.backward()

        grad = x_adv.grad.detach()

        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            x_adv = project_to_linf_ball(x_adv, x_clean, eps)
            x_adv = clamp_to_valid_range(x_adv, mean=mean, std=std)

        x_adv = x_adv.detach()

    return x_adv


def evaluate_accuracy(model, loader, device):
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
    restarts=1,
):
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
        "restarts": restarts,
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
