
from pathlib import Path

import torch

try:
    from src.utils import save_json_log
except ImportError:
    from utils import save_json_log


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    loader — DataLoader, подаёт батчи по 128 картинок
    optimizer — алгоритм обновления весов (Adam)
    criterion — функция потерь (CrossEntropyLoss)
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        # labels - числа
        optimizer.zero_grad() # обнуляем

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward() # градиенты для всех весов
        optimizer.step() # шаг по adam через моменты
        """
        момент 1 порядка - средний градиент за последние шаги
        момент 2 порядка - средний квадрат градиента за последние шаги

        шаг = lr * момент_1 / sqrt(момент_2)
        w_новый = w_старый - шаг
        """
        
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }


def evaluate(model, loader, criterion, device):
    """
    оценка качества

    нельзя делать на обучающих данных, делаем на тестовых
    """
    
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    train_metrics,
    test_metrics,
    stage_name,
    stage_index=None,
    train_size=None,
    seed=None,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(), # все веса
        "optimizer_state_dict": optimizer.state_dict(), # состояние оптимизатора (моменты)
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "stage_name": stage_name,
        "stage_index": stage_index,
        "train_size": train_size,
        "seed": seed,
    }

    torch.save(checkpoint, checkpoint_path)


def train_stage(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    epochs,
    checkpoint_path,
    log_path,
    stage_name,
    stage_index=None,
    train_size=None,
    seed=None,
):
    model.to(device)

    best_test_accuracy = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        is_best = test_metrics["accuracy"] > best_test_accuracy
        # флаг стали ли лучше
        if is_best:
            best_test_accuracy = test_metrics["accuracy"]
            best_epoch = epoch

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                stage_name=stage_name,
                stage_index=stage_index,
                train_size=train_size,
                seed=seed,
            )

        log_record = {
            "stage_name": stage_name,
            "stage_index": stage_index,
            "epoch": epoch,
            "train_size": train_size,
            "seed": seed,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "best_test_accuracy_so_far": best_test_accuracy,
            "best_epoch_so_far": best_epoch,
            "saved_best_checkpoint": is_best,
            "checkpoint_path": str(checkpoint_path),
        }

        history.append(log_record)

        if log_path is not None:
            save_json_log(log_path, log_record)

        print(
            f"[{stage_name}] "
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

    result = {
        "stage_name": stage_name,
        "stage_index": stage_index,
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }

    return result
