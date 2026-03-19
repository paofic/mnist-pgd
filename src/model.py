import torch.nn as nn # слои, nn.Module, Linear
from torchvision.models import resnet18, ResNet18_Weights

'''
    nn.Module — это базовый класс для всех нейросетей в PyTorch.

    Почему это нужно:

        1) PyTorch понимает, что это нейросеть;
        2) внутри такой модели можно хранить слои;
        3) PyTorch умеет находить параметры модели;
        4) модель можно перенести на GPU через .to(device);
        5) модель можно переключать в .train() и .eval();
        6) можно сохранять и загружать веса.
'''


class MNISTResNet18(nn.Module):
    def __init__(
        self,
        num_classes=10,
        pretrained=True,
        freeze_backbone=True,
    ):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet18(weights=weights)

'''
        Первый слой меняем под MNIST:
        было 3 канала (RGB), стало 1 канал (grayscale)
        также делаем более "мягкий" старт для маленьких картинок 28x28
        Я возьму чёрно-белую картинку и прогоню по ней 64 маленьких фильтра размера 3x3.
        Каждый фильтр будет искать какой-то свой шаблон
'''
        
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3, 
            stride=1, # шаг
            padding=1, #рамка толщиной 1 клетку заполненная нулями
            bias=False,
        )

        # Для MNIST стандартный maxpool в начале слишком сильно уменьшает картинку.
        # Поэтому заменяем его на Identity: ничего не делает.
        self.backbone.maxpool = nn.Identity()

        # Последний слой меняем под 10 классов MNIST
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # По ТЗ надо заморозить convolution-слои.
        # Практически удобно заморозить весь backbone,
        # а новый первый и последний слои оставить обучаемыми.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            for param in self.backbone.conv1.parameters():
                param.requires_grad = True

            for param in self.backbone.fc.parameters():
                param.requires_grad = True
'''
    Как вход проходит через модель
'''

    def forward(self, x):
        logits = self.backbone(x)
        return logits


def build_model(
    num_classes=10,
    pretrained=True,
    freeze_backbone=True,
):
    model = MNISTResNet18(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    return model
