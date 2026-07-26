import torch
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.float()
    dimensions = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dimensions)
    denominator = torch.sum(probabilities + one_hot, dimensions)
    return 1 - ((2 * intersection + 1) / (denominator + 1)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target) + dice_loss(logits, target)
