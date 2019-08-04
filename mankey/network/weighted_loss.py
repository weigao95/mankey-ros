import torch


def weighted_mse_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        size_avg: bool = True) -> torch.Tensor:
    """
    Compute the weight mes loss given torch tensor
    :param pred: (batch_size, -1)
    :param target: (batch_size, -1)
    :param weight: should be the same size as pred or target
    :param size_avg: divided by the batch_size or not
    :return: scalar loss function
    """
    out = (pred - target) ** 2
    out = out * weight
    if size_avg:
        return out.sum() / len(pred)
    else:
        return out.sum()


def weighted_l1_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        size_avg: bool = True) -> torch.Tensor:
    """
    Compute the weight l1 loss given torch tensor
    :param pred: (batch_size, -1)
    :param target: (batch_size, -1)
    :param weight:
    :param size_avg: divided by the batch_size or not
    :return:
    """
    out = torch.abs(pred - target)
    out = out * weight
    if size_avg:
        return out.sum() / len(pred)
    else:
        return out.sum()
