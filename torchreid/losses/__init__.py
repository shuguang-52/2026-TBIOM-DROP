from .cross_entropy_loss import CrossEntropyLoss
from .part_averaged_triplet_loss import PartAveragedTripletLoss

__body_parts_losses = {'part_averaged_triplet_loss': PartAveragedTripletLoss}

def init_part_based_triplet_loss(name, **kwargs):
    """Initializes the part based triplet loss based on the part-based distance combination strategy."""
    avai_body_parts_losses = list(__body_parts_losses.keys())
    if name not in avai_body_parts_losses:
        raise ValueError(
            'Invalid loss name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_body_parts_losses)
        )
    return __body_parts_losses[name](**kwargs)


def deep_supervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
