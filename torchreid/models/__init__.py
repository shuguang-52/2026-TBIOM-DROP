"""Public model registry for the HRNet + FPN DROP release."""
from .bpbreid import BPBreID, bpbreid
from .hrnet import hrnet32

# The historical name is kept so existing configurations remain usable.
drop = bpbreid
__model_factory = {'drop': drop, 'bpbreid': bpbreid, 'hrnet32': hrnet32}


def show_avai_models():
    """Display the models included in this release."""
    print(list(__model_factory))


def build_model(name, num_classes, loss='part_based', pretrained=True,
                use_gpu=True, **kwargs):
    """Construct DROP or its HRNet-W32 backbone.

    Device placement remains under the caller's control, including on CPU.
    """
    if name not in __model_factory:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(
            name, list(__model_factory)))
    return __model_factory[name](num_classes=num_classes, loss=loss,
                                 pretrained=pretrained, use_gpu=use_gpu, **kwargs)
