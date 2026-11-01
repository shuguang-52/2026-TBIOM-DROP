import copy
import os
from .image import Market1501, DukeMTMCreID, OccludedDuke, OccludedReID, PDukemtmcReid, MSMT17
from .dataset import Dataset, ImageDataset

__image_datasets = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID,
    'occluded_duke': OccludedDuke,
    'occluded_reid': OccludedReID,
    'p_dukemtmc_reid': PDukemtmcReid,
    'msmt17': MSMT17,
}
__datasets_nicknames = {'market1501': 'mk', 'dukemtmcreid': 'du', 'occluded_duke': 'od', 'occluded_reid': 'or', 'p_dukemtmc_reid': 'pd', 'msmt17': 'ms'}
__datasets_cache = {}

def configure_dataset_class(clazz, **ext_kwargs):
    """
    Wrapper function to provide the class with args external to torchreid
    """
    class ClazzWrapper(clazz):
        def __init__(self, **kwargs):
            self.__name__ = clazz.__name__
            super(ClazzWrapper, self).__init__(**{**kwargs, **ext_kwargs})

    ClazzWrapper.__name__ = clazz.__name__

    return ClazzWrapper


def get_dataset_nickname(name):
    return __datasets_nicknames.get(name, name)


def get_image_dataset(name):
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __image_datasets[name]





def init_image_dataset(name, mode='train', **kwargs):
    """
    Initializes an image dataset.
    The copy.copy() was introduced to fix Torchreid implementing multiple times the same dataset.
    In Datamanager, each dataset was instantiated multiple times via 'init_image_dataset': one for train, one for query
    and one for gallery. Each instance had its own 'data' field containing either train, query or gallery set, based on
    the 'mode' field passed as argument, and its own transforms, to perform training time or test time data transformation.
    However, instantiating the same dataset multiple times is not efficient, as it requires to load the dataset metadata from
    disk multiple times. Moreover, other printing (such as dataset summary) are displayed multiple times.
    To fix this, we copy the dataset class but not its contained objects (such as train/query/gallery) and set a new 'mode' on each copy.
    Thanks to that hack, the data list is created only once, and only the Dataset class is instantiated multiple times
    (for each 'mode'). Therefore, each Dataset uses the same data lists in the background, switching
    between train, query and gallery based on the 'mode' field.
    """
    cache_key = (name, os.path.abspath(os.path.expanduser(kwargs.get('root', ''))), kwargs.get('masks_dir'), kwargs.get('combineall', False))
    if cache_key in __datasets_cache:
        print("Using cached dataset {}.".format(name))
        dataset = __datasets_cache[cache_key]
    else:
        print("Creating new dataset {} and add it to the datasets cache.".format(name))
        dataset = get_image_dataset(name)(mode=mode, **kwargs)
        __datasets_cache[cache_key] = dataset
    mode_dataset = copy.copy(dataset)
    mode_dataset.mode = mode
    for key in ('config', 'transform_tr', 'transform_te', 'use_masks', 'load_masks'):
        if key in kwargs:
            setattr(mode_dataset, 'cfg' if key == 'config' else key, kwargs[key])
    return mode_dataset




def register_image_dataset(name, dataset, nickname=None):
    """Registers a new image dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::

        import torchreid
        import NewDataset
        torchreid.data.register_image_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources=['new_dataset', 'dukemtmcreid']
        )
    """
    global __image_datasets
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __image_datasets[name] = dataset
    __datasets_nicknames[name] = nickname if nickname is not None else name
