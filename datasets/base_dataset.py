"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
#from PIL import Image
import cv2
#import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from albumentations import HorizontalFlip, RandomResizedCrop, Compose, DualTransform
import albumentations.augmentations.transforms as transforms

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataset_root #mia

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

class HCompose(Compose):
    def __init__(self, transforms, *args, additional_targets=None, no_nearest_for_masks=True, **kwargs):
        if additional_targets is None:
            additional_targets = {
                'real': 'image',
                # 'mask': 'mask'
            }
        self.additional_targets = additional_targets
        super().__init__(transforms, *args, additional_targets=additional_targets, **kwargs)
        # if no_nearest_for_masks:
        #     for t in transforms:
        #         if isinstance(t, DualTransform):
        #             t._additional_targets['mask'] = 'image'
                    # t._additional_targets['edge'] = 'image'


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, convert=True, additional_targets=None):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.ToGray())
    if opt.preprocess == 'resize_and_crop':
        if params is None:
            transform_list.append(RandomResizedCrop(opt.crop_size, opt.crop_size, scale=(0.9, 1.0))) # 0.5,1.0
    elif opt.preprocess == 'resize':
        transform_list.append(transforms.Resize(opt.input_size, opt.input_size))
    elif opt.preprocess == 'none':
        return HCompose(transform_list)

    if not opt.no_flip:
        if params is None:
            # print("flip")
            transform_list.append(HorizontalFlip())

    return HCompose(transform_list, additional_targets=additional_targets)

def __make_power_2(img, base):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)


'''
def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
'''

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
