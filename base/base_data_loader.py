import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils import data
from torchvision import transforms
import logging
from copy import copy
from parse_config import ConfigParser


class BaseDataLoader:
    """
    Base class for all data loaders
    """

    def __init__(
        self,
        dataset_train,
        dataset_val=None,
        dataset_test=None,
        batch_size=None,
        shuffle=True,
        validation_split=0.0,
        num_workers=0,
        collate_fn=default_collate,
    ):

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

        self.validation_split = validation_split
        self.batch_idx = 0
        self.n_samples = len(dataset_train)
        self.config = ConfigParser()
        self.mode = os.environ["MODE"]
        self.drop_last = False
        self.shuffle = self.config["data_loader"]["args"]["shuffle"]

        if "drop_last" in self.config["data_loader"]["args"]:
            self.drop_last = self.config["data_loader"]["args"]["drop_last"]

        # Augmentations
        self.apply_augmentation = self.training and self.mode == "TRAINING"
        self.augmentation_applied = False
        self.input_size = self.config["input_size"]

        self.current_transform = self.dataset_train.transform  # Transform before augmentation


        # Set num_classes for model generation
        if "num_classes" not in self.config["arch"]["args"]:
            self.config["arch"]["args"]["num_classes"] = self.num_classes  # From sub class

        # Set num_channels for model generation
        if "num_channels" not in self.config["arch"]["args"]:
            self.config["arch"]["args"]["num_channels"] = self.num_channels  # From sub class

        if self.dataset_val == None and self.validation_split != 0.0:
            self.dataset_train, self.dataset_val = self._split_sampler(self.validation_split)
            self.dataset_val.dataset = copy(self.dataset_val.dataset)

        # Training
        self.init_kwargs = {
            "dataset": self.dataset_train,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "drop_last": self.drop_last,
        }

        self.training_loader = DataLoader(**self.init_kwargs)

        # Test
        if self.dataset_test != None:
            self.init_kwargs = {
                "dataset": self.dataset_test,
                "batch_size": batch_size,
                "shuffle": False,
                "collate_fn": collate_fn,
                "num_workers": num_workers,
                "drop_last": self.drop_last,
            }

            self.test_loader = DataLoader(**self.init_kwargs)

        # Validation
        if self.dataset_val != None:
            self.init_kwargs_val = {
                "dataset": self.dataset_val,
                "batch_size": batch_size,
                "shuffle": False,
                "collate_fn": collate_fn,
                "num_workers": num_workers,
                "drop_last": self.drop_last,
            }

            self.validation_loader = DataLoader(**self.init_kwargs_val)

    def get_training_loader(self):
        return self.training_loader

    def get_validation_loader(self):
        try:
            return self.validation_loader
        except Exception as e:
            print("Validation data is not defined:" + repr(e))

    def get_test_loader(self):
        try:
            return self.test_loader
        except Exception as e:
            print("Test data is not defined:" + repr(e))

    @staticmethod
    def _get_config():
        config = ConfigParser()
        return config

    @staticmethod
    def _get_input_size():
        config = ConfigParser()
        print("Using custom input size.")
        return config["input_size"]

    @staticmethod
    def _get_data_path():
        cluster_used = True if "HOSTNAME" in os.environ else False
        DATA_PATH = os.environ["DATA_PATH_NETAPP"]  # todo: specify path to data here, replace by "/folder/name/data/"
        return DATA_PATH,  cluster_used

    @staticmethod
    def _get_imagenet_normalization():
        normalze_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalze_imagenet

    def _split_sampler(self, split: float):
        if split == 0.0:
            return self.dataset_train, None

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        train_set, val_set = data.random_split(self.dataset_train, [(self.n_samples - len_valid), len_valid])

        return train_set, val_set

    def split_validation(self):
        if self.val_set is None:
            return None
        else:
            if self.cutmix:
                self.val_set.dataset = copy(self.val_set.dataset.dataset)  

            self.val_set.dataset = copy(self.val_set.dataset)
            if hasattr(self, "trsfm_test"):
                self.val_set.dataset.transform = self.trsfm_test
            else:
                self.val_set.dataset.transform = self.trsfm

            val_kwargs = self.init_kwargs
            val_kwargs["dataset"] = self.val_set
            val_kwargs[
                "shuffle"
            ] = False  # NOTE: For validation data no shuffling is used because of bayesian model averaging!
            return DataLoader(**val_kwargs)
