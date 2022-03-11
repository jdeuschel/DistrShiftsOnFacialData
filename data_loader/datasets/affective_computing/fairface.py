"""
Dataloader for various shifts in FairFace
"""

from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from training.data_shifts.rand_augment_fairface import RandAugment_Fairface

from base import BaseDataLoader
import os
import scipy
from random import randrange


from scipy.stats import norm 

__all__ = ["FairFace"]


class FairFace(BaseDataLoader):
    """
    FairFace Data Loader
    =====================

    Task: Age, Gender, Race

    Output:
        Image  - [3, x, y] // Various input sizes 
        Target - [3] (age, gender, race)
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.12,
        num_workers=1,
        training=True,
        target="age",
        **kwargs,
    ):
        self.training = training

        if target == "age":
            self.num_classes = 9
        elif target == "gender":
            self.num_classes = 2
        elif target == "race":
            self.num_classes = 7
        else:
            raise ValueError(f"Target {target} is not defined")

        self.input_size = self._get_input_size()
        self.num_channels = 3
        data_dir, cluster_used = self._get_data_path()
        print("data_dir: ", data_dir)

        rgb_mean = (0.4914, 0.4822, 0.4465)
        rgb_std = (0.2023, 0.1994, 0.2010)


        trsfm = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomCrop(224, padding=4),
            RandAugment_Fairface(n=2,m=8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std)
            ])

        trsfm_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # specify folder here
        PATH = "FairFace_v2" if cluster_used else "FairFace_v2" #todo: specify folder name depending on hardware. in this case they are called the same.


        seed = randrange(1000)
        experiment = kwargs['experiment']

        # if training:
        print(os.path.join(data_dir, PATH))
        self.dataset = FairFaceDataset(
            root_dir=os.path.join(data_dir, PATH), subset="train", transform=trsfm, experiment=experiment, num_classes=self.num_classes, validation_split=validation_split, seed=seed
        )
        self.dataset_val = FairFaceDataset(
            root_dir=os.path.join(data_dir, PATH), subset="val", transform=trsfm_test, experiment=experiment, num_classes=self.num_classes, validation_split=validation_split, seed=seed
        )
        self.dataset_test = FairFaceDataset(
            root_dir=os.path.join(data_dir, PATH), subset="test", transform=trsfm_test, experiment="", num_classes=self.num_classes, validation_split=validation_split, seed=seed
        )
        super().__init__(
            dataset_train=self.dataset,
            dataset_val=self.dataset_val,   # None
            dataset_test=self.dataset_test,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers,
        )


class FairFaceDataset(Dataset):
    def __init__(
        self, root_dir, num_classes, transform=None, subset="train", experiment="", validation_split=0.12, seed=522
    ):
        """
        Parameters
        ----------
        root_dir : str
            Location of FairFace
        num_classes: int
            number of classes depending on target
            Type: [9,2,7]
        transform : [type], optional
            [description], by default None
        subset: str
            Type: ["train", "val"]
        experiment : str, optional
            experiment name
            default: baseline experiment
        validation_split : float, optional
            split between training and validation
        seed: int, optional
            default 522
        """

        self.num_classes=num_classes
        self.root_dir = root_dir
        self.subset = subset
        if self.subset == "train" or self.subset == "val":
            self.data_list = pd.read_csv(os.path.join(self.root_dir, "fairface_label_train.csv"))
        elif self.subset == "test":
            self.data_list = pd.read_csv(os.path.join(self.root_dir, "fairface_label_val.csv"))
        else:
            raise Exception(f"Subset definition {self.subset} does not exist for FairFace")
        mapping_age = {
            "0-2": 0,
            "3-9": 1,
            "10-19": 2,
            "20-29": 3,
            "30-39": 4,
            "40-49": 5,
            "50-59": 6,
            "60-69": 7,
            "more than 70": 8,
        }
        mapping_gender = {"Female": 0, "Male": 1}
        mapping_race = {
            "Black": 0,
            "East Asian": 1,
            "Indian": 2,
            "Latino_Hispanic": 3,
            "Middle Eastern": 4,
            "Southeast Asian": 5,
            "White": 6,
        }

        self.data_list = self.data_list.replace({"race": mapping_race})
        self.data_list = self.data_list.replace({"gender": mapping_gender})
        self.data_list = self.data_list.replace({"age": mapping_age})


        if self.subset == "train" or self.subset == "val":
            train, val = train_test_split(self.data_list, test_size=validation_split, random_state=573) 
            if self.subset == "train":
                self.data_list = train
            elif self.subset == "val":
                self.data_list = val

        if experiment != "" and not self.subset == "val":
            
            #get all experiments on the same size
            train_with_val = True
            if train_with_val:
                self.sample_size_race = 61410  
                self.sample_size_age = 40758
                self.sample_size_gender = 35703 
                self.sample_size_race_sp = 48006
            else:  
                self.sample_size_race = 70217 
                self.sample_size_age = 55592
                self.sample_size_gender = 40758  
                self.sample_size_race_sp = 54482


            # a baseline experiment without any modifications
            if experiment=="verification_baseline": 
                self.data_list = self.data_list

            # baseline experiments with same number of samples as corresponding shifts, to exclude dependency 
            elif experiment=="baseline_race":
                self.data_list = self.data_list.sample(n=self.sample_size_race, replace=False, random_state=seed)

            elif experiment=="baseline_age":
                self.data_list = self.data_list.sample(n=self.sample_size_age, replace=False, random_state=seed)

            elif experiment=="baseline_gender":
                self.data_list = self.data_list.sample(n=self.sample_size_gender, replace=False, random_state=seed)

            # spurious correlations with young and race. possible correlations: if young then race, iff young then race
            elif experiment == 'spurious_correlations_baseline_young':
                # here another baseline is necessary to exclude the sampling bias from the group
                old = 6
                young = 2
                train_data_young = self.data_list[self.data_list["age"]<=young].sample(n=2676, replace=False)
                self.data_list = self.data_list[self.data_list["age"]>young].append(train_data_young).sample(n=self.sample_size_race_sp, replace=False)
            elif experiment == 'spurious_correlations_baseline_old':
                # here another baseline is necessary to exclude the sampling bias from the group
                old = 6
                young = 2
                train_data_old = self.data_list[self.data_list["age"]>=old].sample(n=1243, replace=False)
                self.data_list = self.data_list[self.data_list["age"]<old].append(train_data_old).sample(n=self.sample_size_race_sp, replace=False)



            # gender shifts: 1: no women, 2: no men, 3: less women, 4: less men
            elif experiment in ["split_gen_1", "split_gen_2", "split_gen_3", "split_gen_4"]:
                hard_filter_gender1 = self.data_list[(self.data_list['gender']!=0)]
                hard_filter_gender2 = self.data_list[(self.data_list['gender']!=1)]
                part_soft_gender1, _ = train_test_split(hard_filter_gender1, test_size=round(len(hard_filter_gender2)/2), random_state=seed)
                part_soft_gender2, _ = train_test_split(hard_filter_gender2, test_size=round(len(hard_filter_gender1)/2), random_state=seed)
                soft_filter_gender1 = hard_filter_gender1.append(part_soft_gender2)
                soft_filter_gender2 = hard_filter_gender2.append(part_soft_gender1)
                if experiment=="split_gen_1": 
                    try:
                        self.data_list = hard_filter_gender1.sample(n=self.sample_size_gender, replace=False, random_state=seed) 
                    except:
                        raise Exception("SAMPLE SIZE TOO LARGE") 
                elif experiment=="split_gen_2": 
                    try:
                        self.data_list = hard_filter_gender2.sample(n=self.sample_size_gender, replace=False, random_state=seed)
                    except:
                        raise Exception("SAMPLE SIZE TOO LARGE") 
                elif experiment=="split_gen_3": 
                    self.data_list = self.data_list = soft_filter_gender1.sample(n=self.sample_size_gender, replace=False, random_state=seed)
                elif experiment=="split_gen_4": 
                    self.data_list =  self.data_list = soft_filter_gender2.sample(n=self.sample_size_gender, replace=False, random_state=seed)


            # sampling bias with one underrepresented (partly erased until erased) race group, p={1.0, 0.75, 0.5, 0.25, 0.0}
            elif experiment.startswith("data_shift"):
                # please write something like experiment="data_shift_Black_0.25"
                if "Black" in experiment:
                    race = 0
                elif "East_Asian" in experiment:
                    race = 1
                elif "Indian" in experiment:
                    race = 2
                elif "Latino_Hispanic" in experiment:
                    race = 3
                elif "Middle_Eastern" in experiment:
                    race = 4
                elif "Southeast_Asian" in experiment:
                    race = 5
                elif "White" in experiment:
                    race = 6
                    
                if "0.0" in experiment:
                    frac = 0.0         
                elif "0.25" in experiment:
                    frac = 0.25
                elif "0.5" in experiment:
                    frac = 0.5
                elif "0.75" in experiment:
                    # pdb.set_trace()
                    frac = 0.75
                elif "1.0" in experiment:
                    frac = 1.0
                try:
                    self.data_list = self.data_list_drop(data=self.data_list, race=race, frac=frac, seed=seed).sample(n=self.sample_size_race,
                                                                                                                    replace=False,
                                                                                                                    random_state=seed)
                except:
                    print("sampling did not work for", race, frac)  
                    raise Exception("SAMPLE SIZE TOO LARGE") 
                    
            # left label bias: makes a specific race group younger. Uses Gauss distribution with sigma={0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}
            elif experiment.startswith("left_label_shift"):
                # write something like: "left_label_shift_Black_1.5"
                if "Black" in experiment:
                    race = [0]
                elif "East_Asian" in experiment:
                    race = [1]
                elif "Indian" in experiment:
                    race = [2]
                elif "Latino_Hispanic" in experiment:
                    race = [3]
                elif "Middle_Eastern" in experiment:
                    race = [4]
                elif "Southeast_Asian" in experiment:
                    race = [5]
                elif "White" in experiment:
                    race = [6]
                std = float(experiment[-3:])
                self.data_list = self.left_racial_label_bias(self.data_list, race_li=race, std=std, seed=seed).sample(n=self.sample_size_race, 
                                                                                                                        replace=False, 
                                                                                                                        random_state=seed)



            elif experiment.startswith('spurious_correlations'):
                old = 6
                young = 2
                self.data_list = self.data_list.reset_index(drop=True)

                # 1a: first kind of spurious correlations: if old then race
                if experiment.startswith('spurious_correlations_old_0'):
                    spur_old_0_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 0))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_old_0_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_old_1'):
                    spur_old_1_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 1))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_old_1_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_old_2'):
                    spur_old_2_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 2))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_old_2_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_old_3'):
                    spur_old_3_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 3))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_old_3_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_old_4'):
                    spur_old_4_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 4))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_old_4_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_old_5'):
                    spur_old_5_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 5))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_old_5_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_old_6'):
                    spur_old_6_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 6))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_old_6_all.sample(n=self.sample_size_race_sp, replace=False)

                # 1b first kind of spurious correlations: if young then race
                elif experiment.startswith('spurious_correlations_young_0'):
                    spur_young_0_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 0))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_young_0_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_young_1'):
                    spur_young_1_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 1))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_young_1_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_young_2'):
                    spur_young_2_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 2))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_young_2_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_young_3'):
                    spur_young_3_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 3))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_young_3_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_young_4'):
                    spur_young_4_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 4))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_young_4_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_young_5'):
                    spur_young_5_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 5))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_young_5_all.sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_young_6'):
                    spur_young_6_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 6))).flatten()).reset_index(
                            drop=True)
                    self.data_list = spur_young_6_all.sample(n=self.sample_size_race_sp, replace=False)

                # 2a: second kind of spurious correlations: iff old then race
                elif experiment.startswith('spurious_correlations_iff_old_0'):
                    spur_old_0_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 0))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_old_0_all.drop(np.array(
                        np.where((spur_old_0_all['age'] < old) & (spur_old_0_all['race'] == 0))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_old_1'):
                    spur_old_1_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 1))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_old_1_all.drop(np.array(
                        np.where((spur_old_1_all['age'] < old) & (spur_old_1_all['race'] == 1))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_old_2'):
                    spur_old_2_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 2))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_old_2_all.drop(np.array(
                        np.where((spur_old_2_all['age'] < old) & (spur_old_2_all['race'] == 2))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_old_3'):
                    spur_old_3_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 3))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_old_3_all.drop(np.array(
                        np.where((spur_old_3_all['age'] < old) & (spur_old_3_all['race'] == 3))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_old_4'):
                    spur_old_4_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 4))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_old_4_all.drop(np.array(
                        np.where((spur_old_4_all['age'] < old) & (spur_old_4_all['race'] == 4))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_old_5'):
                    spur_old_5_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 5))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_old_5_all.drop(np.array(
                        np.where((spur_old_5_all['age'] < old) & (spur_old_5_all['race'] == 5))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_old_6'):
                    spur_old_6_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] >= old) & (self.data_list['race'] != 6))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_old_6_all.drop(np.array(
                        np.where((spur_old_6_all['age'] < old) & (spur_old_6_all['race'] == 6))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)

                # 2b: second kind of spurious correlations: iff young then race
                elif experiment.startswith('spurious_correlations_iff_young_0'):
                    spur_young_0_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 0))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_young_0_all.drop(np.array(np.where(
                        (spur_young_0_all['age'] > young) & (spur_young_0_all['race'] == 0))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_young_1'):
                    spur_young_1_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 1))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_young_1_all.drop(np.array(np.where(
                        (spur_young_1_all['age'] > young) & (spur_young_1_all['race'] == 1))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_young_2'):
                    spur_young_2_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 2))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_young_2_all.drop(np.array(np.where(
                        (spur_young_2_all['age'] > young) & (spur_young_2_all['race'] == 2))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_young_3'):
                    spur_young_3_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 3))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_young_3_all.drop(np.array(np.where(
                        (spur_young_3_all['age'] > young) & (spur_young_3_all['race'] == 3))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_young_4'):
                    spur_young_4_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 4))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_young_4_all.drop(np.array(np.where(
                        (spur_young_4_all['age'] > young) & (spur_young_4_all['race'] == 4))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_young_5'):
                    spur_young_5_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 5))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_young_5_all.drop(np.array(np.where(
                        (spur_young_5_all['age'] > young) & (spur_young_5_all['race'] == 5))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)
                elif experiment.startswith('spurious_correlations_iff_young_6'):
                    spur_young_6_all = self.data_list.drop(np.array(
                        np.where(
                            (self.data_list['age'] <= young) & (self.data_list['race'] != 6))).flatten()).reset_index(
                        drop=True)
                    self.data_list = spur_young_6_all.drop(np.array(np.where(
                        (spur_young_6_all['age'] > young) & (spur_young_6_all['race'] == 6))).flatten()).reset_index(
                        drop=True).sample(n=self.sample_size_race_sp, replace=False)

            else:
                print("NO SHIFT SELECTED")
                raise Exception("NO SHIFT SELECTED") 
                



        self.path_list = np.array(self.data_list.iloc[:, 0].tolist())
        self.age = np.array(self.data_list.iloc[:,]["age"].tolist())
        self.gender = np.array(self.data_list.iloc[:,]["gender"].tolist())
        self.race = np.array(self.data_list.iloc[:,]["race"].tolist())
        self.transform = transform



    def __len__(self):
        return len(self.path_list)

    def get_vector(self, targets, nb_classes):
        return np.ones(nb_classes)*targets

    def data_list_drop(self, data, race, frac, seed=123):
        data_list = data.copy(deep=True)
        frac = 1-frac
        data_drop = data_list[data_list['race'] == race].sample(frac=frac, random_state=seed)
        data_ret = data_list.drop(data_drop.index)
        return data_ret

    def left_racial_label_bias(self, data_label, race_li, seed=123, std=1.5):     
        def left_labels_normal(ages, std=1.5, seed=123):
            np.random.seed(seed)
            new_ages = np.zeros(ages.shape, dtype=np.int64)
            x = np.unique(ages)
            for num, mean in enumerate(ages):
                normal = norm(loc=mean, scale=std)
                prob = normal.pdf(x)
                prob[np.argwhere(x>mean)]=0 # set probabilitie greater than mean to zero
                prob = prob / prob.sum() # normalize the probabilities so their sum is 1
                new_ages[num] = np.random.choice(x, p = prob)
            return new_ages

        data_my = data_label.copy(deep=True)
        for race_i in race_li:
            race = data_my[data_my.race.isin([race_i])].copy(deep=True)
            race["age"] = left_labels_normal(ages=race["age"], seed=seed, std=std) 
            data_my.update(race)
            data_my["age"] = data_my["age"].astype(np.int64)
            data_my["gender"] = data_my["gender"].astype(np.int64)
            data_my["race"] = data_my["race"].astype(np.int64)
        return data_my





    def __getitem__(self, idx):

        img_file = self.path_list[idx]
        age, gender, race = self.age[idx], self.gender[idx], self.race[idx]
 
        image_path = os.path.join(self.root_dir, img_file) 

        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)
        return img, age, self.get_vector(age, self.num_classes), gender, race


