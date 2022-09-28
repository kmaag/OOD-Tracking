from email.mime import base
import os
import numpy as np
from matplotlib.style import available
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
from pathlib import Path
from natsort import natsorted
from collections import namedtuple
from typing import Union, List, Any, Tuple, Optional, Callable
from .utils import verify_str_arg

def convert_target(target_list, targets_basenames):
    target = []
    j=0
    for i in range(len(targets_basenames)):
        if targets_basenames[i] == True:
            target.append(target_list[j])
            j +=1
        else:
            target.append(None)
    return target

class sos():
    """
    Dataset class for the Street Obstacle Sequences dataset
    Args:
        root (string): Root directory of dataset
        sequence (string, optional): The image sequences to load
        target_type (string or list, optional): Type of target to use, choose from ("semantic_ood", "instance_ood", "depth_ood", "semantic").
        transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    def __init__(self, root: str, sequences: str = ["all"],  target_type: Union[List[str], str] = "semantic_ood", transforms: Optional[Callable] = None):
        self.root = root
        self.images = []
        self.all_images = []
        self.targets_semantic_ood = []
        self.targets_instance_ood = []
        self.targets_depth_ood = []
        self.targets_semantic = []
        self.basenames = []
        self.all_basenames = []
        self.ood_id = 254
        self.target_type = target_type
        self.transforms = transforms
        self.ood_classes = np.arange(244, 255)
        self.id_dict = {'sequence_001': [1], 
                        'sequence_002': [1],
                        'sequence_003': [1],
                        'sequence_004': [1],
                        'sequence_005': [1],
                        'sequence_006': [1],
                        'sequence_007': [1],
                        'sequence_008': [1],
                        'sequence_009': [1],
                        'sequence_010': [1],
                        'sequence_011': [1],
                        'sequence_012': [1],
                        'sequence_013': [1],
                        'sequence_014': [1],
                        'sequence_015': [1, 2],
                        'sequence_016': [1, 2],
                        'sequence_017': [1, 2],
                        'sequence_018': [1, 2],
                        'sequence_019': [1, 2],
                        'sequence_020': [1, 2]}

        
        if not isinstance(target_type, list):
            self.target_type = [target_type]
        available_target_types = ("semantic_ood", "instance_ood", "depth_ood", "semantic")
        [verify_str_arg(value, "target_type", available_target_types) for value in self.target_type]
        
        if sequences is None or "all" in [str(s).lower() for s in sequences]:
            self.sequences = []
            for sequence in (Path(self.root) / "raw_data").glob("sequence*"):
                self.sequences.append(str(sequence.name))
        elif all(isinstance(s, int) for s in sequences):
            self.sequences = []
            for s in sequences:
                self.sequences.append("sequence_" + str(s).zfill(3))
        else:
            self.sequences = sequences
        self.sequences = natsorted(self.sequences)
        
        for sequence in self.sequences:
            sequence_images_dir = Path(self.root) / "raw_data" / sequence
            sequence_semantic_ood_dir = Path(self.root) / "semantic_ood" / sequence
            sequence_instance_ood_dir = Path(self.root) / "instance_ood" / sequence
            sequence_depth_ood_dir = Path(self.root) / "depth_ood" / sequence
            sequence_semantic_dir = Path(self.root) / "semantic" / sequence
            
            sequence_basenames = []
            for file_path in sequence_semantic_ood_dir.glob("*_semantic_ood.png"):
                sequence_basenames.append(str(Path(sequence) / f"{file_path.stem}").replace("_semantic_ood", ""))
            sequence_basenames = natsorted(sequence_basenames)
            for basename in sequence_basenames:
                self.basenames.append(basename)
                self.images.append(str(sequence_images_dir / f"{Path(basename).stem}_raw_data.jpg"))
                self.targets_semantic_ood.append(str(sequence_semantic_ood_dir / f"{Path(basename).stem}_semantic_ood.png"))
                self.targets_instance_ood.append(str(sequence_instance_ood_dir / f"{Path(basename).stem}_instance_ood.png"))
                self.targets_depth_ood.append(str(sequence_depth_ood_dir / f"{Path(basename).stem}_depth_ood.png"))
                self.targets_semantic.append(str(sequence_semantic_dir / f"{Path(basename).stem}_semantic.png"))
                

            for file_path in sequence_images_dir.glob("*.jpg"):
                self.all_images.append(str(file_path))
                self.all_basenames.append(str(Path(sequence) / file_path.stem.replace('_raw_data','')))
                
        self.all_images = natsorted(self.all_images)
        self.all_basenames = natsorted(self.all_basenames)
        self.targets_basenames = [ self.all_basenames[i] in self.basenames for i in range(len(self.all_basenames))]
        self.targets_semantic_ood = convert_target(self.targets_semantic_ood, self.targets_basenames)
        self.targets_instance_ood = convert_target(self.targets_instance_ood, self.targets_basenames)
        self.targets_depth_ood = convert_target(self.targets_depth_ood, self.targets_basenames)
        self.targets_semantic = convert_target(self.targets_semantic, self.targets_basenames)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Draw one input image of dataset
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item, otherwise target is the OOD segmentation by default.
        """
        image = Image.open(self.all_images[index]).convert('RGB')
        
        targets: Any = []
        for target_type in self.target_type:
            if target_type == "semantic_ood":
                if self.targets_semantic_ood[index] == None:
                    target = None
                else:
                    target = Image.open(self.targets_semantic_ood[index])
            elif target_type == "instance_ood":
                if self.targets_instance_ood[index] == None:
                    target = None
                else:
                    target = Image.open(self.targets_instance_ood[index])
            elif target_type == "depth_ood":
                if self.targets_depth_ood[index] == None:
                    target = None
                else:
                    target = Image.open(self.targets_depth_ood[index])
            elif target_type == "semantic":
                if self.targets_semantic[index] == None:
                    target = None
                else:
                    target = Image.open(self.targets_semantic[index])
            targets.append(target)
        
        target = tuple(targets) if len(targets) > 1 else targets[0]
        
        if self.transforms is not None:
            """transform applied to input and its targets"""
            image, target = self.transforms(image, target)
            
        return image, target

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)

    def __repr__(self):
        """Print some information about the dataset."""
        fmt_str = "DATASET: Street Obstacle Scenes\n---- Dir %s\n" % self.root
        fmt_str += '---- Num found images: %d\n' % len(self.images)
        return fmt_str.strip()


class carla():
    """
    Dataset loader for the Carla Wildlife Sequences dataset
    """
    def __init__(self, root: str, sequences: str = ["all"],  target_type: Union[List[str], str] = "semantic_ood", transforms: Optional[Callable] = None):
        self.root = root
        self.images = []
        self.all_images = []
        self.targets_semantic_ood = []
        self.targets_instance_ood = []
        self.targets_depth_ood = []
        self.targets_semantic = []
        self.basenames = []
        self.all_basenames = []
        self.ood_id = 254
        self.target_type = target_type
        self.transforms = transforms
        self.ood_classes = np.arange(224, 244)
        self.id_dict = {'sequence_001': [1], 
                        'sequence_002': [1, 2],
                        'sequence_003': [1, 2],
                        'sequence_004': [1],
                        'sequence_005': [1, 2, 3, 4],
                        'sequence_006': [1, 2],
                        'sequence_007': [1, 2],
                        'sequence_008': [1],
                        'sequence_009': [1],
                        'sequence_010': [1, 2, 3],
                        'sequence_011': [1, 2, 3],
                        'sequence_012': [1, 2],
                        'sequence_013': [1, 2],
                        'sequence_014': [1],
                        'sequence_015': [1, 2, 3, 4],
                        'sequence_016': [1],
                        'sequence_017': [1, 2, 3, 4],
                        'sequence_018': [1, 2, 3],
                        'sequence_019': [1, 2, 3],
                        'sequence_020': [1, 2, 3],
                        'sequence_021': [1, 2, 3, 4],
                        'sequence_022': [1],
                        'sequence_023': [1, 2, 3, 4],
                        'sequence_024': [1],
                        'sequence_025': [1, 2, 3, 4],
                        'sequence_026': [1, 2, 3]}

        
        if not isinstance(target_type, list):
            self.target_type = [target_type]
        available_target_types = ("semantic_ood", "instance_ood", "depth_ood", "semantic")
        [verify_str_arg(value, "target_type", available_target_types) for value in self.target_type]
        
        if sequences is None or "all" in [str(s).lower() for s in sequences]:
            self.sequences = []
            for sequence in (Path(self.root) / "raw_data").glob("sequence*"):
                self.sequences.append(str(sequence.name))
        elif all(isinstance(s, int) for s in sequences):
            self.sequences = []
            for s in sequences:
                self.sequences.append("sequence_" + str(s).zfill(3))
        else:
            self.sequences = sequences
        self.sequences = natsorted(self.sequences)
        
        for sequence in self.sequences:
            sequence_images_dir = Path(self.root) / "raw_data" / sequence
            sequence_semantic_ood_dir = Path(self.root) / "semantic_ood" / sequence
            sequence_instance_ood_dir = Path(self.root) / "instance_ood" / sequence
            sequence_depth_ood_dir = Path(self.root) / "depth_ood" / sequence
            sequence_semantic_dir = Path(self.root) / "semantic" / sequence
            
            sequence_basenames = []
            for file_path in sequence_semantic_ood_dir.glob("*_semantic_ood.png"):
                sequence_basenames.append(str(Path(sequence) / f"{file_path.stem}").replace("_semantic_ood", ""))
            sequence_basenames = natsorted(sequence_basenames)
            for basename in sequence_basenames:
                self.basenames.append(basename)
                self.images.append(str(sequence_images_dir / f"{Path(basename).stem}_raw_data.png"))
                self.targets_semantic_ood.append(str(sequence_semantic_ood_dir / f"{Path(basename).stem}_semantic_ood.png"))
                self.targets_instance_ood.append(str(sequence_instance_ood_dir / f"{Path(basename).stem}_instance_ood.png"))
                self.targets_depth_ood.append(str(sequence_depth_ood_dir / f"{Path(basename).stem}_depth_ood.png"))
                self.targets_semantic.append(str(sequence_semantic_dir / f"{Path(basename).stem}_semantic.png"))
                
            for file_path in sequence_images_dir.glob("*.png"):
                self.all_images.append(str(file_path))
                self.all_basenames.append(str(Path(sequence) / file_path.stem.replace('_raw_data','')))
                
        self.all_images = natsorted(self.all_images)
        self.all_basenames = natsorted(self.all_basenames)
                

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Draw one input image of dataset
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item, otherwise target is the OOD segmentation by default.
        """
        image = Image.open(self.images[index]).convert('RGB')
        
        targets: Any = []
        for target_type in self.target_type:
            if target_type == "semantic_ood":
                target = Image.open(self.targets_semantic_ood[index])
            elif target_type == "instance_ood":
                target = Image.open(self.targets_instance_ood[index])
            elif target_type == "depth_ood":
                target = Image.open(self.targets_depth_ood[index])
            elif target_type == "semantic":
                target = Image.open(self.targets_semantic[index])
            targets.append(target)
        
        target = tuple(targets) if len(targets) > 1 else targets[0]
        
        if self.transforms is not None:
            """transform applied to input and its targets"""
            image, target = self.transforms(image, target)
            
        return image, target

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)

    def __repr__(self):
        """Print some information about the dataset."""
        fmt_str = "DATASET: CARLA WildLife\n---- Dir %s\n" % self.root
        fmt_str += '---- Num found images: %d\n' % len(self.images)
        return fmt_str.strip()


class Cityscapes():
    """
    Cityscapes Dataset http://www.cityscapes-dataset.com/
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                        'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
        CityscapesClass('dog', 34, 224, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('jetski', 35, 225, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('crocodile', 36, 226, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('tarp', 37, 227, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('traffic barrier', 38, 228, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('folded cartons', 39, 229, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('wheel barrel', 40, 230, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('fox', 41, 231, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('gym bench', 42, 232, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('backpack', 43, 233, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('palette', 44, 234, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('pylon', 45, 235, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('exercise ball', 46, 236, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('concrete bags', 47, 237, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('crow', 48, 238, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('wolf', 49, 239, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('pig', 50, 240, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('canoe', 51, 241, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('box', 52, 242, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('stool', 53, 243, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('crutch', 54, 244, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('umbrella', 55, 245, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('ball', 56, 246, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('toy', 57, 247, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('caddy', 58, 248, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('trash can', 59, 249, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('gnome', 60, 250, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('trash bag', 61, 251, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('bottle', 62, 252, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('bucket', 63, 253, 'ood', 8, False, True, (255, 102, 0)),
        CityscapesClass('scooter', 64, 254, 'ood', 8, False, True, (255, 102, 0)),
    ]

    """Normalization parameters"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    name2train_id = {label.name: label.train_id for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))
    
cityscapes_class_names_to_train_id = Cityscapes().name2train_id
cityscapes_color_palette_train_ids = Cityscapes().color_palette_train_ids
