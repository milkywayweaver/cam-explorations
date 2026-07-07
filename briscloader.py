import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
import os
import glob
from tqdm.auto import tqdm

class LoadBRISC():
    def __init__(self):
        self.cls_root = './brisc2025/classification_task'
        self.seg_root = './brisc2025/segmentation_task'
    
    def load(self,classes:str='all',planes:str='all',label_types:str='class',train_transform=None,test_transform=None):
        '''
        Loads BRISC 2025 dataset into a PyTorch Dataset.

        Args:
            classes (str, default='all'): Classes to pick.
                Supported values are:
                    - 'all' : Picks all classes
                    - 'tumor' : Picks all classes except 'no_tumor' 
            planes (str: default='all') : Planes to pick
                Supported values are:
                    - 'all' : Picks all planes
                    - 'ax' : Axial plane
                    - 'sa' : Sagital plane
                    - 'co' : Coronal plane
            label_types (str, default='class'): Label return type
                Supported values are:
                    - 'class' : Returns tumor type as labels
                    - 'plane' : Returns plane type as labels
                    - 'both' : Returns tumor and plane types as labels
            train_transform (torchvision.transforms): Image transformation for the training dataset
            test_transform (torchvision.transforms): Image transformation for the testing dataset

        Returns:
            PyTorch Dataset object of selected configuration.
        '''
        if classes not in ['all','tumor']:
            raise NotImplementedError('Unknown value for "classes" arguement! Use "all" or "tumor".')
        if planes not in ['all','ax','sa','co']:
            raise NotImplementedError('Unknown value for "classes" arguement! Use "all", "ax", "sa", or "co".')
        if label_types not in ['class','plane','both']:
            raise NotImplementedError('Unknown value for "classes" arguement! Use "class", "plane", or "both".')

        self.classes_dict = {
            'all':['gl','me','no','pi'],
            'tumor':['gl','me','pi']
        }
        self.planes_dict = {
            'all':['ax','sa','co'],
            'ax':['ax'],
            'sa':['sa'],
            'co':['co']
        }
        self.classes = self.classes_dict[classes]
        self.planes = self.planes_dict[planes]

        # Get image paths and labels
        train_img_paths = []
        train_mask_paths = []
        train_labels = []
        test_img_paths = []
        test_mask_paths = []
        test_labels = []
        for cls in self.classes:
            for pln in self.planes:
                labels = f'{cls}' if label_types == 'class' else f'{pln}' if label_types == 'plane' else f'{cls} {pln}'
                if cls == 'no':
                    train_paths = glob.glob(os.path.join(self.cls_root,'train','no_tumor',f'*{pln}*'))
                    test_paths = glob.glob(os.path.join(self.cls_root,'test','no_tumor',f'*{pln}*'))

                    train_img_paths.extend(train_paths)
                    test_img_paths.extend(test_paths)

                    train_mask_paths.extend([None for i in range(len(train_paths))])
                    test_mask_paths.extend([None for i in range(len(test_paths))])
                else:
                    train_paths = glob.glob(os.path.join(self.seg_root,'train','images',f'*{cls}_{pln}*'))
                    test_paths = glob.glob(os.path.join(self.seg_root,'test','images',f'*{cls}_{pln}*'))

                    train_img_paths.extend(train_paths)
                    test_img_paths.extend(test_paths)

                    train_mask_paths.extend(glob.glob(os.path.join(self.seg_root,'train','masks',f'*{cls}_{pln}*')))
                    test_mask_paths.extend(glob.glob(os.path.join(self.seg_root,'test','masks',f'*{cls}_{pln}*')))
                train_labels.extend([labels for i in range(len(train_paths))])
                test_labels.extend([labels for i in range(len(test_paths))])
            
        # Get images and masks
        train_imgs,train_masks = self.__import_image(train_img_paths,train_mask_paths)
        test_imgs,test_masks = self.__import_image(test_img_paths,test_mask_paths)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        # Put images and masks into a Dataset object
        self.train_ds = DatasetClass(train_imgs,train_masks,train_labels,transform=train_transform)
        self.test_ds = DatasetClass(test_imgs,test_masks,test_labels,transform=test_transform)
        return self.train_ds,self.test_ds

    def __import_image(self,img_paths,mask_paths):
        '''
        Imports images and masks from given list of paths
        Args:
            img_paths (list): List of path to images
            mask_paths (list) : List of path to masks
        
        Returns:
            PyTorch tensor of the images and masks in one batch
        '''
        imgs,masks = [],[]
        for i in tqdm(range(len(img_paths))):
            img = read_image(img_paths[i])
            img = v2.functional.resize(img,(224,224))
            if mask_paths[i]:
                mask = read_image(mask_paths[i])
                mask = v2.functional.resize(mask,(224,224))
            else:
                mask = torch.zeros_like(img)
 
            imgs.append(img)
            masks.append(mask)
        imgs = torch.cat(imgs)
        masks = torch.cat(masks)
        return imgs,masks

class DatasetClass(Dataset):
    def __init__(self,imgs,masks,labels,transform=None):
        self.imgs = imgs
        self.masks = masks
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img,mask,label