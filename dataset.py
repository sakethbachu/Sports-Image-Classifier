import os
import cv2
import torch
import albumentations as A
import pandas as pd
import csv
import torch.nn.functional as F


import config as CFG

#this dataloader will help us with pretraining task where we have image and caption pairs
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


#change this path
FILE_PATH = '/content/drive/MyDrive/sports_dataset/logical-rythm-2k20-sports-image-classification'

#Using pandas to read the csv files which have the ground truth labels
train_df = pd.read_csv(os.path.join(FILE_PATH,'train_labels.csv'))

#listing down the classes
classes = list(train_df['sports'])
classes = list(set(classes))

#these two lines help in going from textual classes to their unique numerical indexes and back
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

#this dataloader will help us in finetuning for the sports classification task
class CLIPDataset_classification(torch.utils.data.Dataset):
    def __init__(self,  csv_file, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        #the image filenames and the classes
        self.image_filenames = pd.read_csv(csv_file, error_bad_lines=False, sep=",", header=None, quoting=csv.QUOTE_NONE)
        self.image_filenames = self.image_filenames.iloc[1:,:]

        #captions here are classes of the sports 
        captions = self.image_filenames[1]
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        ext = self.image_filenames.iloc[idx, 0]

        ext = ext[8:]
        image = cv2.imread(f"{CFG.image_path}/{ext}")
        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        # print(self.captions[idx])

        #converting the tensor to a one hot vector as it is required while performing the classification task
        item['caption'] = F.one_hot(torch.tensor(class_to_idx[self.captions[idx]]), num_classes=22)


        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )