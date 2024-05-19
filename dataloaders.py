import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from typing import Tuple
from PIL import Image
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.labels = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.all_images = []
        self.all_labels = []
        self.prepare_data()

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        image_path = self.all_images[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_idx = self.all_labels[index]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image, label_tensor
    
    def prepare_data(self):
        for idx,row in self.labels.iterrows():
            label = row[0]
            label_dir = os.path.join(self.img_dir, label)
            images=[]
            dir_file_list=os.listdir(label_dir)
            for el in dir_file_list:
                el_path=os.path.join(label_dir,el)
                if os.path.isdir(el_path):
                    img_list=os.listdir(el_path)
                    for image in img_list:
                        images.append(os.path.join(el_path,image))
                else:
                    images.append(el_path)
            if len(images)<=100:
                continue
            
            images=images[:100] #take only the first 100 images
            #print(idx,label, len(images))
            self.all_images.extend(images)
            self.all_labels.extend([idx] * len(images))  # Extend labels repeated for each image

def load_dataset(batchsize:int,numworkers:int,use_distributed_sampler: bool = False) ->Tuple[DataLoader,DistributedSampler] :
    img_dir = 'bone-marrow-cell-classification/bone_marrow_cell_dataset/'
    csv_file='bone-marrow-cell-classification/abbreviations.csv'
    
    df = pd.read_csv(csv_file, sep=';', header=None)

    # List of codes to remove
    codes_to_remove = ['ABE', 'FGC', 'KSC', 'OTH', 'LYI']

    # Identify the rows to remove and store their original indices
    rows_to_remove = df[df[0].isin(codes_to_remove)]
    original_indices = rows_to_remove.index.tolist()

    # Remove the specified rows
    df_filtered = df[~df[0].isin(codes_to_remove)]

    # Reinitialize the indices of the remaining rows
    df_filtered.reset_index(drop=True, inplace=True)

    # Output the original indices of the removed rows
    #print("Original indices of removed rows:", original_indices)
     # Define transformations
    trans = transforms.Compose([
        transforms.Resize((256, 256)),   # Resize the images to a fixed size
        transforms.ToTensor(),           # Convert images to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
    ])
    
    dataset = CustomDataset(dataframe=df_filtered, img_dir=img_dir, transform=trans)
    # Create distributed sampler
    if use_distributed_sampler:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        num_workers=numworkers,
        sampler=sampler,
        drop_last=True
    )
    dataset_centers=np.load('centers.npy')
    dataset_labels=np.load('labels.npy')

   # Remove the rows from dataset_centers corresponding to original_indices
    dataset_centers_filtered = np.delete(dataset_centers, original_indices, axis=0)
    dataset_centers_filtered=torch.from_numpy(dataset_centers_filtered)
    # Optionally, remove the corresponding labels
    dataset_labels_filtered = np.delete(dataset_labels, original_indices, axis=0)

    # Output the filtered arrays
    # print("Filtered dataset_centers shape:", dataset_centers_filtered.shape)
    # print("Filtered dataset_labels:", dataset_labels_filtered)
    
    return dataloader, sampler,dataset_centers_filtered
