import torch
import torch.utils.data as data
from PIL import Image
import os
from glob import glob
from torchvision.datasets import ImageFolder
from torchvision import transforms

class Dataset(data.Dataset):
    def __init__(self, class_mapping = None, root_folder = None, transformer = None, phase = None) -> None:

        self.transformer = transformer
        self.phase = phase

        self.labels = []
        self.classes = set()
        self.class_name = {}
        self.class_mapping = {}
        # print(root_folder)
        if not class_mapping is None:
            self.class_mapping = class_mapping
            for clas, idx in class_mapping.items():
                self.class_name[idx] = clas
        else:
            for id, class_folder in enumerate(glob(root_folder+"/*")):
                if os.path.isdir(class_folder):
                        class_name = class_folder.split('/')[-1]
                        # print(class_name)
                        self.class_mapping[class_name] = id
                        self.class_name[id] = class_name 

        #create image_paths
        self.images = []

        for class_name in self.class_mapping.keys():
             class_path = os.path.join(root_folder, class_name)
             for img_path in glob(class_path + "/*"):
                  self.images.append(img_path)
                  self.labels.append(self.class_mapping[class_name])

    def __len__(self):
        return len(self.images)
    
    def num_classes(self):
        return len(self.class_mapping)

    def __getitem__(self, index):

        #get image, label_idx
        image = Image.open(self.images[index]).convert('RGB')
        target = self.labels[index]

        #transform image
        if self.transformer is not None: 
            image = self.transformer(image)
        return image, target
