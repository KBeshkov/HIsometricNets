import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json

class InatCustom(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load labels from json file
        with open(os.path.join(root_dir, 'annotations.json'), 'r') as f:
            self.labels_info = json.load(f)
            
        #define hierarhical properties 
        self.species = set(category["species"] for category in self.labels_info["root"]["categories"])
        self.kingdoms = set(category["kingdom"] for category in self.labels_info["root"]["categories"])
        self.phylum = set(category["phylum"] for category in self.labels_info["root"]["categories"])
        self.class_ = set(category["class"] for category in self.labels_info["root"]["categories"])
        self.order = set(category["order"] for category in self.labels_info["root"]["categories"])
        self.family = set(category["family"] for category in self.labels_info["root"]["categories"])
        self.genus = set(category["genus"] for category in self.labels_info["root"]["categories"])
        
        
        self.targets = []
        self.species_labels = {species: [] for species in self.species}
        self.kingdom_labels = {kingdom: [] for kingdom in self.kingdoms}
        self.phylum_labels = {phylum: [] for phylum in self.phylum}
        self.class_labels = {class_: [] for class_ in self.class_}
        self.order_labels = {order: [] for order in self.order}
        self.family_labels = {family: [] for family in self.family}
        self.genus_labels = {genus: [] for genus in self.genus}

        # Iterate through each subfolder
        for folder_name in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, folder_name)):
                folder_path = os.path.join(root_dir, folder_name)
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    img = self.transform(Image.open(img_path).convert('RGB'))
                    self.data.append(img.T)
                    label = self.labels_info["root"]["categories"][int(folder_name)]
                    self.targets.append(label['class_id'])
                    self.species_labels[label["species"]].append(label["class_id"])
                    self.kingdom_labels[label["kingdom"]].append(label["class_id"])
                    self.phylum_labels[label["phylum"]].append(label["class_id"])
                    self.class_labels[label["class"]].append(label["class_id"])
                    self.order_labels[label["order"]].append(label["class_id"])
                    self.family_labels[label["family"]].append(label["class_id"])
                    self.genus_labels[label["genus"]].append(label["class_id"])
                    self.labels.append(label)
        self.data = torch.stack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        return image, label

# Example usage
root_dir = '/Users/kosio/Repos/HIsometricNets/Data/inat-2021/semi-inat-2021-mini/train/'
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

dataset = InatCustom(root_dir=root_dir, transform=transform)

