from torch.utils.data import Dataset
import json
from PIL import Image
import torch
from utils import train_transform, test_transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np



class Satellite(Dataset):
    def __init__(self, filename, type='train', size=(256, 256)):
        super(Satellite, self).__init__()
        f = open(filename)
        self.data = json.load(f)

        if type == "train":
            self.transform = train_transform
        else:
            self.transform = test_transform
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, mask = self.data[index]
        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        temp = torch.zeros((5, self.size[1], self.size[0]))
        for i in range(temp.shape[0]):
            temp[i] = torch.as_tensor((mask == i), dtype= torch.int8)

        return image.to(torch.float32), temp.to(torch.float32)


# for i, j in Satellite("test.json", type="test"):
#     print(i, j)