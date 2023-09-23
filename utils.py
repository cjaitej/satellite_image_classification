import os
import json
import random
from torchvision import transforms
import torch

def create_dataset(filename):
    dataset = []

    image_path = filename + "/images/"
    mask_path = filename + "/masks/"

    for i, j in zip(os.listdir(image_path), os.listdir(mask_path)):
        dataset.append([image_path + i, mask_path + i])
    random.shuffle(dataset)
    x = int(len(dataset)*0.8//1)
    with open('train.json', "w", encoding='utf-8') as f:
        json.dump(dataset[:x], f)

    with open('test.json', "w", encoding='utf-8') as f:
        json.dump(dataset[x:], f)


    print(f"Length of train dataset = {x}")
    print(f"Length of test dataset = {len(dataset) - x}")


def train_transform(image, mask):
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    vflip = transforms.RandomVerticalFlip(p=0.5)
    totensor = transforms.PILToTensor()


    if random.random() > 0.5:
        image = hflip(image)
        mask = hflip(mask)

    #Vertical Flipping
    if random.random() > 0.5:
        image = vflip(image)
        mask = vflip(mask)

    #Converting Image to torch tensor.
    image = totensor(image)
    mask = totensor(mask)

    return image/255., mask

def test_transform(image, mask):
    totensor = transforms.PILToTensor()

    image = totensor(image)
    mask = totensor(mask)
    return image/255., mask


def add_result(result):
    with open('results_v1.txt', 'a') as f:
        f.write(result + "\n")
    f.close()

def save_checkpoint(epoch, model, version):
    state = {'epoch': epoch,
             'model': model}
    filename = f"satellite_v{version}.pth.tar"
    torch.save(state, filename)