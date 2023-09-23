from model import UNET
from dataset import Satellite
from torch.utils.data import DataLoader
import torch
from utils import *
import os


def train(checkpoint):
    if checkpoint == None:
        model = UNET(in_c=3, out_c=5)
        start_epoch = 0
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']

    model = model.to(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f" -- Initiating the Training Process -- ")
    print(f"Epoch: {start_epoch}: ")

    for epoch in range(start_epoch, epochs):
        average_loss = 0
        for i, (image, mask) in enumerate(test_gen):
            image = image.to(device)
            mask = mask.to(device)
            pred_mask = model(image)
            loss = criterion(pred_mask, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            average_loss = average_loss + loss
            del image, mask, pred_mask

            if i%100 == 0:
                print("=", end="")

        validation_loss = 0
        for j, (image, mask) in enumerate(test_gen):
            image = image.to(device)
            mask = mask.to(device)
            pred_mask = model(image)
            loss = criterion(pred_mask, mask)

            validation_loss = validation_loss + loss
            del image, mask, pred_mask
        save_checkpoint(epoch=epoch, model=model, version=version)
        add_result(f"Epoch: {epoch} | Average Loss: {average_loss/(i + 1)} | Val Loss: {validation_loss/(j + 1)}")
        print(f"   Epoch: {epoch} | Average Loss: {average_loss/(i + 1)} | Val Loss: {validation_loss/(j + 1)}")


if __name__ == "__main__":
    version = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if f"satellite_v{version}.pth.tar" in os.listdir():
        checkpoint = f"satellite_v{version}.pth.tar"
    else:
        checkpoint = None
    batch_size = 2
    iterations = 10000
    workers = 4
    epochs = 1000
    lr = 1e-5
    train_file = "train.json"
    test_file = "test.json"

    train_dataset = Satellite(train_file)
    test_dataset = Satellite(test_file, type='test')

    train_gen = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )

    test_gen = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
    )
    train(checkpoint)

