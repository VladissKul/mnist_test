import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from classify_model import MNIST_Classify_Model
from classify_train import train_model


def main():
    with open("../config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    device = config["app"]["device"]
    EPOCH = config["model"]["epoch"]
    BATCH_SIZE = config["model"]["batch_size"]
    LEARNING_RATE = config["model"]["learning_rate"]

    train_datasets = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)

    test_datasets = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=True)

    model = MNIST_Classify_Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_dataloader, test_dataloader, optimizer, criterion, device, num_epochs=EPOCH)

    print("Finished Training\n[INFO] Saving Model...")
    torch.save(model.state_dict(), 'model.pth')
    print("Finished Saving Model\nExiting...")


if __name__ == "__main__":
    main()
