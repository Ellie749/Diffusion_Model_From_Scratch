import torch
from torchvision.transforms import Compose, ToTensor, Lambda, Normalize
from torchvision.datasets import MNIST, CIFAR10



def load_data():
    preprocess = Compose([
        ToTensor(),
        Lambda(lambda x: x * 2. - 1.)  # [-1, 1]
    ])

    train_dataset = MNIST(root='./data', train=True, download=True, transform=preprocess)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=preprocess)

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)



    # transform = Compose([
    #             ToTensor(), 
    #             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # for RGB channels
    #             ])

    # # Download CIFAR-10 training set
    # train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    # train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # # Optional: test set
    # test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    # test_data = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    return train_data, test_data

