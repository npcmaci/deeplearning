import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def get_mnist_loaders(
    data_dir='./data/MNIST/MNIST',
    batch_size=64,
    test_batch_size=1000,
    seed=1,
    use_cuda=True,
    shuffle=True,
    num_workers=1,
    pin_memory=True
):
    """
    返回 MNIST 的训练和测试数据加载器，支持自定义参数。
    手动加载本地MNIST数据（无下载）。
    """

    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if device.type == "cuda":
        cuda_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'shuffle': shuffle
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update({'num_workers': num_workers, 'pin_memory': pin_memory, 'shuffle': False})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 设置 download=False，避免重新下载
    dataset1 = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)

    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader, device
