import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

class ImageDataset(data.Dataset):
    def __init__(self, dataset, target, transform):
        self.dataset = dataset
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        imgs, targets = self.dataset[index], self.target[index]

        # imgs = Image.fromarray(imgs)
        imgs = self.transform(imgs)

        return imgs, targets, index

    def __len__(self):
        return len(self.target)

def build_dataset(args):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    input_channel = train_dataset.data.shape[-1]
    num_classes = len(np.unique(train_dataset.targets))
    train_dataset = ImageDataset(train_dataset.data, train_dataset.targets, transform_train)
    test_dataset = ImageDataset(test_dataset.data, test_dataset.targets, transform_test)

    return train_dataset, test_dataset, input_channel, num_classes
