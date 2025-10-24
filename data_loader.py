import numpy as np, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Cutout(object):
    def __init__(self, no_of_holes=1, length=16):
        self.no_of_holes = no_of_holes
        self.length = length
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.no_of_holes):
            y = np.random.randint(h); x = np.random.randint(w)
            y1, y2 = np.clip([y - self.length//2, y + self.length//2], 0, h)
            x1, x2 = np.clip([x - self.length//2, x + self.length//2], 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

def get_loaders(batch_size=128, augment=True, val_ratio=0.1, num_workers=2, data_root='./data'):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2023, 0.1994, 0.2010)
    train_tfms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ] if augment else [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if augment: train_tfms.append(Cutout(no_of_holes=1, length=16))
    test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    full_train = datasets.CIFAR10(data_root, train=True, download=True,
                                  transform=transforms.Compose(train_tfms))
    test_set   = datasets.CIFAR10(data_root, train=False, download=True,
                                  transform=test_tfms)

    val_size   = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_set, val_set
