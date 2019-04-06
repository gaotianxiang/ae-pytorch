import torch
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils import data as data


class CIFAR10DL:
    def __init__(self, train_set=True, batch_size=32, num_workers=8):
        self.train_set = train_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dl = self.build_data_loader()

    def build_data_loader(self):
        if self.train_set:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()
        cifar10_dtst = CIFAR10(root='data', download=True, train=self.train_set, transform=transform)
        cifar10_dl = data.DataLoader(cifar10_dtst, batch_size=self.batch_size, shuffle=self.train_set,
                                     num_workers=self.num_workers)
        return cifar10_dl

    def __call__(self, *args, **kwargs):
        return self.dl


def dl_test(train):
    dl = CIFAR10DL(train_set=train, batch_size=32, num_workers=0)
    i = 0
    for x, _ in dl():
        grid = torchvision.utils.make_grid(x, nrow=8, padding=2, pad_value=255)
        torchvision.utils.save_image(grid, filename='./data/train_{}_{}.png'.format(train, i))
        i += 1
        if i == 5:
            break


if __name__ == '__main__':
    dl_test(True)
    dl_test(False)
