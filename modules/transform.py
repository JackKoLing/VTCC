import torchvision
from PIL import ImageFilter, ImageOps
import random

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    def __call__(self, x):
        return ImageOps.solarize(x)


class Augmentation:
    def __init__(self, size):
        self.train_transform_1 = [
            torchvision.transforms.RandomResizedCrop(size=size, scale=(0.08, 1.)),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]

        self.train_transform_2 = [
            torchvision.transforms.RandomResizedCrop(size=size, scale=(0.08, 1.)),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            torchvision.transforms.RandomApply([Solarize()], p=0.2),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
        
        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
        
        self.train_transform_1 = torchvision.transforms.Compose(self.train_transform_1)
        self.train_transform_2 = torchvision.transforms.Compose(self.train_transform_2)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform_1(x), self.train_transform_2(x)
