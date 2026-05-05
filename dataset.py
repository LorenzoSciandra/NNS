import torch
from torch.utils.data import Subset
import random 
from utils import *
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cub2011 import Cub2011
from datasets import load_dataset
from torch.utils.data import Dataset

DATASET_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "fashion_mnist": 10,
    "svhn": 10,
    "stl10": 10,
    "imagenet100": 100,
    "cub": 200
}

DATASET_FEATURES = {
    "cifar10": 32*32*3,
    "cifar100": 32*32*3,
    "mnist": 32*32,
    "fashion_mnist": 32*32,
    "svhn": 32*32*3,
    "stl10": 96*96*3,
    "imagenet100": 224*224*3,
    "cub": 224*224*3
}

DATASET_IMG_SIZE = {
    "cifar10": 32,
    "cifar100": 32,
    "mnist": 32,
    "fashion_mnist": 32,
    "svhn": 32,
    "stl10": 96,
    "imagenet100": 224,
    "cub": 224
}

DATASET_NUM_CHANNELS = {
    "cifar10": 3,
    "cifar100": 3,
    "mnist": 1,
    "fashion_mnist": 1,
    "svhn": 3,
    "stl10": 3,
    "imagenet100": 3,
    "cub": 3
}

DATASET_NUM_EXAMPLES = {
    "cifar10": 50000,
    "cifar100": 50000,
    "mnist": 60000,
    "fashion_mnist": 60000,
    "svhn": 73257,
    "stl10": 5000,
    "imagenet100": 126689,
    "cub": 5994
}

DATASET_FLATTEN_FEATURES = {
    "cifar10": 128 * 8 * 8,
    "cifar100": 128 * 8 * 8,
    "mnist": 128 * 7 * 7,
    "fashion_mnist": 128 * 7 * 7,
    "svhn": 128 * 8 * 8,
    "stl10": 128 * 24 * 24,
    "imagenet100": 128 * 56 * 56,
    "cub": 128 * 56 * 56
}

class HFDatasetWrapper(Dataset):
    """Simple wrapper to make HF datasets behave like standard PyTorch datasets"""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert("RGB") # Ensure 3 channels
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

class ImageDataset:
    def __init__(self, data_args):
        self.data_args = data_args
        self.name = data_args["name"]
        self.frac_train = data_args["frac_train"]
        self.frac_valid = data_args["frac_valid"]
        self.frac_tunnel = data_args["frac_tunnel"]
        
        assert self.frac_train + self.frac_valid <= 1.0, "Fractions must sum to 1 or less"
        
        # self.tunnel_set_size = data_args["tunnel_set_size"]
        self.num_classes = DATASET_CLASSES[self.name]
        self.features = DATASET_FEATURES[self.name]
        self.train_set, self.valid_set, self.tunnel_set, self.test_set = self.get_dataset()
        # self.train_targets = [self.train_set[i][1] for i in range(len(self.train_set))]
        # self.valid_targets = [self.valid_set[i][1] for i in range(len(self.valid_set))]
        # self.tunnel_targets = [self.tunnel_set[i][1] for i in range(len(self.tunnel_set))]
        # self.test_targets = [self.test_set[i][1] for i in range(len(self.test_set))]

    
    def __split_dataset(self, dataset, targets):
        
        total_size = len(dataset)
        valid_size = int(len(dataset) * self.frac_valid)
        train_size = total_size - valid_size
        tunnel_size = int(train_size * self.frac_tunnel)
        
        valid_indices, tunnel_indices, train_indices = [], [], []
        
        indices = np.arange(total_size)
        random.shuffle(indices)
        if valid_size > 0:  
            train_indices = indices[:train_size]
            valid_indices = indices[train_size:]
            self.valid_targets = targets[valid_indices]
            self.train_targets = targets[train_indices]
        else:
            train_indices = indices
            self.valid_targets = []
        
        if tunnel_size > 0:
            
            train_targets = targets[train_indices]
            tunnel_indices = stratified_sample(train_targets, tunnel_size)
            tunnel_indices = train_indices[tunnel_indices]
            tunnel_targets = targets[tunnel_indices]
            self.train_targets = train_targets
            self.tunnel_targets = tunnel_targets
        
        train_set = Subset(dataset, train_indices)
        validation_set = Subset(dataset, valid_indices)
        tunnel_set = Subset(dataset, tunnel_indices)

        return train_set, validation_set, tunnel_set
    

    def get_dataset(self):
        if self.name not in ['cifar10', 'cifar100', 'stl10', 'svhn', 'fashion_mnist', 'mnist', 'cub', 'imagenet100']:
            raise AttributeError("Dataset not available")
        
        if self.name=='cifar10':
            transform_train = transforms.Compose(
                                    [
                                    #transforms.RandomCrop(32, padding=4),
                                    #transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                    )
            
            transform_test = transforms.Compose(
                                    [
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                    )

            train_set = torchvision.datasets.CIFAR10('./data/cifar10/', train = True, transform = transform_train, download = True)
            

            test_set = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False,
                                                download=True, transform=transform_test)
        
        elif self.name=='cifar100':
            transform_train = transforms.Compose([
            # Uncomment these if you want data augmentation:
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            ])

            train_set = torchvision.datasets.CIFAR100('./data/cifar100/', train=True,
                                                    transform=transform_train, download=True)

            test_set = torchvision.datasets.CIFAR100(root='./data/cifar100/', train=False,
                                                    download=True, transform=transform_test)

        elif self.name=='stl10':
            train_transform = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomCrop(96, padding=4),
                #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239))
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239))
            ])

            train_set = torchvision.datasets.STL10('./data/stl10', split='train', transform=train_transform, download=True)
            test_set = torchvision.datasets.STL10('./data/stl10', split='test', transform=test_transform, download=True)


        elif self.name=='svhn':
            train_transform = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                #transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])

            train_set = torchvision.datasets.SVHN('./data/svhn', split='train', transform=train_transform, download=True)
            test_set = torchvision.datasets.SVHN('./data/svhn', split='test', transform=test_transform, download=True)


        elif self.name=='fashion_mnist':
            train_transform = transforms.Compose([
                transforms.Resize((32,32)),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
            
            test_transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])

            train_set = torchvision.datasets.FashionMNIST('./data/fashionmnist', train=True, transform=train_transform, download=True)
            test_set = torchvision.datasets.FashionMNIST('./data/fashionmnist', train=False, transform=test_transform, download=True)


        elif self.name=='mnist':
            train_transform = transforms.Compose([
                transforms.Resize((32,32)),
                # transforms.RandomRotation(10),
                # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_set = torchvision.datasets.MNIST('./data/mnist', train=True, transform=train_transform, download=True)
            test_set = torchvision.datasets.MNIST('./data/mnist', train=False, transform=test_transform, download=True)

        elif self.name=='cub':
            transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_set = Cub2011('./data/cub2011', train=True, transform=transform_train, download=True)
            test_set = Cub2011('./data/cub2011', train=False, transform = transform_test, download=True)
        
        elif self.name == 'imagenet100':
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            raw_ds = load_dataset("clane9/imagenet-100", cache_dir="./data/imagenet100'")
            
            train_data = raw_ds['train']
            test_data = raw_ds['validation']

            train_set = HFDatasetWrapper(train_data, transform=transform_train)
            test_set = HFDatasetWrapper(test_data, transform=transform_test)

        if self.name == 'imagenet100':
            train_targets = train_data['label']
            self.test_targets = test_data['label']
        else:
            # train_targets = [train_set[i][1] for i in range(len(train_set))]
            train_targets = np.array(train_set.targets)
            self.test_targets = np.array(test_set.targets)

        train_set, validation_set, tunnel_set = self.__split_dataset(train_set, train_targets)
        
        return train_set, validation_set, tunnel_set, test_set

    def __make_loader(self, dataset, targets, batch_size, stratified_batch, num_workers=1, shuffle=True):
        
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        
        if stratified_batch:
            return DataLoader(dataset, 
                              batch_sampler=StratifiedBatchSampler(targets, batch_size, shuffle=shuffle), 
                              num_workers=num_workers)
        else:
            return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        
    def return_loaders(self, batch_size, num_workers, stratified_sampler):
        train_loader = self.__make_loader(self.train_set, self.train_targets, batch_size, stratified_sampler, num_workers, shuffle=True)
        # valid_loader = self.__make_loader(self.valid_set, self.valid_targets, batch_size, stratified_sampler, num_workers, shuffle=True)
        valid_loader = None
        tunnel_loader = self.__make_loader(self.tunnel_set, self.tunnel_targets, batch_size, stratified_sampler, num_workers, shuffle=False)
        test_loader = self.__make_loader(self.test_set, self.test_targets, batch_size, stratified_sampler, num_workers, shuffle=False)
        return train_loader, valid_loader, tunnel_loader, test_loader