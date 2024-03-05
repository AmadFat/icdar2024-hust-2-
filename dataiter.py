import os
import torch
import random
import torchvision.transforms
from PIL import Image

class GetDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

def train_data_folder():
    return os.path.join(os.getcwd(), 'data', 'train')

def train_image_folder():
    return os.path.join(train_data_folder(), 'train_image')

def train_label_txt():
    return os.path.join(train_data_folder(), 'train_labels.txt')

def test_data_folder():
    return os.path.join(os.getcwd(), 'data', 'test')

def test_image_folder():
    return os.path.join(train_data_folder(), 'test_image')

def test_label_txt():
    return os.path.join(train_data_folder(), 'test_labels.txt')

def train_image_location(image_name):
    return os.path.join(train_image_folder(), image_name)

def test_image_location(image_name):
    return os.path.join(test_image_folder(), image_name)

def image_name2tensor(image, trans, location):
    return trans(Image.open(location(image)))

def label_line2token(line):
    return line.split( )[1].rstrip('\n')

def images_labels(trans, is_train):
    """input transforms' classes and is_train == True or False, return images and labels"""
    if is_train is True:
        image_folder = train_image_folder
        label_txt = train_label_txt
        location = train_image_location
    ### Wait for test dataset.
    # else:
    #     image_folder = test_image_folder
    #     label_folder = test_label_txt
    #     location = test_image_location
    images = [image_name2tensor(image, trans, location) for image in os.listdir(image_folder())]
    labels = [label_line2token(line) for line in open(label_txt())]
    return images, labels

def get_dataiter(batch_size, resize, is_train, num_workers=12, **kwargs):
    """batch_size is the mini-batch size. resize == tuple or None is the resized shape on images.
    is_train == True or False means it's a train_iter or not. num_workers has default num of 12.
    Test codes:
    data = get_dataiter(batch_size=32, resize=(120, 100), is_train=True)
    for x, y in data:
        print(x[0].shape, len(y), y[0]) ##The output img is of (Channel=3, Height=120, Width=100)
        break
------------------------------------------------------------------------------------------------------
    data = get_dataiter(batch_size=32, resize=None, is_train=True)
    for x, y in data:
        print(x[0].shape, len(y), y[0]) ##The output img is of original size
        break
    """
    if resize is None:
        img_trans = torchvision.transforms.ToTensor()
        # collate_fn = lambda batch: [data for data in batch]
        collate_fn = lambda batch: (batch[:][0], batch[:][1])
    else:
        img_trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])
        collate_fn = None
    images, labels = images_labels(img_trans, is_train)
    dataset = GetDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
                                       collate_fn=collate_fn, **kwargs)


if __name__ != '__main__':
    __all__ = ['get_dataiter', 'images_labels']
