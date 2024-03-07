import os
import torch
import torchvision.transforms
from PIL import Image

os.chdir(os.path.join('/data', 'sft_lab', 'xjl', 'sync', 'icdar2024_sync_project'))
class GetDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

def data_folder():
    return os.path.join(os.getcwd(), 'data')

def train_data_folder():
    return os.path.join(data_folder(), 'train')

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

def label_line2nameandtoken(line):
    name, token = line.split(' ')
    return name.split('\\')[1], token.rstrip('\n')

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
    with open(label_txt()) as f:
        labels = [label_line2nameandtoken(line) for line in f]
    images = [image_name2tensor(name, trans, location) for name, _ in labels]
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

def image_blk(image_list, window_size):
    """Test codes:
img = '320.png'
img = Image.open(train_image_location(img))
trans = torchvision.transforms.ToTensor()
img_t = trans(img)
print(img_t.shape)
crops = image_blk([img_t], (3, 4))
print(len(crops))"""
    result_list = []
    is_tensor = True if type(image_list[0]) == torch.Tensor else False
    window_h, window_w = window_size
    for image in image_list:
        crops = []
        if not is_tensor:
            image = Image.open(image)
            w, h = image.size
        else:
            #transforms.ToTensor automatically turn Image.open:(w, h) into torch.Tensor:(c, h, w)
            c, h, w = image.shape

        num_window_h = h // window_h
        num_window_w = w // window_w
        x, y = [i for i in range(w)], [j for j in range(h)]
        if not is_tensor:
            for left in x[0: w: window_w]:
                for top in y[0: h: window_h]:
                    crops.append(image.crop((left, top, left + window_w, top + window_h)))
        else:
            windowed_image = torch.zeros((c, window_h, window_w))
            for left in range(num_window_w):
                for top in range(num_window_h):
                    for row in range(window_h):
                        for i in range(c):
                            windowed_image[i][row] = image[i][top*window_h+row][left*window_w:(left+1)*window_w]
                    crops.append(windowed_image)
        result_list.append(crops)
    return result_list


if __name__ != '__main__':
    __all__ = ['get_dataiter', 'images_labels', 'data_folder', 'train_data_folder', 'train_image_folder',
               'train_label_txt', 'test_data_folder', 'test_image_folder', 'test_label_txt']


