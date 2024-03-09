import os
import torch
import torchvision.transforms
from PIL import Image
import collections
from torch.nn import functional as F
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
    return name.split('\\')[0].split('/')[1], token.rstrip('\n')

def label_line2locationandtoken(line):
    location, token = line.split(' ')
    return location.split('/')[-1], token.rstrip('\n')

def image_location2tensor(location, trans):
    return trans(Image.open(location))

def images_labels(trans, is_train, is_location=False, num_images=20000):
    """input transforms' classes and is_train == True or False, return images and labels"""
    if is_train:
        image_folder = train_image_folder
        label_txt = train_label_txt
        location = train_image_location
    ### Wait for test dataset.
    # else:
    #     image_folder = test_image_folder
    #     label_folder = test_label_txt
    #     location = test_image_location
    labels = []
    images = []
    with open(label_txt()) as f:
        count = 0
        if not is_location:
            for line in f:
                labels.append(label_line2nameandtoken(line))
                count +=1
                if count == num_images:
                    break
        else:
            for line in f:
                labels.append(label_line2locationandtoken(line))
                count += 1
                if count == num_images:
                    break
    if not is_location:
        for name, _ in labels:
            images.append(image_name2tensor(name, trans, location))
    if is_location:
        for location, _ in labels:
            images.append(image_location2tensor(location, trans))
    labels, vocab = label_wordize([label[1] for label in labels])
    return images, F.one_hot(torch.tensor(labels), 1000), vocab


# def collate_fn(batch):
#     X, Y = [], []
#     for sample in batch:
#         X.append(sample[0]), Y.append(sample[1])
#     # print('X: ', len(X), X[0])
#     # print('Y: ', len(Y), Y[0])
#     # exit(0)
#     X, Y = torch.stack(X, dim=0), torch.tensor(Y)
#     return X, F.one_hot(Y, 1000).float()

def get_dataiter(batch_size, resize, is_train=True, is_location=False,
                 num_images=200, num_workers=0, pin_memory=True, **kwargs):
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
    else:
        img_trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor()
        ])
    fn = None
    images, labels, vocab = images_labels(img_trans, is_train, is_location=is_location, num_images=num_images)
    dataset = GetDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, collate_fn=fn,
                                       num_workers=num_workers, pin_memory=pin_memory, **kwargs)

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

class Vocab(object):
    def __init__(self, tokens, special_tokens):
        counter = collections.Counter(tokens)
        self.tokens_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.tokens_freqs.sort(key=lambda x: x[1], reverse=True)
        if special_tokens is False:
            self.unk = 0
            tokens = ['<unk>']
        if special_tokens is True:
            self.pad, self.bos, self.eos, self.unk = 0, 1, 2, 3
            tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        tokens += [token for token, freq in self.tokens_freqs]
        self.idx_to_token = []
        self.token_to_idx = dict()
        for token in tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        else:
            return [self.__getitem__(token) for token in tokens]

def label_tokenize(label_list, use_special_token=False):
    linkall = ''
    for label in label_list:
        linkall += label
    vocab_dict = Vocab(linkall, special_tokens=use_special_token)
    label_list = [[vocab_dict.token_to_idx[char] for char in label] for label in label_list]
    return label_list, vocab_dict

def label_wordize(label_list, use_special_token=False):
    vocab_dict = Vocab(label_list, special_tokens=use_special_token)
    label_list = [vocab_dict.token_to_idx[word] for word in label_list]
    return label_list, vocab_dict


def folder_list():
    return ['data_folder', 'train_data_folder', 'train_image_folder', 'train_label_txt',
            'test_data_folder', 'test_image_folder', 'test_label_txt']

if __name__ != '__main__':
    __all__ = ['get_dataiter', 'images_labels', 'folder_list', 'image_blk',
               'Vocab', 'label_tokenize']
