import dataProcess
import torchvision
import torch
from matplotlib import pyplot as plt
trans = torchvision.transforms.ToTensor()
images, labels = dataProcess.images_labels(trans=trans, is_train=True)

def array_gradient_improve(arr, alpha=2):
    result = torch.zeros(arr.shape)
    H, W = result.shape
    result[0][:] = torch.pow(torch.abs(arr[0][:] - arr[1][:]), alpha)
    result[1:][0] = torch.pow(torch.abs(arr[1:][0] - arr[1:][1]), alpha)
    for i in range(1, H):
        result[i][:] += torch.pow(torch.abs(arr[i][:] - arr[i-1][:]), alpha)
    result, arr = result.T, arr.T
    for i in range(1, W):
        result[i][:] += torch.pow(torch.abs(arr[i][:] - arr[i-1][:]), alpha)
    result= result.T
    return  result

def array_quality_improve(arr, sharp=4, thresh=0.6):
    arr = torch.pow(arr, 1./sharp)
    maxvalue, minvalue = torch.max(arr), torch.min(arr)
    arr = (arr - minvalue) / (maxvalue - minvalue)
    # arr = torch.tensor([[1. if x > thresh else 0. for x in y] for y in arr])
    return arr

def image_gradient_improve(X):
    """Test codes:
        image = images[30]
        image_t = image_gradient_improve(image)
        plt.imshow(image_t.permute(1 ,2, 0))
        plt.show()"""
    X = [array_gradient_improve(x) for x in X]
    X = [array_gradient_improve(x) for x in X]
    X = [array_quality_improve(x) for x in X]
    return torch.stack(X, dim=0)


if __name__ != '__main__':
    __all__ = ['image_gradient_improve']

image = images[20]
image_t = image_gradient_improve(image)
plt.imshow(image_t.permute(1 ,2, 0))
plt.show()
