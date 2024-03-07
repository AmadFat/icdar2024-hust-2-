import dataiter
import torchvision
import torch
trans = torchvision.transforms.ToTensor()
images, labels = dataiter.images_labels(trans=trans, is_train=True)

def array_gradient_improve(arr, alpha=1.3):
    result = torch.zeros(arr.shape)
    H, W = result.shape
    result[0][:] = torch.pow(torch.abs(arr[0][:] - arr[1][:]), alpha)
    result[1:][0] = torch.pow(torch.abs(arr[1:][0] - arr[1:][1]), alpha)
    for i in range(1, H):
        result[i][:] += torch.pow(torch.abs(arr[i][:] - arr[i-1][:]), alpha)
    result, arr = result.T, arr.T
    for i in range(1, W):
        result[i][:] += torch.pow(torch.abs(arr[i][:] - arr[i-1][:]), alpha)
    result, arr = result.T, arr.T
    maxvalue, minvalue = torch.max(result), torch.min(result)
    return (result - minvalue) / (maxvalue - minvalue)

def image_gradient_improve(X):
    """Test codes:
        image = images[30]
        image_t = image_gradient_improve(image)
        plt.imshow(image_t.permute(1 ,2, 0))
        plt.show()"""
    X = [array_gradient_improve(x) for x in X]
    return torch.stack(X, dim=0)


if __name__ != '__main__':
    __all__ = ['image_gradient_improve']