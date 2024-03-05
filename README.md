# icdar2024-hust-2-
icdar-2024
dataiter.py has 2 accessible functions 'get_dataiter' and 'images_labels'
get_dataiter(): Return a iteration of datasets. The resize=None has some problem, but resize=(H, W) is available.
images_labels(): Return (images, labels) according to train_set or test_set.
