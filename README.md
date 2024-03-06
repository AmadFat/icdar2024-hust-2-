# icdar2024-hust-2-

dataiter.py has 2 accessible functions 'get_dataiter' and 'images_labels'

    get_dataiter(): Return a iteration of datasets. The resize=None problem has fixed.

    images_labels(): Return (images, labels) according to train_set or test_set.



make_smallset.py has 1 accessible function 'make_smallset'

    make_smallset(): No input. Automatically extract at most 128 photo-label pairs both from 'train' and 'test' in 'data'. The number 128 can be changed in python file.
