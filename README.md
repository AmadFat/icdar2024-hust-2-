# icdar2024-hust-2-

dataProcess.py from dataiter.py has some accessible functions.

    get_dataiter(): Return a iteration of datasets. The resize=None problem has fixed.

    images_labels(): Return (images, labels) according to train_set or test_set.

    image_blk(): Input photo_location/tensor list and a window_size. Output is the croping result.

    label_tokenize(): Input label_list. Output transformed label_list and corresponding vocabulary_diction.



make_smallset.py has 1 accessible function 'make_smallset'

    make_smallset(): No input. Automatically extract at most 128 photo-label pairs both from 'train' and 'test' in 'data'. The number 128 can be changed in python file.

gradImprove.py has 'image_gradient_improve' function

    image_gradient_improve(): Input a tensor. Return the gradient of square.
