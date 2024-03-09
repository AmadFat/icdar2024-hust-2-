# icdar2024-hust-2-


We will face a very LARGE PROJECT CONSTRUCTION ALTERNATION. Thank you for your patience.



dataProcess.py from dataiter.py has some accessible functions. This script has moved to /data.

    get_dataiter(): Return a iteration of datasets. The resize=None problem has fixed.

    images_labels(): Return (images, labels) according to train_set or test_set.

    image_blk(): Input photo_location/tensor list and a window_size. Output is the croping result.

    label_tokenize(): Input label_list. Output transformed label_list and corresponding vocabulary_diction.



make_smallset.py has 1 accessible function 'make_smallset'. This script has moved to /utils.

    make_smallset(): No input. Automatically extract at most 128 photo-label pairs both from 'train' and 'test' in 'data'. The number 128 can be changed in python file.
    

gradImprove.py has 'image_gradient_improve' function. This script has moved to /data.

    image_gradient_improve(): Input a tensor. Return the gradient of square.


/parallelTrain is used to help all of us use torchrun or distributed.DistributedDataParallel.

    This folder has 2 example scripts: DDP1.py and DDP2.py.
    
