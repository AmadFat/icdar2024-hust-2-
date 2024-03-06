import os
import dataiter
import shutil

def small_folder():
    return os.path.join(dataiter.data_folder(), 'small_set')

def small_train_folder():
    return os.path.join(small_folder(), 'train')

def small_train_image_folder():
    return os.path.join(small_train_folder(), 'train_image')

def small_train_label_txt():
    return os.path.join(small_train_folder(), 'train_labels.txt')

def small_test_folder():
    return os.path.join(small_folder(), 'test')

def small_test_image_folder():
    return os.path.join(small_test_folder(), 'test_image')

def small_test_label_txt():
    return os.path.join(small_test_folder(), 'test_labels.txt')

def make_small_train():
    os.makedirs(small_train_image_folder())

    count = 0
    image_list = os.listdir(dataiter.train_image_folder())
    for image in image_list:
        source_path = os.path.join(dataiter.train_image_folder(), image)
        target_path = os.path.join(small_train_image_folder(), image)
        shutil.copy(source_path, target_path)
        count += 1
        if count >= 128:
            break

    small_labels = ''
    count = 0
    with open(dataiter.train_label_txt()) as labels:
        for line in labels:
            small_labels += line
            count += 1
            if count >= 128:
                break
        with open(small_train_label_txt(), "w+") as sf:
            sf.write(small_labels)

def make_small_test():
    os.makedirs(small_test_image_folder())

    count = 0
    image_list = os.listdir(dataiter.test_image_folder())
    for image in image_list:
        source_path = os.path.join(dataiter.test_image_folder(), image)
        target_path = os.path.join(small_test_image_folder(), image)
        shutil.copy(source_path, target_path)
        count += 1
        if count >= 128:
            break

    small_labels = ''
    count = 0
    with open(dataiter.test_label_txt()) as labels:
        for line in labels:
            small_labels += line
            count += 1
            if count >= 128:
                break
        with open(small_test_label_txt(), "w+") as sf:
            sf.write(small_labels)

def make_smallset():
    """This function is used to make a small train/test set for experiments."""
    # assert small_folder() not in os.listdir(dataiter.data_folder()), 'Have gotten "small_set" folder in data folder!'
    os.makedirs(small_folder(), exist_ok=True)
    check_list = os.listdir(small_folder())


    if 'train' in check_list:
        print('"train" in "small_set" have found. The construction request of "train" folder is ignored.')
    else:
        print('"train" in "small_set" not found. The construction request of "train" folder will be processing.')
        make_small_train()


    if 'test' in check_list:
        print('"test" in "small_set" have found. The construction request of "test" folder is ignored.')
    else:
        print('"test" in "small_set" not found. The construction request of "test" folder will be processing.')
        make_small_test()
