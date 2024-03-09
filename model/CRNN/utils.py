import os
import random

if __name__ == '__main__':
    total_images_path_list = os.listdir("../data/images")

    for image_file_name in total_images_path_list:
        num = random.random()
        if num < 0.1:
            with open('../data/labels/test.txt','a') as f:
                f.write(f'{image_file_name}\n')
        else:
            with open('../data/labels/train.txt','a') as f:
                f.write(f'{image_file_name}\n')