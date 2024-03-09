import os
import dataProcess
from PIL import Image

StableImageNet_path = os.path.join('/data', 'sft_lab', 'xjl', 'dataset', 'imagenet1k')
save_folder = os.path.join(os.getcwd(), 'data', 'train')

def reset_image_folder(image_folder_path):
    for image in os.listdir(image_folder_path):
        file_path = os.path.join(image_folder_path, image)
        if os.path.isfile(file_path):
            os.remove(file_path)

def reset_label_txt(label_txt_path):
    with open(label_txt_path, "w") as file:
        file.write("")

def copy_images_labels(dataset_src_folder, dataset_save_folder):
    image_save_folder = os.path.join(dataset_save_folder, 'train_image')
    classes = os.listdir(dataset_src_folder)
    for clas in classes:
        clas_folder = os.path.join(dataset_src_folder, clas)
        imgs = os.listdir(clas_folder)
        clas_t = clas.split('_')[1].split(',')[0].replace(' ', '_')
        for img in imgs:
            img_src_path = os.path.join(clas_folder, img)
            img_cpy_path = os.path.join(image_save_folder, clas_t + '_' + img.split('_')[0] + '.jpg')
            img = Image.open(img_src_path)
            img.save(img_cpy_path)

def create_labels_txt(dataset_save_folder):
    txt_file = os.path.join(dataset_save_folder, 'train_labels.txt')
    image_save_folder = os.path.join(dataset_save_folder, 'train_image')
    images = os.listdir(image_save_folder)
    with open(txt_file, 'w') as f:
        for image in images:
            label = image.split('_')[0]
            f.write(os.path.join('train_image', image) + ' ' + label + '\n')

def load_StableImageNet():
    train_image_path = dataProcess.train_image_folder()
    train_txt_path = dataProcess.train_label_txt()
    reset_image_folder(train_image_path)
    reset_label_txt(train_txt_path)
    copy_images_labels(StableImageNet_path, save_folder)
    create_labels_txt(save_folder)

load_StableImageNet()
exit(0)