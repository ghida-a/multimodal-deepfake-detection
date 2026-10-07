import cv2
import os
import numpy as np

def load_images(folder_path, label, img_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def load_dataset(real_path, fake_path):
    real_imgs, real_labels = load_images(real_path, 0)
    fake_imgs, fake_labels = load_images(fake_path, 1)
    X = np.concatenate((real_imgs, fake_imgs), axis=0)
    y = np.concatenate((real_labels, fake_labels), axis=0)
    return X, y
