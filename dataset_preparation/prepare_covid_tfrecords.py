import os
import sys
import tqdm
import random
import numpy as np
sys.path.append(os.getcwd())
import tensorflow as tf
import imageio
from defaults import get_cfg_defaults
from utils import get_main_directory
import cv2


def compile(cfg, logger):
    # initial test using part of the data I have. all of covid and half of normal ct scans
    # thinking of what is the best way to do so
    # I'll generate train and test files by iterating through the original files
    data_path = os.path.join(get_main_directory(), "data/")
    dataset_name = "COVID-19_Radiography_Dataset"
    data_list = ["COVID", "Normal"]
    count = []
    if not os.path.exists(data_path):
        print("------Downloading Dataset from Kaggle------")
        os.system(
            f"kaggle datasets download tawsifurrahman/covid19-radiography-database -p {data_path} --unzip")
    if not os.path.exists(os.path.dirname(cfg.DATASET.PATH)):
        os.makedirs(os.path.dirname(cfg.DATASET.PATH))

    for r in range(2, cfg.DATASET.MAX_RESOLUTION_LEVEL+1):
        img_size = 2**r
        for data_dir in data_list:
            root, dirs, files = next(os.walk(os.path.join(data_path, dataset_name, data_dir), topdown=False))
            count.append(len(files))
            random.shuffle(files)
            files = files[:min(count)]
            images = []
            for file in tqdm.tqdm(files):
                image = imageio.imread(root + "/" + file)
                image = cv2.resize(image, (img_size,img_size))
                image = np.reshape(image, [1, img_size, img_size])
                # put images in dict
                images.append(image)
            split = 0.85
            train, test = images[:int(len(images)*split)], images[int(len(images)*split):]

            part_path = cfg.DATASET.PATH % (r, 0)
            tfr_writer = tf.io.TFRecordWriter(part_path)
            for image in train:
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))
                }))
                tfr_writer.write(ex.SerializeToString())
            tfr_writer.close()

            part_path = cfg.DATASET.PATH_TEST % (r, 0)
            tfr_writer = tf.io.TFRecordWriter(part_path)
            for image in test:
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))
                }))
                tfr_writer.write(ex.SerializeToString())
            tfr_writer.close()


im_size = 256
cfg = get_cfg_defaults()
compile(cfg, None)
