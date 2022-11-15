from tensorflow.python.keras.preprocessing.image import Iterator
from keras_applications import imagenet_utils
from utils.utils import *
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from imageio import imread
import cv2
import os

keras_utils = tf.keras.utils


class DataIterator(Iterator):
    def __init__(self,
                 image_data_generator,
                 images_list,
                 num_classes,
                 batch_size,
                 target_size,
                 shuffle=True,
                 seed=None,
                 data_aug_rate=0.):
        num_images = len(images_list)

        self.image_data_generator = image_data_generator
        self.images_list = images_list
        self.num_classes = num_classes
        self.target_size = target_size
        self.data_aug_rate = data_aug_rate

        super(DataIterator, self).__init__(num_images, batch_size, shuffle, seed)

    @classmethod
    def load_mask(self,img_path,thick=5):
        anno_path = ""
        if "train" in img_path:
            anno_path = img_path.replace("train","train_anno").replace(".png","_PLINE.json")
        elif "test" in img_path:
            anno_path = img_path.replace("test","test_anno").replace(".png","_PLINE.json")

        with open(anno_path, 'r', encoding='utf-8') as f:
            contents = f.read()
            json_data = json.loads(contents)

        # array 형태로 이미지 로드
        load_img = np.array(Image.open(img_path))
        # 검정색 색공간 생성.
        lbl = np.zeros((load_img.shape[0], load_img.shape[1]),
                       np.int32)  # 어차피 나중에 true mask를 float형태로 변환해주니 이 부분은 굳이 안바꿔도 될듯하다.

        # 차례대로 polylines 불러옴.
        for idx in range(len(json_data["annotations"])):

            temp = np.array(json_data["annotations"][idx]["polyline"]).reshape(-1)
            try:
                temp_round = np.apply_along_axis(np.round, arr=temp, axis=0)
                temp_int = np.apply_along_axis(np.int32, arr=temp_round, axis=0)
            except:
                t = json_data["annotations"][idx]["polyline"]
                none_json = [[x for x in t[0] if x is not None]]
                temp = np.array(none_json).reshape(-1)
                temp_round = np.apply_along_axis(np.round, arr=temp, axis=0)
                temp_int = np.apply_along_axis(np.int32, arr=temp_round, axis=0)

            temp_re = temp_int.reshape(-1, 2)
            lbl = cv2.polylines(img=lbl,
                                pts=[temp_re],
                                isClosed=False,
                                color=(1),
                                thickness=thick)
        return lbl

    # 어디서 호출되는 거지....?, Iterator 클래스 안에 있는 함수이다. 여기서 Override 하는듯.
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(shape=(len(index_array),) + self.target_size + (3,))
        batch_y = np.zeros(shape=(len(index_array),) + self.target_size + (self.num_classes,))

        for i, idx in enumerate(index_array):
            # image load
            image = load_image(self.images_list[idx])

            label = self.load_mask(self.images_list[idx])

            # random crop
            if self.image_data_generator.random_crop:
                image, label = random_crop(image, label, self.target_size)
            else:
                image, label = resize_image(image, label, self.target_size)

            # data augmentation
            if np.random.uniform(0., 1.) < self.data_aug_rate:
                # random vertical flip
                if np.random.randint(2):
                    image, label = random_vertical_flip(image, label, self.image_data_generator.vertical_flip)
                # random horizontal flip
                if np.random.randint(2):
                    image, label = random_horizontal_flip(image, label, self.image_data_generator.horizontal_flip)
                # random brightness
                if np.random.randint(2):
                    image, label = random_brightness(image, label, self.image_data_generator.brightness_range) # 항상 augmentation 되게 만듦

                # random rotation
                if np.random.randint(2):
                    image, label = random_rotation(image, label, self.image_data_generator.rotation_range)

                # random channel shift
                if np.random.randint(2):
                    image, label = random_channel_shift(image, label, self.image_data_generator.channel_shift_range)
                # random zoom
                if np.random.randint(2):
                    image, label = random_zoom(image, label, self.image_data_generator.zoom_range)

            # standardization
            image = imagenet_utils.preprocess_input(image.astype('float32'), data_format='channels_last',
                                                    mode='torch')

            # label = tf.squeeze(label,axis=-1)  # 차원 하나 감소
            label = tf.one_hot(label,self.num_classes,axis=-1) # 기존의 one-hot encoding이 잘못된것은 아닐까....

            batch_x[i], batch_y[i] = image, label

        return batch_x, batch_y


class ImageDataGenerator(object):
    def __init__(self,
                 random_crop=False,
                 rotation_range=0,
                 brightness_range=None,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 horizontal_flip=False,
                 vertical_flip=False):
        self.random_crop = random_crop
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def flow(self,
             images_list,
             num_classes,
             batch_size,
             target_size,
             shuffle=True,
             seed=None,
             data_aug_rate=0.):
        return DataIterator(image_data_generator=self,
                            images_list=images_list,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            target_size=target_size,
                            shuffle=shuffle,
                            seed=seed,
                            data_aug_rate=data_aug_rate)
