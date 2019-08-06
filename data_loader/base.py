from abc import ABCMeta, abstractmethod
from pathlib import Path

import tensorflow as tf
from utils.factory import Factory


class DatasetFactory(Factory):
    _registered_ = {}


class DatasetMeta(ABCMeta):

    def __init__(cls, name, bases, methods):
        super(DatasetMeta, cls).__init__(name, bases, methods)
        DatasetFactory.register(cls, suffix='Dataset')


class _BaseDataset(object, metaclass=DatasetMeta):

    def __init__(self,
                 batch_size=2,
                 repeat_n=1,
                 num_workers=2,
                 shuffle=True,
                 img_fmt='.jpg',
                 img_channel_n=3,
                 fast_mode=True,
                 color_ordering=0):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.repeat_n = repeat_n
        self.num_workers = num_workers
        self.img_channel_n = img_channel_n
        self.fast_mode = fast_mode
        self.color_ordering = color_ordering
        if img_fmt == '.png':
            self.img_decoder = tf.image.decode_png
        else:
            self.img_decoder = tf.image.decode_jpeg

    @property
    def image_path(self):
        raise NotImplementedError('Method image_path not implemented')

    @property
    def anno(self):
        raise NotImplementedError('Method _annos not implemented')

    @property
    def shuffle_size(self):
        raise NotImplementedError('Method shuffle_size not implemented')

    @property
    def output_shape(self):
        raise NotImplementedError('Method output_shape not implemented')

    def __len__(self):
        return NotImplemented

    @property
    def num_batch(self):
        raise NotImplementedError('Method num_batch not implemented')

    def _preprocess(self, ds, is_training):
        raise NotImplementedError('_preprocess not implemented.')

    def get(self, is_training=True):
        with tf.device('/cpu:0'):
            ds = tf.data.Dataset.from_tensor_slices(
                (self.image_path, self.anno))
            if self.shuffle:
                ds = ds.shuffle(buffer_size=self.shuffle_size)
            ds = ds.repeat(self.repeat_n)
            ds = self._preprocess(ds, is_training)
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(None)
            return ds

    def resize(self, image):
        return tf.image.resize_images(
            image,
            self.output_shape,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def random_distort_color(self, image):
        if self.color_ordering < 0:
            return image

        if self.fast_mode:
            if self.color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=0.1)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            else:
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
                image = tf.image.random_brightness(image, max_delta=0.1)
        else:
            if self.color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=0.1)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
                image = tf.image.random_hue(image, max_delta=0.1)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            elif self.color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
                image = tf.image.random_brightness(image, max_delta=0.1)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
                image = tf.image.random_hue(image, max_delta=0.1)
            elif self.color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
                image = tf.image.random_hue(image, max_delta=0.1)
                image = tf.image.random_brightness(image, max_delta=0.1)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            else:
                image = tf.image.random_hue(image, max_delta=0.1)
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
                image = tf.image.random_brightness(image, max_delta=0.1)
        return image
