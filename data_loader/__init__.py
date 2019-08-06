from collections import namedtuple

from utils.utils import get_available_gpus

from .base import DatasetFactory


InputReturns = namedtuple('InputReturns', [
    'train_iter', 'test_iter', 'handle_iterator', 'handle', 'steps_per_epoch'
])


def input_fn(data_reader_config):
    import tensorflow as tf

    name = data_reader_config['name']
    ds = DatasetFactory.create(name)

    train_ds = ds(data_reader_config,
                  data_reader_config['train_file'],
                  batch_size=data_reader_config.train_batch_size,
                  repeat_n=data_reader_config.epochs,
                  num_workers=data_reader_config.preprocess_workers,
                  fast_mode=data_reader_config.fast_mode,
                  color_ordering=data_reader_config.color_ordering,
                  img_fmt=data_reader_config.img_fmt)

    test_ds = ds(data_reader_config,
                 data_reader_config['test_file'],
                 batch_size=data_reader_config.test_batch_size,
                 repeat_n=-1,
                 num_workers=data_reader_config.preprocess_workers,
                 fast_mode=data_reader_config.fast_mode,
                 color_ordering=data_reader_config.color_ordering,
                 img_fmt=data_reader_config.img_fmt)

    # data_size = len(train_ds)
    steps_per_epoch = len(train_ds) // data_reader_config.train_batch_size
    steps_per_epoch /= len(get_available_gpus())
    train_ds = train_ds.get(is_training=True)
    test_ds = test_ds.get(is_training=False)

    train_iter = train_ds.make_one_shot_iterator()
    test_iter = test_ds.make_one_shot_iterator()

    # handle
    handle = tf.placeholder(tf.string, shape=[])
    handle_iterator = tf.data.Iterator.from_string_handle(
        handle, train_ds.output_types, train_ds.output_shapes)
    return InputReturns(train_iter=train_iter,
                        test_iter=test_iter,
                        handle_iterator=handle_iterator,
                        handle=handle,
                        steps_per_epoch=steps_per_epoch)
    # return train_iter, test_iter, handle_iterator, handle
