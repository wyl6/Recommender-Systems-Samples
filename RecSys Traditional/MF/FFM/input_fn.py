# -*- coding: utf-8 -*-

import tensorflow as tf

def create_train_input_fn(features,label, batch_size=32, num_epoches=10):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((features, label))
        dataset = dataset.shuffle(20).repeat(num_epoches).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    print('hello')
    return input_fn
