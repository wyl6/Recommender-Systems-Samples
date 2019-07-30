# -*- coding: utf-8 -*-

import tensorflow as tf
from util import FieldHandler, transformation_data
from model_fn import create_model_fn
from input_fn import create_train_input_fn
from hparams import create_hparams, FLAGS
from model.ffm import FFM

# In[1]: FFM call
def main(_):
    fh = FieldHandler(train_file_path=FLAGS.train_file_path,
                      category_columns=FLAGS.category_columns,
                      continuation_columns=FLAGS.continuation_columns)
    
    features, labels = transformation_data(file_path=FLAGS.train_file_path,
                                           field_hander=fh,
                                           label=FLAGS.label)
    
    hparams = create_hparams(fh.field_nums, fh.feature_nums)
    
    train_input_fn = create_train_input_fn(features,
                                           label=labels,
                                           batch_size=hparams.batch_size,
                                           num_epoches=hparams.epoches)
    model_fn = create_model_fn(FFM)
    
    estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=FLAGS.model_path,
            params=hparams,
            config=tf.estimator.RunConfig(
                    tf_random_seed=hparams.seed,
                    log_step_count_steps=500))
    
    show_dict = {
            'loss':'loss',
            'accuracy':'accuracy/value',
            'auc':'auc/value'}
    log_hook = tf.train.LoggingTensorHook(show_dict, every_n_iter=100)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[log_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn, steps=None)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
# In[2]: 
if(__name__ == '__main__'):
    tf.app.run()