# -*- coding: utf-8 -*-

import os
import inspect
import tensorflow as tf

def class_vars(obj):
    return {k: v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}
    
class BaseModel(object):
    
    def __init__(self, config):
        self.built = False
        self.saver = None
        
        for attr in class_vars(config):
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(config, attr))
        
    def save_model(self, model_dir):
        '''
        save model into model_dir
        '''
        if(not self.built):
            raise RunTimeError('The model must be trained before training')
        
        self.saver = tf.train.Saver()
        model_name = type(self).__name__
        if(not os.path.exists(model_dir)):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, model_name)
        self.saver.save(self.sess, model_path)

    def load_model(self, model_dir):
        
        tensor_names = ['placeholder/users:0', 'placeholder/items:0',
                        'placeholder/ratings:0', 'prediction/pred:0']
        operation_names = ['optimizer/optimizer']

        model_name = type(self).__name__
        model_path = os.path.join(model_dir, model_name)
        
        self.saver = tf.train.import_meta_graph(model_path+'.meta')
        self.saver.restore(self.sess, model_path)
        
        for name in tensor_names:
            attr = '_'+name.split('/')[1].split(':')[0]
            setattr(self, attr, tf.get_default_graph().get_tensor_by_name(name))
            
        for name in operation_names:
            attr = '_'+name.split('/')[1].split(':')[0]
            setattr(self, attr, tf.get_default_graph().get_operation_by_name(name))
            
        self.built = True