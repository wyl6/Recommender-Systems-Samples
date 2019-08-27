from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# import progressbar

import os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not p in sys.path:
    sys.path.append(p)

import utils
from models import  FM, FNN

# In[1]: data setting
#ffm = pd.read_csv('./intern_data/train_ffm.csv')
#train_file,test_file = train_test_split(ffm,test_size = 0.2)
train_file = '../data/train_ffm.csv'
test_file = '../data/test_ffm.csv'
pkl_path = '../model/fm.pkl'
#test_final_file = './data/test_ffm_final.csv'
input_dim = utils.INPUT_DIM

train_data = utils.read_data(train_file)
train_data = utils.shuffle(train_data)
test_data = utils.read_data(test_file)


if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')
print('train data size:', train_data[0].shape)
print('test data size:', test_data[0].shape)

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(utils.FIELD_SIZES)


min_round = 1
num_round = 10
early_stop_round = 10
batch_size = 1024
iters = int((train_size+batch_size-1) / batch_size)

field_sizes = utils.FIELD_SIZES
field_offsets = utils.FIELD_OFFSETS

print("field_size", field_sizes)
print("field_offsets", field_offsets)
# In[2]: model configuration
# algo = 'fm'
algo = 'fnn'
if algo in {'fnn','anfm','amlp','ccpm','pnn1','pnn2'}:
    train_data = utils.split_data(train_data)
    test_data = utils.split_data(test_data)
    tmp = []
    for x in field_sizes:
        if x > 0:
            tmp.append(x)
    field_sizes = tmp
    print('remove empty fields', field_sizes)
    
if algo == 'fm':
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 128,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_w': 0,
        'l2_v': 0,
    }
    print(fm_params)
    model = FM(**fm_params)
elif algo == 'fnn':
    fnn_params = {
        'field_sizes': field_sizes,
        'embed_size': 129,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0, 0],
        'random_seed': 0,
        'init_path':pkl_path,
    }
    print(fnn_params)
    model = FNN(**fnn_params)

# In[3]: model training
def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        #if model.model_file is not None:
        #	model.saver.restore(model.sess,model.model_file)
        if batch_size > 0:
            ls = []
            print('[%d]\ttraining...' % i)
            for j in range(iters):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
#                if(j == 0):
#                    print('---------------------------')
#                    print(X_i[0])
#                    print('---------------------------')
                    
        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            print ('xi.shape', X_i.shape)
            print ('yi.shape', y_i.shape)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        #model.saver.save(model.sess,'./Save/train.ckpt')
        train_preds = []
        print('[%d]\tevaluating...' % i)
        for j in range(int(train_size / 10000 + 1)):
            X_i, _ = utils.slice(train_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            train_preds.extend(preds)
        test_preds = []
        for j in range(int(test_size / 10000 + 1)):
            X_i, _ = utils.slice(test_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            test_preds.extend(preds)
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break
#    var_map = {}
#    for name, var in model.vars.items():
#        print(name, var)
#        ans = model.run(var)
#        var_map[name] = ans
#        print(var_map[name])
    if algo == 'fm':
        model.dump(pkl_path)

print('iters', iters)
train(model)


# In[4]: model testing
#def test(model):
#	test_data_final = utils.read_new_data(test_final_file)
#	predict = pd.read_csv('./data/test1.csv')
#	res = predict[['aid','uid']]
#	model.saver.restore(model.sess,model.model_file)
#	test_new_preds = []
#	print ('testing new data...')
#	for i in range(len(test_data_final)):
#		preds = model.run(model.y_prob,test_data_final[i],mode='test')
#		test_new_preds.extend(preds)
#	np.save('./data/test_new_preds',np.array(test_new_preds))
#	res['score'] = test_new_preds
#	res.to_csv('./data/submission.csv', index=False)
#	print ('All finished!')
# test(model)

# In[5]:
#from tensorflow.python import pywrap_tensorflow


#reader = pywrap_tensorflow.NewCheckpointReader(init_path)
## reader = tf.train.NewCheckpointReader(checkpoint_path) # 用tf.train中的NewCheckpointReader方法
#var_to_shape_map = reader.get_variable_to_shape_map()
## 输出权重tensor名字和值
#for key in var_to_shape_map:
#    print("tensor_name: ", key,reader.get_tensor(key).shape)
    
    
    
    
    
    
'''
random initialize
[0]     training...
[0]     evaluating...
[0]     loss (with l2 norm):0.358097    train-auc: 0.610657     eval-auc: 0.661392
[1]     training...
[1]     evaluating...
[1]     loss (with l2 norm):0.350506    train-auc: 0.624879     eval-auc: 0.679986
[2]     training...
[2]     evaluating...
[2]     loss (with l2 norm):0.348581    train-auc: 0.631834     eval-auc: 0.688470
[3]     training...
[3]     evaluating...
[3]     loss (with l2 norm):0.347268    train-auc: 0.637031     eval-auc: 0.694607
[4]     training...
[4]     evaluating...
[4]     loss (with l2 norm):0.346279    train-auc: 0.641287     eval-auc: 0.699670
[5]     training...
[5]     evaluating...
[5]     loss (with l2 norm):0.345490    train-auc: 0.644798     eval-auc: 0.703892
[6]     training...
[6]     evaluating...
[6]     loss (with l2 norm):0.344828    train-auc: 0.647727     eval-auc: 0.707407
[7]     training...
[7]     evaluating...
[7]     loss (with l2 norm):0.344262    train-auc: 0.650155     eval-auc: 0.710297
[8]     training...
[8]     evaluating...
[8]     loss (with l2 norm):0.343769    train-auc: 0.652261     eval-auc: 0.712707
[9]     training...
[9]     evaluating...
[9]     loss (with l2 norm):0.343332    train-auc: 0.654116     eval-auc: 0.714787
'''

'''
fm initialize
[0]     training...
[0]     evaluating...
[0]     loss (with l2 norm):0.361066    train-auc: 0.607293     eval-auc: 0.642668
[1]     training...
[1]     evaluating...
[1]     loss (with l2 norm):0.353281    train-auc: 0.634517     eval-auc: 0.679833
[2]     training...
[2]     evaluating...
[2]     loss (with l2 norm):0.350498    train-auc: 0.640884     eval-auc: 0.688085
[3]     training...
[3]     evaluating...
[3]     loss (with l2 norm):0.347988    train-auc: 0.648423     eval-auc: 0.696806
[4]     training...
[4]     evaluating...
[4]     loss (with l2 norm):0.345739    train-auc: 0.657166     eval-auc: 0.706803
[5]     training...
[5]     evaluating...
[5]     loss (with l2 norm):0.343678    train-auc: 0.665929     eval-auc: 0.716429
[6]     training...
[6]     evaluating...
[6]     loss (with l2 norm):0.341738    train-auc: 0.674693     eval-auc: 0.725318
[7]     training...
[7]     evaluating...
[7]     loss (with l2 norm):0.339869    train-auc: 0.682893     eval-auc: 0.733139
[8]     training...
[8]     evaluating...
[8]     loss (with l2 norm):0.338055    train-auc: 0.690134     eval-auc: 0.739590
[9]     training...
[9]     evaluating...
[9]     loss (with l2 norm):0.336269    train-auc: 0.696557     eval-auc: 0.744801
'''
    
    
    
    
    


    
