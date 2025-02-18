import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from src.models import lstm, tcn
from src.utils import auxiliary_plots, metrics
from src.utils.print_functions import notify_slack
from src.preprocessing import normalization, data_generation
import itertools
import pandas as pd
device='cuda'
import time
from utils import linf_attack, l0_attack, l2_attack

EPSILIONS = [10,50,100,500,1000]
ALPHAS = [0.0001,0.0003,0.001,0.003,0.01, 0.03,.1,.3,1,3][::-1]
ITERATIONS = [50,100,200,500]

CONSTS = [5, 5.25,5.5,5.75,6]
L2_LRs = [.3,.01,.03,.001]
L2_ITERATIONS = [100,300,500,1000]



# load model
m = tf.keras.models.load_model('../files/models/best-model-bs-100-ph-288-seed-9-12-31_22-39-15.ckpt/')
# load data

norm_params = {'mean': 28505.41, 'std': 4596.946, 'max': 41217.0, 'min': 17714.0}
batch_size=128
past_history = 288
forecast_horizon = 24

test_file_name='data/hourly_20140102_20191101_test.csv'

# Read test data file
with open(test_file_name, 'r') as datafile:
    ts_test = datafile.readlines()[1:]  # skip the header
    ts_test = np.asarray([np.asarray(l.rstrip().split(',')[0], dtype=np.float32) for l in ts_test])
    ts_test = np.reshape(ts_test, (ts_test.shape[0],))

# Normalize test data with train params
ts_test = normalization.normalize(ts_test, norm_params)

x_test, y_test = data_generation.univariate_data(ts_test, 0, ts_test.shape[0], past_history, forecast_horizon)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

x = next(iter(test_data))[0]
y = next(iter(test_data))[1]



print('l0 attack')
for c in CONSTS:
    for l2_lr in L2_LRs:
        for l2_iter in L2_ITERATIONS:
            pert, duration = l0_attack(x,y,m,l2_iter=l2_iter, l2_c=c,l2_lr=l2_lr)

            
print('linf attack')
# linf attack, different epsilons
for eps in EPSILIONS:
    for alpha in ALPHAS:
        for iteration in ITERATIONS:
            pert = linf_attack(m,tf.convert_to_tensor(x), tf.convert_to_tensor(y), epsilon = eps, alpha=alpha, iterations=iteration)
            adv_ex = x + pert
            clean_preds = m.predict(x)
            adv_preds = m.predict(adv_ex)


print('l2 attack')
for c in CONSTS:
    for l2_lr in L2_LRs:
        for l2_iter in L2_ITERATIONS:
            pert, duration = l2_attack(x,y,m, iter=l2_iter, const=c,lr=l2_lr)

