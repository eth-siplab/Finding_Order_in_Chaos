import numpy as np, os, sys, joblib
import scipy.io
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from scipy import signal
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
####################################################### Combining two sets for exps ############################
chapman_data = scipy.io.loadmat('chapman.mat')
chapman_data = chapman_data['whole_data']
file_train = open('cpsc_data_train.pkl', 'rb')
file_test = open('cpsc_data_test.pkl', 'rb')
cpsc_data_train = pickle.load(file_train)
cpsc_data_test = pickle.load(file_test)
################################
chapman_labels = chapman_data[:,1]
chapman_data = chapman_data[:,0]
chapman_y = np.zeros((len(chapman_labels),))
chapman_x = np.zeros((len(chapman_labels),1000,4))
counter = 0
for idx, k in enumerate(chapman_data):
    if np.isnan(k).any():
        continue
    chapman_y[counter] = chapman_labels[idx][0][0]
    chapman_x[counter] = k
    counter += 1
# Uncomment to check if there is any NaN values in the data
# last_index = np.argmax(chapman_y==0)
# chapman_x, chapman_y = chapman_x[0:last_index,:,:], chapman_y[0:last_index]
chapman_train_x, chapman_test_x, chapman_train_y, chapman_test_y = train_test_split(chapman_x, chapman_y, test_size=0.20, random_state=40, stratify=chapman_y)
cpsc_train_x, cpsc_train_y = cpsc_data_train[0], cpsc_data_train[1]
Nans = np.unique(np.where(np.isnan(cpsc_train_x)==True)[0])
cpsc_train_x = np.delete(cpsc_train_x, Nans, axis=0)
cpsc_train_y = np.delete(cpsc_train_y, Nans, axis=0)
##################
cpsc_test_x, cpsc_test_y = cpsc_data_test[0], cpsc_data_test[1]
Nans = np.unique(np.where(np.isnan(cpsc_test_x)==True)[0])
cpsc_test_x = np.delete(cpsc_test_x, Nans, axis=0)
cpsc_test_y = np.delete(cpsc_test_y, Nans, axis=0)
###################
whole_dataset = []
whole_dataset.append([cpsc_train_x, cpsc_train_y])
whole_dataset.append([cpsc_test_x, cpsc_test_y])
whole_dataset.append([chapman_train_x, chapman_train_y-1])
whole_dataset.append([chapman_test_x, chapman_test_y-1])

dataecg = dict(whole_dataset=whole_dataset)
with open('ECG_data.pkl', 'wb') as handle:
    pickle.dump(dataecg, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('exit')
