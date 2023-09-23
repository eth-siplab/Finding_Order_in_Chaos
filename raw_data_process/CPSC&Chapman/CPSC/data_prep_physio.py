#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from scipy import signal
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle

# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)

# Choose leads from the recording file.
def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording

def train_12ECG_classifier(input_directory, output_directory):
    # Load data.
    print('Loading data...')
    fs = 500
    leads = ['I','II','III','V2']
    lead_index = [0,1,2,7]
    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)

    classes = get_classes(input_directory, header_files)
    num_classes = len(classes)
    num_files = len(header_files)
    recordings = list()
    headers = list()

    for i in range(num_files):
        recording, header = load_challenge_data(header_files[i])
        #recording = choose_leads(recording, header, leads)
        recordings.append(recording[lead_index,:])
        headers.append(header)

    # Train model.
    print('Training model...')

    features = list()
    labels = list()

    for i in range(num_files):
        recording = recordings[i]
        header = headers[i]

        features.append(recording)

        for l in header:
            if l.startswith('# Dx:'):
                labels_act = np.zeros(num_classes)
                arrs = l.strip().split(' ')
                for arr in arrs[2].split(','):
                    class_index = classes.index(arr.rstrip()) # Only use first positive index
                    labels_act[class_index] = 1
        labels.append(labels_act)
    chunk_size = 5000  # Take 10 seconds
    whole_data,whole_label = [], []
    features, features_test = train_test_split(features, test_size=0.20, random_state=40,stratify=labels)
    for idx, sample in enumerate(features):
        num_chunks = sample.shape[1] // chunk_size
        chunks = [sample[:, i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        label = labels[idx]
        while chunks:
            array = chunks.pop()
            if np.isnan(array).any():
                continue
            else:
                whole_data.append(array)
                whole_label.append(np.argmax(label))
    for i in range(len(whole_data)):
        x = signal.resample(whole_data[i], 1000, axis=1).T  # Resample
        if np.all(np.isnan(x.std(axis=0))) or np.all(x.std(axis=0) == 0):
            continue
        whole_data[i] = (x - x.mean(axis=0)) / x.std(axis=0)
    features = np.array(whole_data)
    labels1 = np.array(whole_label)
    cpsc_data = [features, labels1]
    with open('cpsc_data_train.pkl', 'wb') as f:
        pickle.dump(cpsc_data, f)

    whole_data, whole_label = [], []
    for idx, sample in enumerate(features_test):
        num_chunks = sample.shape[1] // chunk_size
        chunks = [sample[:, i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        label = labels[idx]
        while chunks:
            array = chunks.pop()
            if np.isnan(array).any():
                continue
            else:
                whole_data.append(array)
                whole_label.append(np.argmax(label))
    for i in range(len(whole_data)):
        x = signal.resample(whole_data[i], 1000, axis=1).T  # Resample
        if np.all(np.isnan(x.std(axis=0))) or np.all(x.std(axis=0) == 0):
            continue
        whole_data[i] = (x - x.mean(axis=0)) / x.std(axis=0)
    features = np.array(whole_data)
    labels = np.array(whole_label)
    cpsc_data = [features, labels]
    with open('cpsc_data_test.pkl', 'wb') as f:
        pickle.dump(cpsc_data, f)

# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('# Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)


if __name__ == '__main__':
    train_12ECG_classifier('PhysioNetChallenge2020_Training_CPSC','output')