# general
import os,sys
import glob
import numpy as np
import pandas as pd
# from utils import equalise_dicts, limit_dicts, UAR, printProgressBar, writeLineToCSV, softmax
import pickle
import random
import json
import shutil
import soundfile as sf
import matplotlib.pyplot as pl # plots for inter-annot agreement
import time # for timing the training
from datetime import timedelta # for writing the timing the training to a file
# import multiprocessing # for multi-processing
from threading import Thread
import speech_recognition # for adding automatic transcriptions
import torchaudio
# statistics
from scipy import stats as st # to calculate pearson correlation and mode for decision level fusion
from sklearn.metrics import r2_score
from scipy.interpolate import UnivariateSpline

# specific (can be used in both features and models below)
import speechbrain as sb
import torch
# from torch.multiprocessing import Pool, Process, set_start_method # for multi-processing

# features
from speechbrain.lobes.features import Fbank
from sklearn.feature_extraction.text import TfidfVectorizer

# models
from sklearn import svm
from sklearn.linear_model import LinearRegression # for decision fusion
from sklearn.neural_network import MLPClassifier # for decision fusion
from sklearn.svm import SVR, SVC # for decision fusion

# models
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import ast 
from tqdm import tqdm


'''

def get_mean_std_mfb(train_dict):
    feat_extractor = Feature_extracter(os.path.join(data_path, "Features"),
                                       "mfb",
                                       **{"n_mels": 80},
                                       feat_func = lambda x: x.squeeze(0),
                                      )
    
    keys = list(train_dict.keys())
    all_feats = []
    for i, ID in enumerate(keys):
        print_progress_bar(i + 1, len(keys), prefix = f'Calculating MFBs:', suffix = 'completed', length=50)
        feats = feat_extractor(ID=[ID], wav_path=[train_dict[ID]["wav_path"]])
        feats = np.squeeze(feats, axis=0)
        all_feats += feats.tolist()
    all_feats = np.array(all_feats)
    # print(all_feats.shape)
    mean = all_feats.mean(0)
    std  = all_feats.std(0)
    return mean, std

csv_mean_path = "mfb_mean.csv"
csv_std_path  = "mfb_std.csv"
if not os.path.exists(csv_mean_path):
    mean, std = get_mean_std_mfb(train_dict)
    df_m = pd.DataFrame(mean)
    df_m.to_csv(csv_mean_path, index=False)
    df_s = pd.DataFrame(std)
    df_s.to_csv(csv_std_path, index=False)
       
print('saved')
sys.exit()
'''