# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import itertools
from pandas import DataFrame

# Visualization
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
import plotly.express as px

# Feature extraction approach
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Classification
import xgboost as xgb
import lightgbm as lgbm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# BERT classifier
# Installing a custom version of Simple Transformers
# !git clone https://github.com/NVIDIA/apex
# !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
#!git init
# !pip install --upgrade tqdm
# !git remote add origin https://github.com/ThilinaRajapakse/simpletransformers.git
# !git pull origin master
# !pip install -r requirements-dev.txt
# !pip install transformers
# !pip install tensorboardX

# !pip install simpletransformers
from simpletransformers.classification import ClassificationModel

import torch

# # Parallelize apply on Pandas
# !pip install pandarallel
from pandarallel import pandarallel
pandarallel.initialize()

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    import argparse
    from utils import run_iterations

    datasets = ['imdb62', 'enron', 'imdb', 'blog']
    parser = argparse.ArgumentParser(description=f'Training models for datasets {datasets}')
    parser.add_argument('--dataset', type=str, help='the dataset used for training')
    args = parser.parse_args()
    list_scores = run_iterations(source=args.dataset)
