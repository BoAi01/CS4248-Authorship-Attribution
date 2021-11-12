import argparse
from train import run_iterations
import os
import torch
import numpy
import random

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    datasets = ['imdb62', 'enron', 'imdb', 'blog']
    parser = argparse.ArgumentParser(description=f'Training models for datasets {datasets}')
    parser.add_argument('--dataset', type=str, help='the dataset used for training')
    parser.add_argument('--gpu', type=str, help='the cuda devices used for training', default="0,1,2,3")
    parser.add_argument('--author', type=str, help='the number of authors')
    parser.add_argument('--ensemble', type=str, help='the type of ensemble')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    list_scores = run_iterations(source=args.dataset, num_authors=int(args.author),
        ensemble_type=args.ensemble)
