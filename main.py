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
    datasets = ['imdb62', 'enron', 'imdb', 'blog', 'ccat50']
    parser = argparse.ArgumentParser(description=f'Training models for datasets {datasets}')
    parser.add_argument('--dataset', type=str, help='the dataset used for training')
    parser.add_argument('--id', type=int, help='the id of the experiment')
    parser.add_argument('--gpu', type=str, help='the cuda devices used for training', default="0,1,2,3")
    parser.add_argument('--tqdm', type=bool, help='whether tqdm is on', default=False)
    parser.add_argument('--samples-per-author', type=int, help='number of samples per author', default=None)
    parser.add_argument('--test', type=bool, help='whether test mode is on', default=False)

    args = parser.parse_args()
    if args.samples_per_author is not None:
        raise NotImplementedError("samples per author deprecated")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    list_scores = run_iterations(source=args.dataset, per_author=args.samples_per_author, id=args.id, tqdm=args.tqdm)
