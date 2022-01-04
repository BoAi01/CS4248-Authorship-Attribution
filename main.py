import argparse
import os
import torch
import numpy
import random
import warnings
import logging
from learning_methods import train_ensemble
from utils import load_dataset_dataframe, build_train_test
from models import BertClassiferHyperparams


random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)


if __name__ == '__main__':
    datasets = ['imdb62', 'enron', 'imdb', 'blog', 'ccat50']
    parser = argparse.ArgumentParser(description=f'Training models for datasets {datasets}')
    parser.add_argument('--dataset', type=str, help='the dataset used for training', choices=datasets)
    parser.add_argument('--id', type=int, help='the id of the experiment')
    parser.add_argument('--gpu', type=str, help='the cuda devices used for training', default="0,1,2,3")
    parser.add_argument('--tqdm', type=bool, help='whether tqdm is on', default=False)
    parser.add_argument('--authors', type=int, help='number of authors', default=None)
    parser.add_argument('--samples-per-auth', type=int, help='number of samples per author', default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ensem-type', type=str, default='aggregate')

    # dataset - num of authors mapping
    default_num_authors = {
        'imdb62': 62,
        'enron': 50,
        'imdb': 100,
        'blog': 50,
        'ccat50': 50
    }

    # parse args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = args.dataset
    num_authors = args.authors if args.authors is not None else default_num_authors[args.dataset]

    # Load data and remove emails containing the sender's name
    df = load_dataset_dataframe(source)

    if args.authors is not default_num_authors[args.dataset]:
        warnings.warn(f"Number of authors for dataset {args.dataset} is {default_num_authors[args.dataset]}, "
                      f"but got {args.authors} instead. ")

    if args.samples_per_auth is not None:
        warnings.warn(f"Number of samples per author specified as {args.samples_per_auth}, which is a "
                      f"dangerous argument. ")

    logging.basicConfig(filename="Log_id_" + str(id) + "_.txt",
                        level=logging.INFO,
                        format='%(levelname)s: %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S')

    limit = args.authors
    print("Number of authors: ", limit)
    logging.info("Number of authors: " + str(limit))

    # Select top N senders and build Train and Test
    nlp_train, nlp_val, nlp_test, list_bigram, list_trigram = build_train_test(df, source, limit,
                                                                               per_author=args.samples_per_auth)

    # train an ensemble
    train_ensemble(nlp_train, nlp_test,
                BertClassiferHyperparams( # bert
                    mlp_size=512,
                    token_len=256,
                    embed_len=768
                ),
                BertClassiferHyperparams( # deberta
                    mlp_size=512,
                    token_len=372,
                    embed_len=768
                ),
                BertClassiferHyperparams( # roberta
                    mlp_size=512,
                    token_len=256,
                    embed_len=768
                ),
                BertClassiferHyperparams( # gpt2
                    mlp_size=512,
                    token_len=256,
                    embed_len=768
                ),
                num_epochs=args.epochs, base_bs=1, base_lr=1e-3, mlp_size=256, dropout=0.2, num_authors=num_authors, # tune - parameters for ensemble final layer LR
                ensemble_type=args.ensem_type)    #"simple", "fixed", "dynamic", "aggregate"
