import argparse
import os
import torch
import numpy
import random
import warnings
import logging
from learning_methods import train_ensemble, train_bert
from utils import load_dataset_dataframe, build_train_test, build_train_test_ntg
from models import BertClassiferHyperparams


random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)


if __name__ == '__main__':
    datasets = ['imdb62', 'enron', 'imdb', 'blog', 'ccat50', 'ccat10', 'ntg', 'turing']
    parser = argparse.ArgumentParser(description=f'Training models for datasets {datasets}')
    parser.add_argument('--dataset', type=str, help='the dataset used for training', choices=datasets)
    parser.add_argument('--id', type=str, help='the id of the experiment')
    parser.add_argument('--gpu', type=str, help='the cuda devices used for training', default="0,1,2,3")
    parser.add_argument('--tqdm', type=bool, help='whether tqdm is on', default=False)
    parser.add_argument('--authors', type=int, help='number of authors', default=None)
    parser.add_argument('--samples-per-auth', type=int, help='number of samples per author', default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--ensem-type', type=str, default='aggregate')
    parser.add_argument('--model', type=str, default='microsoft/deberta-base')
    parser.add_argument('--train-ensemble', type=bool, default=False)
    parser.add_argument('--coe', type=float, default=1)

    # dataset - num of authors mapping
    default_num_authors = {
        'imdb62': 62,
        'enron': 50,
        'imdb': 100,
        'blog': 50,
        'ccat50': 50,
        'ccat10': 10,
        'ntg': 9,        # 2
        'turing': 20
    }

    # parse args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = args.dataset
    num_authors = args.authors if args.authors is not None else default_num_authors[args.dataset]
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))  # print all args

    # masked classes
    if args.authors == 100:
        mask_classes = {
            'blog': [],
            'imdb62': [],
            'imdb': [],
            'enron': []
        }
    elif args.authors == 50:
        mask_classes = {
            'blog': [],
            'imdb62': [],
            'imdb':[],
            'enron': []
        }
    elif args.authors == 10:
        mask_classes = {
            'blog': [],
            'imdb62': [],
            'imdb': [],
            'enron': [],
            'ccat10': []
        }
    else:
        mask_classes = {
            'blog': [],
            'imdb62': [],
            'imdb': [],
            'enron': [],
            'ccat10': [],
            'ccat50': [],
            'ntg': [],
            'turing': []
        }

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

    limit = num_authors
    print("Number of authors: ", limit)
    logging.info("Number of authors: " + str(limit))

    # Select top N senders and build Train and Test
    if source != 'ntg':
        nlp_train, nlp_test, list_bigram, list_trigram = build_train_test(df, source, limit,
                                                                          per_author=args.samples_per_auth)
    else:
        nlp_train, nlp_test, list_bigram, list_trigram = build_train_test_ntg(df, source, limit,
                                                                              per_author=args.samples_per_auth)

    # train an ensemble
    if args.train_ensemble:
        train_ensemble(nlp_train, nlp_test,
                    BertClassiferHyperparams( # bert
                        mlp_size=512,
                        token_len=256,
                        embed_len=768
                    ),
                    BertClassiferHyperparams( # deberta
                        mlp_size=512,
                        token_len=256,  # 372
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
                    num_epochs=args.epochs, base_bs=2, base_lr=1e-5, mlp_size=256, dropout=0.2, num_authors=num_authors, # tune - parameters for ensemble final layer LR
                    ensemble_type=args.ensem_type, model_id=args.id)    #"simple", "fixed", "dynamic", "aggregate"
    else:
        train_bert(nlp_train, nlp_test, args.tqdm, args.model, 768, args.id, args.epochs, base_bs=8, base_lr=1e-5,
                   mask_classes=mask_classes[args.dataset], coefficient=args.coe, num_authors=num_authors)
