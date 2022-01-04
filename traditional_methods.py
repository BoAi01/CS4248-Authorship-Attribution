from typing import final

from pandas.core import base
from utils import *  # bad practice, nvm

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import NumpyDataset, TransformerEnsembleDataset, TrainSamplerMultiClass, TrainSampler, TrainSamplerMultiClassUnit
from models import AggregateFeatEnsemble, DynamicWeightEnsemble, LogisticRegression, BertClassiferHyperparams, SimpleEnsemble, FixedWeightEnsemble
from tqdm import tqdm
import time
import torch.nn.functional as F
from contrastive_utils import compute_sim_matrix, compute_target_matrix, contrastive_loss

import logging

ckpt_dir = 'exp_data'


def train_model(model, train_set, train_loader, test_loader, criterion, scheduler, optimizer, num_epochs=1,
                ckpt_name=None, model_name=None, hidden_dim=10, base_lr=1e-4, base_bs=8, out_dim=5, dropout=0.3):
    """
    A training loop implementation suitable for most models
    """

    final_test_acc = None
    final_train_preds, final_test_preds = [], []

    for epoch in range(num_epochs):
        if epoch == num_epochs - 1:
            raise NotImplementedError()
            ngpus = torch.cuda.device_count()
            train_loader = DataLoader(train_set, batch_size=32 * ngpus, shuffle=False, num_workers=12 * ngpus,
                                      pin_memory=True)

        train_acc, train_loss = AverageMeter(), AverageMeter()
        pg = tqdm(train_loader, leave=False, total=len(train_loader))
        for i, (x, y) in enumerate(pg):
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            if epoch == num_epochs - 1:  # for feature ensemble prediction
                final_train_preds.append(pred.cpu().detach())
            loss = criterion(pred, y.long())  # target must be of dtype = Long!
            train_acc.update((pred.argmax(1) == y).sum().item() / len(y))
            train_loss.update(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pg.set_postfix({
                'train acc': '{:.6f}'.format(train_acc.avg),
                'train loss': '{:.6f}'.format(train_loss.avg),
                'epoch': '{:03d}'.format(epoch + 1)
            })

        pg = tqdm(test_loader, leave=False, total=len(test_loader))
        test_acc = AverageMeter()
        with torch.no_grad():
            for i, (x, y) in enumerate(pg):
                x, y = x.cuda(), y.cuda()
                pred = model(x)
                if epoch == num_epochs - 1:  # for feature ensemble prediction
                    final_test_preds.append(pred.cpu().detach())
                test_acc.update((pred.argmax(1) == y).sum().item() / len(y))

                pg.set_postfix({
                    'test acc': '{:.6f}'.format(test_acc.avg),
                    'epoch': '{:03d}'.format(epoch + 1)
                })

        scheduler.step()

        print(f'epoch {epoch + 1}, train acc {train_acc.avg}, test acc {test_acc.avg}')
        final_test_acc = test_acc.avg

    # save checkpoint
    save_model(os.path.join(ckpt_dir, model_name),
               f'{out_dim}auth_hid{hidden_dim}_epoch{num_epochs}_lr{base_lr}_bs{base_bs}_drop{dropout}_acc{final_test_acc:.5f}.pt',
               model)


    final_train_preds, final_test_preds = torch.cat(final_train_preds, dim=0), torch.cat(final_test_preds, dim=0)

    return final_test_acc, final_train_preds, final_test_preds


def train_char_ngram(nlp_train, nlp_test, list_bigram, list_trigram, return_features=False):
    print("#####")
    print("Character N-gram")

    feats_train = nlp_train['content'].apply(lambda x: find_freq_n_gram_in_txt(x, list_bigram, list_trigram)).values
    feats_test = nlp_test['content'].apply(lambda x: find_freq_n_gram_in_txt(x, list_bigram, list_trigram)).values

    feats_train = pd.DataFrame(feats_train)[0].apply(lambda x: pd.Series(x))
    feats_test = pd.DataFrame(feats_test)[0].apply(lambda x: pd.Series(x))

    train_x, train_y = feats_train.to_numpy(), nlp_train['Target'].to_numpy()
    test_x, test_y = feats_test.to_numpy(), nlp_test['Target'].to_numpy()
    num_epochs, base_lr, base_bs, ngpus = 100, 5e-2, 32, torch.cuda.device_count()
    in_dim, out_dim = train_x.shape[1], test_y.max() + 1
    model = LogisticRegression(in_dim=in_dim, hid_dim=in_dim * 3, out_dim=out_dim, dropout=0.3)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr * ngpus, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_set, test_set = NumpyDataset(train_x, train_y), NumpyDataset(test_x, test_y)
    train_loader = DataLoader(train_set, batch_size=base_bs * ngpus, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=0, pin_memory=True)

    model = nn.DataParallel(model).cuda()

    # this sometimes works and sometimes doesn't, no idea why
    final_test_acc, final_train_preds, final_test_preds = train_model(model, train_set, train_loader=train_loader,
                                                                      test_loader=test_loader, criterion=criterion,
                                                                      scheduler=scheduler, optimizer=optimizer,
                                                                      num_epochs=num_epochs)
    del model
    del train_loader, test_loader

    if return_features:
        return final_test_acc, final_train_preds, final_test_preds, train_x, test_x

    return final_test_acc, final_train_preds, final_test_preds


def train_style_based(nlp_train, nlp_test, return_features=False):
    print("#####")
    print("Training style classifier")

    X_style_train = nlp_train[
        ["avg_len", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d", "f_e", "f_f", "f_g", "f_h",
         "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s", "f_t", "f_u", "f_v", "f_w",
         "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_e_0",
         "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9", "f_e_10", "f_e_11",
         "richness"]]
    X_style_test = nlp_test[
        ["avg_len", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d", "f_e", "f_f", "f_g", "f_h",
         "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s", "f_t", "f_u", "f_v", "f_w",
         "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_e_0",
         "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9", "f_e_10", "f_e_11",
         "richness"]]

    train_x, train_y = X_style_train.to_numpy(), nlp_train['Target'].to_numpy()
    test_x, test_y = X_style_test.to_numpy(), nlp_test['Target'].to_numpy()
    num_epochs, base_lr, base_bs, ngpus = 100, 5e-2, 32, torch.cuda.device_count()
    in_dim, out_dim = train_x.shape[1], test_y.max() + 1
    model = LogisticRegression(in_dim=in_dim, hid_dim=out_dim * 3, out_dim=out_dim, dropout=0.3)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr * ngpus, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_set, test_set = NumpyDataset(train_x, train_y), NumpyDataset(test_x, test_y)
    train_loader = DataLoader(train_set, batch_size=base_bs * ngpus, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=0, pin_memory=True)

    model = nn.DataParallel(model).cuda()

    score_style, style_prob_train, style_prob_test = train_model(model, train_set, train_loader=train_loader,
                                                                 test_loader=test_loader, criterion=criterion,
                                                                 scheduler=scheduler, optimizer=optimizer,
                                                                 num_epochs=num_epochs)

    del model
    del train_loader, test_loader

    if return_features:
        return score_style, style_prob_train, style_prob_test, train_x, test_x
    return score_style, style_prob_train, style_prob_test


def train_tf_idf(nlp_train, nlp_test, num_authors=5):
    print("#####")
    print("Training TF-IDF")

    # data
    vectorizer = TfidfVectorizer()  # ngram_range=(1,2), max_features=3000
    train_x, train_y = vectorizer.fit_transform(nlp_train['content_tfidf']), nlp_train['Target']
    test_x, test_y = vectorizer.transform(nlp_test['content_tfidf']), nlp_test['Target']

    train_x, train_y = train_x.toarray(), train_y.to_numpy()
    test_x, test_y = test_x.toarray(), test_y.to_numpy()

    # training setup
    num_epochs, base_lr, base_bs, ngpus, dropout = 9, 1e-5, 32, torch.cuda.device_count(), 0.0
    in_dim, out_dim, hidden_dim = train_x.shape[1], test_y.max()+1, 20
    model = LogisticRegression(in_dim=in_dim, hid_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr*ngpus, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_set, test_set = NumpyDataset(train_x, train_y), NumpyDataset(test_x, test_y)
    train_loader = DataLoader(train_set, batch_size=base_bs*ngpus, shuffle=True, num_workers=12*ngpus, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=base_bs*ngpus, shuffle=False, num_workers=12*ngpus, pin_memory=True)

    # load model into multi-gpu
    model = nn.DataParallel(model).cuda()

    return train_model(model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, scheduler=scheduler, optimizer=optimizer, train_set=train_set,
                        num_epochs=num_epochs, base_lr=base_lr, base_bs=base_bs, model_name='tf_idf', out_dim=num_authors, dropout=dropout, hidden_dim=hidden_dim)


