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

def train_ensemble(nlp_train, nlp_test,
                    bert_path, deberta_path, roberta_path, gpt2_path,
                    bert_hyperparams : BertClassiferHyperparams, deberta_hyperparams : BertClassiferHyperparams, roberta_hyperparams : BertClassiferHyperparams, gpt2_hyperparams : BertClassiferHyperparams,
                    num_epochs=10, base_bs=8, base_lr=1e-3, mlp_size=256, dropout=0.2, num_authors=5,
                    ensemble_type="dynamic"):
    print("#####")
    print("Training Ensemble")
    from models import LogisticRegression
    from dataset import BertDataset
    from models import BertClassifier
    from transformers import BertTokenizer, BertModel, DebertaTokenizer, DebertaModel, RobertaTokenizer, RobertaModel, GPT2Tokenizer, GPT2Model

    ngpus = torch.cuda.device_count()
    train_x, train_y = nlp_train['content'].tolist(), nlp_train['Target'].tolist()
    test_x, test_y = nlp_test['content'].tolist(), nlp_test['Target'].tolist()
    out_dim = max(test_y) + 1

    bertTokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    debertaTokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    robertaTokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    gpt2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2Tokenizer.pad_token = gpt2Tokenizer.eos_token  # for gpt tokenizer only

    ensemble_train_set = TransformerEnsembleDataset(train_x, train_y,
        [bertTokenizer, debertaTokenizer, robertaTokenizer, gpt2Tokenizer],
        [bert_hyperparams.token_len, deberta_hyperparams.token_len, roberta_hyperparams.token_len, gpt2_hyperparams.token_len])
    ensemble_test_set = TransformerEnsembleDataset(test_x, test_y,
        [bertTokenizer, debertaTokenizer, robertaTokenizer, gpt2Tokenizer],
        [bert_hyperparams.token_len, deberta_hyperparams.token_len, roberta_hyperparams.token_len, gpt2_hyperparams.token_len])

    ensemble_train_loader, ensemble_test_loader = DataLoader(ensemble_train_set, batch_size=base_bs * ngpus, shuffle=True, num_workers=12 * ngpus, pin_memory=True), \
                                                  DataLoader(ensemble_test_set, batch_size=base_bs * ngpus, shuffle=True, num_workers=12 * ngpus, pin_memory=True)

    bertExtractor = BertModel.from_pretrained('bert-base-cased')
    debertaExtractor = DebertaModel.from_pretrained('microsoft/deberta-base')
    robertaExtractor = RobertaModel.from_pretrained('roberta-base')
    gpt2Extractor = GPT2Model.from_pretrained('gpt2')

    extractors = [bertExtractor, debertaExtractor, robertaExtractor, gpt2Extractor]

    # Freeze all extractor params
    for extractor in extractors:
        for param in extractor.parameters():
            param.requires_grad = False
            
    bertModel = BertClassifier(bertExtractor,
        LogisticRegression(bert_hyperparams.embed_len * bert_hyperparams.token_len,
            bert_hyperparams.mlp_size, 5, dropout=0.0))
    debertaModel = BertClassifier(debertaExtractor,
        LogisticRegression(deberta_hyperparams.embed_len * deberta_hyperparams.token_len,
            deberta_hyperparams.mlp_size, 5, dropout=0.0))
    robertaModel = BertClassifier(robertaExtractor,
        LogisticRegression(roberta_hyperparams.embed_len * roberta_hyperparams.token_len,
            roberta_hyperparams.mlp_size, 5, dropout=0.0))
    gpt2Model = BertClassifier(gpt2Extractor,
        LogisticRegression(gpt2_hyperparams.embed_len * gpt2_hyperparams.token_len,
            gpt2_hyperparams.mlp_size, 5, dropout=0.0))

    bertModel = load_model_dic(bertModel, bert_path)
    debertaModel = load_model_dic(debertaModel, deberta_path)
    robertaModel = load_model_dic(robertaModel, roberta_path)
    gpt2Model = load_model_dic(gpt2Model, gpt2_path)

    for model in [bertModel, debertaModel, robertaModel, gpt2Model]:
        model = nn.DataParallel(model).cuda()

    final_test_acc = None
    if "simple" in ensemble_type:
        ensembleModel = SimpleEnsemble([bertModel, debertaModel, robertaModel, gpt2Model])
        ensembleModel = nn.DataParallel(ensembleModel).cuda()

        ensembleModel.eval()
        pg = tqdm(ensemble_test_loader, leave=False, total=len(ensemble_test_loader))
        with torch.no_grad():
            test_acc = AverageMeter()
            for i, (x, y) in enumerate(pg):
                x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
                x1 = (x1[0].cuda(), x1[1].cuda(), x1[2].cuda())
                x2 = (x2[0].cuda(), x2[1].cuda(), x2[2].cuda())
                x3 = (x3[0].cuda(), x3[1].cuda(), x3[2].cuda())
                x4 = (x4[0].cuda(), x4[1].cuda(), x4[2].cuda())
                y = y.cuda()
                pred = ensembleModel([x1, x2, x3, x4])
                test_acc.update((pred.argmax(1) == y).sum().item() / len(y))

                pg.set_postfix({
                    'test acc': '{:.6f}'.format(test_acc.avg),
                })
                final_test_acc = test_acc.avg
                print(f'test acc {final_test_acc}')
                return final_test_acc

    ensembleModel = DynamicWeightEnsemble([bertModel, debertaModel, robertaModel, gpt2Model],
                                            768 * (bert_hyperparams.token_len + deberta_hyperparams.token_len + roberta_hyperparams.token_len + gpt2_hyperparams.token_len),
                                            hidden_len=mlp_size) \
                    if "dynamic" in ensemble_type else \
                    AggregateFeatEnsemble([bertModel, debertaModel, robertaModel, gpt2Model],
                                            768 * (bert_hyperparams.token_len + deberta_hyperparams.token_len + roberta_hyperparams.token_len + gpt2_hyperparams.token_len),
                                            num_classes=num_authors,
                                            hidden_len=mlp_size) \
                    if "aggregate" in ensemble_type else \
                    FixedWeightEnsemble([bertModel, debertaModel, robertaModel, gpt2Model])

    ensembleModel = nn.DataParallel(ensembleModel).cuda()

    # training loop
    optimizer = torch.optim.AdamW(params=ensembleModel.parameters(), lr=base_lr * ngpus, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        if epoch == num_epochs - 1:
            ensemble_train_loader = DataLoader(ensemble_train_set, batch_size=base_bs * ngpus, shuffle=True, num_workers=12 * ngpus, pin_memory=True)
        train_acc = AverageMeter()
        train_loss = AverageMeter()

        ensembleModel.train()
        pg = tqdm(ensemble_train_loader, leave=False, total=len(ensemble_train_loader))
        for i, (x, y) in enumerate(pg):
            x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
            x1 = (x1[0].cuda(), x1[1].cuda(), x1[2].cuda())
            x2 = (x2[0].cuda(), x2[1].cuda(), x2[2].cuda())
            x3 = (x3[0].cuda(), x3[1].cuda(), x3[2].cuda())
            x4 = (x4[0].cuda(), x4[1].cuda(), x4[2].cuda())
            y = y.cuda()
            pred = ensembleModel([x1, x2, x3, x4])
            loss = criterion(pred, y.long())
            train_acc.update((pred.argmax(1) == y).sum().item() / len(y))
            train_loss.update(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pg.set_postfix({
                'train acc': '{:.6f}'.format(train_acc.avg),
                'train loss': '{:.6f}'.format(train_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

        ensembleModel.eval()
        pg = tqdm(ensemble_test_loader, leave=False, total=len(ensemble_test_loader))
        with torch.no_grad():
            test_acc = AverageMeter()
            for i, (x, y) in enumerate(pg):
                x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
                x1 = (x1[0].cuda(), x1[1].cuda(), x1[2].cuda())
                x2 = (x2[0].cuda(), x2[1].cuda(), x2[2].cuda())
                x3 = (x3[0].cuda(), x3[1].cuda(), x3[2].cuda())
                x4 = (x4[0].cuda(), x4[1].cuda(), x4[2].cuda())
                y = y.cuda()
                pred = ensembleModel([x1, x2, x3, x4])
                test_acc.update((pred.argmax(1) == y).sum().item() / len(y))

                pg.set_postfix({
                    'test acc': '{:.6f}'.format(test_acc.avg),
                    'epoch': '{:03d}'.format(epoch)
                })

        scheduler.step()

        print(f'epoch {epoch}, train acc {train_acc.avg}, test acc {test_acc.avg}')
        final_test_acc = test_acc.avg

        # save checkpoint
        save_model(os.path.join(ckpt_dir, "ensemble"),
                f'{id}_{out_dim}auth_hid{mlp_size}_epoch{num_epochs}_lr{base_lr}_bs{base_bs}_drop{dropout}_acc{final_test_acc:.5f}.pt',
                ensembleModel)

        for model in [bertModel, debertaModel, robertaModel, gpt2Model]:
            del model
        del ensembleModel
        del ensemble_train_loader, ensemble_test_loader

        return final_test_acc


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


def train_bert(nlp_train, nlp_val, nlp_test, tqdm_on, return_features=True, model_name='microsoft/deberta-base', embed_len=768,
               id=None):

    print("#####")
    print("Training BERT")
    from models import LogisticRegression
    from dataset import BertDataset
    from models import BertClassifier

    tokenizer, extractor = None, None
    if 'bert-base' in model_name:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained(model_name)
        extractor = BertModel.from_pretrained(model_name)
    elif 'deberta' in model_name:
        from transformers import DebertaTokenizer, DebertaModel
        tokenizer = DebertaTokenizer.from_pretrained(model_name)
        extractor = DebertaModel.from_pretrained(model_name)
    elif 'roberta' in model_name:  # roberta-base
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        extractor = RobertaModel.from_pretrained(model_name)
    elif 'gpt2' in model_name:
        from transformers import GPT2Tokenizer, GPT2Model
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        extractor = GPT2Model.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # for gpt tokenizer only
    elif 'gpt' in model_name:  # 'openai-gpt'
        from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
        extractor = OpenAIGPTModel.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.unk_token  # for gpt tokenizer only
        print(f'pad token {tokenizer.unk_token}')
    elif 'xlnet' in model_name:
        from transformers import XLNetTokenizer, XLNetModel
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        extractor = XLNetModel.from_pretrained(model_name)
    else:
        raise NotImplementedError(f"model {model_name} not implemented")

    # freeze extractor
    for param in extractor.parameters():
        param.requires_grad = True

    # business logic
    train_x, train_y = nlp_train['content'].tolist(), nlp_train['Target'].tolist()
    val_x, val_y = nlp_val['content'].tolist(), nlp_val['Target'].tolist()
    test_x, test_y = nlp_test['content'].tolist(), nlp_test['Target'].tolist()

    # training setup
    num_epochs, base_lr, base_bs, ngpus, dropout = 5, 1e-5, 6, torch.cuda.device_count(), 0.35
    num_tokens, hidden_dim, out_dim = 256, 512, max(test_y) + 1
    model = BertClassifier(extractor, LogisticRegression(embed_len * num_tokens, hidden_dim, out_dim, dropout=dropout))
    # model.load_state_dict(torch.load("0_deberta-base_coe1.0")) #load trained model
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr * ngpus, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    train_set = BertDataset(train_x, train_y, tokenizer, num_tokens)
    val_set = BertDataset(val_x, val_y, tokenizer, num_tokens)
    test_set = BertDataset(test_x, test_y, tokenizer, num_tokens)

    coefficient, temperature, sample_unit_size = 1.0, 0.1, 2
    print(f'coefficient, temperature, sample_unit_size = {coefficient, temperature, sample_unit_size}')
    logging.info(f'coefficient, temperature, sample_unit_size = {coefficient, temperature, sample_unit_size}')

    # recorder
    exp_dir = os.path.join(ckpt_dir, f'{id}_{model_name.split("/")[-1]}_coe{coefficient}_temp{temperature}_unit{sample_unit_size}_epoch{num_epochs}')
    writer = SummaryWriter(os.path.join(exp_dir, 'board'))

    train_sampler = TrainSamplerMultiClassUnit(train_set, sample_unit_size=sample_unit_size)
    train_loader = DataLoader(train_set, batch_size=base_bs * ngpus, sampler=train_sampler, shuffle=False,
                              num_workers=4 * ngpus, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                            pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                             pin_memory=True, drop_last=True)
    
    # training loop
    final_test_acc = None
    final_train_preds, final_test_preds = [], []
    train_feats, test_feats = [], []
    best_acc = -1

    for epoch in range(num_epochs):
        # if epoch == num_epochs - 1:
        #     train_loader = DataLoader(train_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=12 * ngpus,
        #                               pin_memory=True)
        train_acc = AverageMeter()
        train_loss = AverageMeter()
        train_loss_1 = AverageMeter()
        train_loss_2 = AverageMeter()

        # bad_classes = [16, 25, 39, 44, 48]
        bad_classes = [50]

        model.train()
        pg = tqdm(train_loader, leave=False, total=len(train_loader), disable=not tqdm_on)
        for i, (x1, x2, x3, y) in enumerate(pg):            # for x1, x2, x3, y in train_set:
            if not True in (y == 50).tolist():
                continue
            x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
            pred, feats = model(x, return_feat=True)

            # generate the mask
            mask = y.clone().cpu().apply_(lambda x: x not in bad_classes).type(torch.bool).cuda()
            pred = pred[mask]
            y_2 = y[mask]

            # classification loss
            loss_1 = criterion(pred, y_2.long())

            # generate the mask
            # mask = y.clone().cpu().apply_(lambda x: x not in bad_classes).type(torch.bool).cuda()
            # feats = feats[mask]
            # y_2 = y[mask]

            # import pdb
            # pdb.set_trace()

            # contrastive learning
            sim_matrix = compute_sim_matrix(feats)
            target_matrix = compute_target_matrix(y)
            loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

            # total loss
            loss = loss_1 + coefficient * loss_2

            train_acc.update((pred.argmax(1) == y_2).sum().item() / len(y))
            train_loss.update(loss.item())
            train_loss_1.update(loss_1.item())
            train_loss_2.update(loss_2.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pg.set_postfix({
                'train acc': '{:.6f}'.format(train_acc.avg),
                'train L1': '{:.6f}'.format(train_loss_1.avg),
                'train L2': '{:.6f}'.format(train_loss_2.avg),
                'train L': '{:.6f}'.format(train_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

        print('train acc: {:.6f}'.format(train_acc.avg), 'train L1 {:.6f}'.format(train_loss_1.avg),
              'train L2 {:.6f}'.format(train_loss_2.avg), 'train L {:.6f}'.format(train_loss.avg), f'epoch {epoch}')
        logging.info(f'epoch {epoch}, train acc {train_acc.avg}, train L1 {train_loss_1.avg}, '
                     f'train L2 {train_loss_2.avg}, train L {train_loss.avg}')

        # logger
        writer.add_scalar("train/L1", train_loss_1.avg, epoch)
        writer.add_scalar("train/L2", train_loss_2.avg, epoch)
        writer.add_scalar("train/L", train_loss.avg, epoch)
        writer.add_scalar("train/acc", train_acc.avg, epoch)

        model.eval()
        pg = tqdm(val_loader, leave=False, total=len(val_loader), disable=not tqdm_on)
        with torch.no_grad():
            test_acc = AverageMeter()
            test_loss_1 = AverageMeter()
            test_loss_2 = AverageMeter()
            test_loss = AverageMeter()
            for i, (x1, x2, x3, y) in enumerate(pg):
                if not True in (y == 50).tolist():
                    continue

                x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
                pred, feats = model(x, return_feat=True)

                # generate the mask
                mask = y.clone().cpu().apply_(lambda x: x not in bad_classes).type(torch.bool).cuda()
                pred = pred[mask]
                y_2 = y[mask]


                # classification
                loss_1 = criterion(pred, y_2.long())

                # contrastive learning
                sim_matrix = compute_sim_matrix(feats)
                target_matrix = compute_target_matrix(y)
                loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

                # total loss
                loss = loss_1 + coefficient * loss_2

                # logger
                test_acc.update((pred.argmax(1) == y_2).sum().item() / len(y))
                test_loss.update(loss.item())
                test_loss_1.update(loss_1.item())
                test_loss_2.update(loss_2.item())

                pg.set_postfix({
                    'test acc': '{:.6f}'.format(test_acc.avg),
                    'epoch': '{:03d}'.format(epoch)
                })

        # logger
        writer.add_scalar("test/L1", test_loss_1.avg, epoch)
        writer.add_scalar("test/L2", test_loss_2.avg, epoch)
        writer.add_scalar("test/L", test_loss.avg, epoch)
        writer.add_scalar("test/acc", test_acc.avg, epoch)

        # scheduler.step(test_loss.avg)
        scheduler.step()

        print(f'epoch {epoch}, train acc {train_acc.avg}, test acc {test_acc.avg}')
        logging.info(f'epoch {epoch}, train acc {train_acc.avg}, test acc {test_acc.avg}')
        final_test_acc = test_acc.avg

        best_acc = max(best_acc, test_acc.avg)

    # test
    model.eval()
    pg = tqdm(test_loader, leave=False, total=len(val_loader), disable=not tqdm_on)
    with torch.no_grad():
        test_acc_2 = AverageMeter()
        for i, (x1, x2, x3, y) in enumerate(pg):
            x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
            pred, feats = model(x, return_feat=True)
            test_acc_2.update((pred.argmax(1) == y).sum().item() / len(y))

    # save checkpoint
    save_model(exp_dir, f'{id}_val{final_test_acc:.5f}_test{test_acc_2.avg:.5f}.pt', model)

    print(f'Training complete after {num_epochs} epochs. Final val acc = {final_test_acc}, best val acc = {best_acc}.'
          f'Final test acc {test_acc_2.avg}')
    logging.info(f'Training complete after {num_epochs} epochs. Final val acc = {final_test_acc}, best val acc = {best_acc}.'
          f'Final test acc {test_acc_2.avg}')

    del model
    del train_loader, val_loader

    if return_features:
        return final_test_acc, final_train_preds, final_test_preds, train_feats, test_feats

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


def run_iterations(source, per_author, id, tqdm):
    # Load data and remove emails containing the sender's name
    df = load_dataset_dataframe(source)

    list_senders = [51]
    
    logging.basicConfig(filename="Log_id_" + str(id) + "_.txt",
                level=logging.INFO,
                format='%(levelname)s: %(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S')

    if source == "imdb62":
        list_senders = [62]

    # start training
    for limit in list_senders:
        print("Number of authors: ", limit)
        logging.info("Number of authors: " + str(limit))

        # Select top N senders and build Train and Test
        nlp_train, nlp_val, nlp_test, list_bigram, list_trigram = build_train_test(df, source, limit, per_author=None)

        # # TF-IDF + LR
        # final_test_acc, final_train_preds, final_test_preds = train_tf_idf(nlp_train, nlp_test, num_authors=limit)
        # print("Training done, accuracy is : ", final_test_acc)
        #
        # # Style-based classifier
        # score_style, style_prob_train, style_prob_test, style_feat_train, style_feat_test = train_style_based(nlp_train, nlp_test, return_features=True)
        # print("Training done, accuracy is : ", score_style)

        # train the ensemble
        # train_ensemble(nlp_train, nlp_test,
        #             'ckpt/bert-base-cased/23_5auth_256tokens_hid512_epoch1_lr0.0001_bs8_drop0.4_acc0.41800.pt',        # bert
        #             'ckpt/microsoft/deberta-base/22_5auth_372tokens_hid512_epoch5_lr1e-05_bs4_drop0.35_acc0.99513.pt', # deberta
        #             'ckpt/roberta-base/24_5auth_256tokens_hid512_epoch6_lr1e-05_bs8_drop0.35_acc0.99693.pt',           # roberta
        #             'ckpt/gpt2/18_5auth_256tokens_hid512_epoch1_lr1e-05_bs8_drop0.3.pt',                               # gpt2
        #
        #             # Copy checkpoint parameters from Google sheet
        #             BertClassiferHyperparams( # bert
        #                 mlp_size=512,
        #                 token_len=256,
        #                 embed_len=768
        #             ),
        #             BertClassiferHyperparams( # deberta
        #                 mlp_size=512,
        #                 token_len=372,
        #                 embed_len=768
        #             ),
        #             BertClassiferHyperparams( # roberta
        #                 mlp_size=512,
        #                 token_len=256,
        #                 embed_len=768
        #             ),
        #             BertClassiferHyperparams( # gpt2
        #                 mlp_size=512,
        #                 token_len=256,
        #                 embed_len=768
        #             ),
        #             num_epochs=10, base_bs=8, base_lr=1e-3, mlp_size=256, dropout=0.2, num_authors=5, # tune - parameters for ensemble final layer LR
        #             ensemble_type="dynamic")    #"simple", "fixed", "dynamic", "aggregate"

        # Bert + Classification Layer
        score_bert, bert_prob_train, bert_prob_test, bert_feat_train, bert_feat_test = train_bert(nlp_train, nlp_val,
                                                                                                  nlp_test,
                                                                                                  tqdm_on=tqdm,
                                                                                                  return_features=True,
                                                                                                  model_name='bert-base-cased',
                                                                                                  id=id)
          
        print("Training done, accuracy is : ", score_bert)
        logging.info("Training done, accuracy is : " + str(score_bert))

        # # Character N-gram only
        # score_char, char_prob_train, char_prob_test, char_feat_train, char_feat_test = train_char_ngram(nlp_train, nlp_test, list_bigram, list_trigram, return_features=True)
        # print("Training done, accuracy is : ", score_char)

    print(f'authors = {list_senders}')
