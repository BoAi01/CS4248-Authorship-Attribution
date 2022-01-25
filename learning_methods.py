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
from dataset import NumpyDataset, TransformerEnsembleDataset, TrainSamplerMultiClass, TrainSampler, \
    TrainSamplerMultiClassUnit
from models import AggregateFeatEnsemble, DynamicWeightEnsemble, LogisticRegression, BertClassiferHyperparams, \
    SimpleEnsemble, FixedWeightEnsemble
from tqdm import tqdm
import time
import torch.nn.functional as F
from contrastive_utils import compute_sim_matrix, compute_target_matrix, contrastive_loss

from models import LogisticRegression
from dataset import BertDataset
from models import BertClassifier
from transformers import BertTokenizer, BertModel, DebertaTokenizer, DebertaModel, RobertaTokenizer, RobertaModel, \
    GPT2Tokenizer, GPT2Model

import logging
from models import LogisticRegression
from dataset import BertDataset
from models import BertClassifier
from sklearn.metrics import f1_score

ckpt_dir = 'exp_data'


def train_ensemble(nlp_train, nlp_test,
                   bert_hyperparams: BertClassiferHyperparams, deberta_hyperparams: BertClassiferHyperparams,
                   roberta_hyperparams: BertClassiferHyperparams, gpt2_hyperparams: BertClassiferHyperparams,
                   num_epochs, base_bs=8, base_lr=1e-3, mlp_size=256, dropout=0.2, num_authors=5,
                   ensemble_type="dynamic", model_id=id):
    print("#####")
    print("Training Contrastive Ensemble")

    ngpus = torch.cuda.device_count()
    train_x, train_y = nlp_train['content'].tolist(), nlp_train['Target'].tolist()
    test_x, test_y = nlp_test['content'].tolist(), nlp_test['Target'].tolist()
    out_dim = max(test_y) + 1

    bertTokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    debertaTokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta')
    # robertaTokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # gpt2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # gpt2Tokenizer.pad_token = gpt2Tokenizer.eos_token  # for gpt tokenizer only

    # ensemble_train_set = TransformerEnsembleDataset(train_x, train_y,
    #     [bertTokenizer, debertaTokenizer, robertaTokenizer, gpt2Tokenizer],
    #     [bert_hyperparams.token_len, deberta_hyperparams.token_len, roberta_hyperparams.token_len, gpt2_hyperparams.token_len])
    # ensemble_test_set = TransformerEnsembleDataset(test_x, test_y,
    #     [bertTokenizer, debertaTokenizer, robertaTokenizer, gpt2Tokenizer],
    #     [bert_hyperparams.token_len, deberta_hyperparams.token_len, roberta_hyperparams.token_len, gpt2_hyperparams.token_len])
    ensemble_train_set = TransformerEnsembleDataset(train_x, train_y,
                                                    [bertTokenizer, debertaTokenizer],
                                                    [bert_hyperparams.token_len, deberta_hyperparams.token_len])
    ensemble_test_set = TransformerEnsembleDataset(test_x, test_y,
                                                   [bertTokenizer, debertaTokenizer],
                                                   [bert_hyperparams.token_len, deberta_hyperparams.token_len])
    train_sampler = TrainSamplerMultiClassUnit(ensemble_train_set, sample_unit_size=2)

    ensemble_train_loader = DataLoader(ensemble_train_set, batch_size=base_bs * ngpus, sampler=train_sampler,
                                       shuffle=False, num_workers=8 * ngpus, pin_memory=True)
    ensemble_test_loader = DataLoader(ensemble_test_set, batch_size=base_bs * ngpus, shuffle=True,
                                      num_workers=8 * ngpus, pin_memory=True)

    bertExtractor = BertModel.from_pretrained('bert-base-cased').cuda()
    debertaExtractor = DebertaModel.from_pretrained('microsoft/deberta-base').cuda()
    # robertaExtractor = RobertaModel.from_pretrained('roberta-base').cuda()
    # gpt2Extractor = GPT2Model.from_pretrained('gpt2')

    extractors = [bertExtractor, debertaExtractor]

    # Freeze all extractor params
    for extractor in extractors:
        for param in extractor.parameters():
            param.requires_grad = True

    bertModel = BertClassifier(bertExtractor,
                               LogisticRegression(bert_hyperparams.embed_len * bert_hyperparams.token_len,
                                                  bert_hyperparams.mlp_size, num_authors, dropout=0.2))
    bertModel.load_state_dict(torch.load(
        "./exp_data/1b_bert-base-cased_coe1.0_temp0.1_unit2_epoch8/1b_val0.66041_e4.pt"))  # load trained model
    debertaModel = BertClassifier(debertaExtractor,
                                  LogisticRegression(deberta_hyperparams.embed_len * deberta_hyperparams.token_len,
                                                     deberta_hyperparams.mlp_size, num_authors, dropout=0.2))
    debertaModel.load_state_dict(
        torch.load("./exp_data/0c_deberta-base_coe1.0_temp0.1_unit2_epoch8/0c_val0.69859_e4.pt"))
    # robertaModel = BertClassifier(robertaExtractor,
    #     LogisticRegression(roberta_hyperparams.embed_len * roberta_hyperparams.token_len,
    #         roberta_hyperparams.mlp_size, num_authors, dropout=0.2))
    # gpt2Model = BertClassifier(gpt2Extractor,
    #     LogisticRegression(gpt2_hyperparams.embed_len * gpt2_hyperparams.token_len,
    #         gpt2_hyperparams.mlp_size, num_authors, dropout=0.2))

    # for model in [bertModel, debertaModel, robertaModel, gpt2Model]:
    #     # model = nn.DataParallel(model).cuda()

    final_test_acc = None
    best_acc = -1

    # if "simple" in ensemble_type:
    #     ensembleModel = SimpleEnsemble([bertModel, debertaModel, robertaModel, gpt2Model])
    #     ensembleModel = nn.DataParallel(ensembleModel).cuda()
    #
    #     ensembleModel.eval()
    #     pg = tqdm(ensemble_test_loader, leave=False, total=len(ensemble_test_loader))
    #     with torch.no_grad():
    #         test_acc = AverageMeter()
    #         for i, (x, y) in enumerate(pg):
    #             x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    #             x1 = (x1[0].cuda(), x1[1].cuda(), x1[2].cuda())
    #             x2 = (x2[0].cuda(), x2[1].cuda(), x2[2].cuda())
    #             x3 = (x3[0].cuda(), x3[1].cuda(), x3[2].cuda())
    #             x4 = (x4[0].cuda(), x4[1].cuda(), x4[2].cuda())
    #             y = y.cuda()
    #             pred = ensembleModel([x1, x2, x3, x4])
    #             test_acc.update((pred.argmax(1) == y).sum().item() / len(y))
    #
    #             pg.set_postfix({
    #                 'test acc': '{:.6f}'.format(test_acc.avg),
    #             })
    #             final_test_acc = test_acc.avg
    #             print(f'test acc {final_test_acc}')
    #             return final_test_acc

    ensembleModel = DynamicWeightEnsemble([bertModel, debertaModel, robertaModel],
                                          768 * (
                                                  bert_hyperparams.token_len + deberta_hyperparams.token_len + roberta_hyperparams.token_len + gpt2_hyperparams.token_len),
                                          hidden_len=mlp_size) \
        if "dynamic" in ensemble_type else \
        AggregateFeatEnsemble([bertModel, debertaModel],  # , robertaModel
                              # 768 * (bert_hyperparams.token_len + deberta_hyperparams.token_len
                              #        + roberta_hyperparams.token_len),
                              768 * (bert_hyperparams.token_len + deberta_hyperparams.token_len),
                              num_classes=num_authors,
                              hidden_len=mlp_size) \
            if "aggregate" in ensemble_type else \
            FixedWeightEnsemble([bertModel, debertaModel, robertaModel, gpt2Model])

    # training loop
    optimizer = torch.optim.AdamW(params=ensembleModel.parameters(), lr=base_lr * ngpus, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # contrastive learning
    temperature, coefficient = 0.1, 1
    ensembleModel = nn.DataParallel(ensembleModel).cuda()

    for epoch in range(num_epochs):
        # if epoch == num_epochs - 1:
        #     ensemble_train_loader = DataLoader(ensemble_train_set, batch_size=base_bs * ngpus, shuffle=True,
        #                                        num_workers=12 * ngpus, pin_memory=True)
        train_acc = AverageMeter()
        train_loss = AverageMeter()
        train_loss_1 = AverageMeter()
        train_loss_2 = AverageMeter()
        train_loss_3 = AverageMeter()

        ensembleModel.train()
        pg = tqdm(ensemble_train_loader, leave=False, total=len(ensemble_train_loader))
        for i, (x, y) in enumerate(pg):
            # x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
            x1, x2 = x[0], x[1]
            x1 = (x1[0].cuda(), x1[1].cuda(), x1[2].cuda())
            x2 = (x2[0].cuda(), x2[1].cuda(), x2[2].cuda())
            # x3 = (x3[0].cuda(), x3[1].cuda(), x3[2].cuda())
            # x4 = (x4[0].cuda(), x4[1].cuda(), x4[2].cuda())
            # xs = [x1, x2, x3]
            # feats = [model(*x) for x, model in zip(xs, extractors)]
            # pred = ensembleModel(torch.cat(feats, dim=1))
            y = y.cuda()

            # inference
            pred, feats, preds = ensembleModel([x1, x2], return_feats=True, return_preds=True)  # x2, x3
            feats = torch.cat(feats, dim=0)
            ys = torch.cat([y, y], dim=0)
            #             feats = torch.cat(feats, dim=1)
            #             ys = y
            # loss 1
            loss_1 = criterion(pred, y.long())

            # contrastive learning
            sim_matrix = compute_sim_matrix(feats)
            #             sim_matrix = compute_sim_matrix(hidden_feat)
            target_matrix = compute_target_matrix(ys)
            loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, ys)

            # loss 3
            loss_3 = 0.0
            for pred in preds:
                loss_3 += criterion(pred, y.long())
            #             loss_3 = loss_2

            # total loss
            loss = loss_1 + coefficient * loss_2 + loss_3

            train_acc.update((pred.argmax(1) == y).sum().item() / len(y))
            train_loss.update(loss.item())
            train_loss_1.update(loss_1.item())
            train_loss_2.update(loss_2.item())
            train_loss_3.update(loss_3.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pg.set_postfix({
                'train acc': '{:.6f}'.format(train_acc.avg),
                'train L1': '{:.6f}'.format(train_loss_1.avg),
                'train L2': '{:.6f}'.format(train_loss_2.avg),
                'train L3': '{:.6f}'.format(train_loss_3.avg),
                'train L': '{:.6f}'.format(train_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

        print('train acc: {:.6f}'.format(train_acc.avg), 'train L1 {:.6f}'.format(train_loss_1.avg),
              'train L2 {:.6f}'.format(train_loss_2.avg), 'train L3 {:.6f}'.format(train_loss_3.avg),
              'train L {:.6f}'.format(train_loss.avg), f'epoch {epoch}')

        exp_dir = os.path.join(ckpt_dir, "ensemble")

        ensembleModel.eval()
        pg = tqdm(ensemble_test_loader, leave=False, total=len(ensemble_test_loader))
        with torch.no_grad():
            test_acc = AverageMeter()
            for i, (x, y) in enumerate(pg):
                x1, x2 = x[0], x[1]
                x1 = (x1[0].cuda(), x1[1].cuda(), x1[2].cuda())
                x2 = (x2[0].cuda(), x2[1].cuda(), x2[2].cuda())
                # x3 = (x3[0].cuda(), x3[1].cuda(), x3[2].cuda())
                # x4 = (x4[0].cuda(), x4[1].cuda(), x4[2].cuda())
                y = y.cuda()
                pred = ensembleModel([x1, x2])  # x2, x3
                test_acc.update((pred.argmax(1) == y).sum().item() / len(y))

                pg.set_postfix({
                    'test acc': '{:.6f}'.format(test_acc.avg),
                    'epoch': '{:03d}'.format(epoch)
                })

        scheduler.step()

        print(f'epoch {epoch}, train acc {train_acc.avg}, test acc {test_acc.avg}')
        final_test_acc = test_acc.avg

        if test_acc.avg:
            if test_acc.avg >= best_acc:
                cur_models = os.listdir(exp_dir)
                for cur_model in cur_models:
                    if cur_model.endswith(".pt"):
                        os.remove(os.path.join(exp_dir, cur_model))
                save_model(exp_dir, f'{model_id}_val{final_test_acc:.5f}_e{epoch}.pt', ensembleModel)
        best_acc = max(best_acc, test_acc.avg)

        # save checkpoint
    #         save_model(os.path.join(ckpt_dir, "ensemble"),
    #                    f'{id}_{out_dim}auth_hid{mlp_size}_epoch{num_epochs}_lr{base_lr}_bs{base_bs}_drop{dropout}_acc{final_test_acc:.5f}.pt',
    #                    ensembleModel)

    # for model in [bertModel, debertaModel, robertaModel, gpt2Model]:
    #     del model
    # del ensembleModel
    # del ensemble_train_loader, ensemble_test_loader

    return final_test_acc


def train_bert(nlp_train, nlp_val, tqdm_on, model_name, embed_len, id, num_epochs, base_bs, base_lr,
               mask_classes, coefficient, num_authors):
    print(f'mask classes = {mask_classes}')
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

    # update extractor
    for param in extractor.parameters():
        param.requires_grad = True
    # print([k for k, v in list(extractor.named_parameters())])
    # import pdb
    # pdb.set_trace()
    # for i in range(8, 12):
    #    for param in extractor.encoder.layer[i].parameters():
    #        param.requires_grad = False

    # business logic
    train_x, train_y = nlp_train['content'].tolist(), nlp_train['Target'].tolist()
    val_x, val_y = nlp_val['content'].tolist(), nlp_val['Target'].tolist()
    # test_x, test_y = nlp_test['content'].tolist(), nlp_test['Target'].tolist()

    # for ntg only, otherwise uncomment the above
    # train_x, train_y = nlp_train[0].tolist(), nlp_train[1].tolist()
    # val_x, val_y = nlp_val[0].tolist(), nlp_val[1].tolist()

    # training setup
    ngpus, dropout = torch.cuda.device_count(), 0.35
    num_tokens, hidden_dim, out_dim = 256, 512, num_authors
    model = BertClassifier(extractor, LogisticRegression(embed_len * num_tokens, hidden_dim, out_dim, dropout=dropout))

    # model.load_state_dict(torch.load(
    #     "/home/aibo/aa/aa3/exp_data/3b_s75B_bert-base-cased_coe1.0_temp0.1_unit2_epoch8/3b_s75B_val0.62372_e6.pt"))  # load trained model
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr * ngpus, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    train_set = BertDataset(train_x, train_y, tokenizer, num_tokens)
    val_set = BertDataset(val_x, val_y, tokenizer, num_tokens)
    # test_set = BertDataset(test_x, test_y, tokenizer, num_tokens)

    temperature, sample_unit_size = 0.1, 2
    print(f'coefficient, temperature, sample_unit_size = {coefficient, temperature, sample_unit_size}')
    logging.info(f'coefficient, temperature, sample_unit_size = {coefficient, temperature, sample_unit_size}')

    # recorder
    exp_dir = os.path.join(ckpt_dir,
                           f'{id}_{model_name.split("/")[-1]}_coe{coefficient}_temp{temperature}_unit{sample_unit_size}_epoch{num_epochs}')
    writer = SummaryWriter(os.path.join(exp_dir, 'board'))

    # load the training data
    train_sampler = TrainSamplerMultiClassUnit(train_set, sample_unit_size=sample_unit_size)
    train_loader = DataLoader(train_set, batch_size=base_bs * ngpus, sampler=train_sampler, shuffle=False,
                              num_workers=4 * ngpus, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                            pin_memory=True, drop_last=True)
    # test_loader = DataLoader(test_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
    #                          pin_memory=True, drop_last=True)

    # training loop
    final_test_acc = None
    final_train_preds, final_test_preds = [], []
    train_feats, test_feats = [], []
    best_acc = -1

    for epoch in range(num_epochs):
        train_acc = AverageMeter()
        train_loss = AverageMeter()
        train_loss_1 = AverageMeter()
        train_loss_2 = AverageMeter()

        # decay coefficient
        # coefficient = coefficient - 1 / num_epochs

        model.train()
        pg = tqdm(train_loader, leave=False, total=len(train_loader), disable=not tqdm_on)
        for i, (x1, x2, x3, y) in enumerate(pg):  # for x1, x2, x3, y in train_set:
            x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
            pred, feats = model(x, return_feat=True)

            # classification loss
            loss_1 = criterion(pred, y.long())

            # generate the mask
            mask = y.clone().cpu().apply_(lambda x: x not in mask_classes).type(torch.bool).cuda()
            feats, pred, y = feats[mask], pred[mask], y[mask]
            if len(y) == 0:
                continue

            # contrastive learning
            sim_matrix = compute_sim_matrix(feats)
            target_matrix = compute_target_matrix(y)
            loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

            # total loss
            loss = loss_1 + coefficient * loss_2

            # train_acc.update((pred.argmax(1) == y).sum().item() / len(y))
            train_acc.update(f1_score(y.cpu().detach().numpy(), pred.argmax(1).cpu().detach().numpy(), average='macro'))
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
                x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
                pred, feats = model(x, return_feat=True)

                # classification
                loss_1 = criterion(pred, y.long())

                # contrastive learning
                sim_matrix = compute_sim_matrix(feats)
                target_matrix = compute_target_matrix(y)
                loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

                # total loss
                loss = loss_1 + coefficient * loss_2

                # logger
                # test_acc.update((pred.argmax(1) == y).sum().item() / len(y))
                test_acc.update(
                    f1_score(y.cpu().detach().numpy(), pred.argmax(1).cpu().detach().numpy(), average='macro'))
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

        if test_acc.avg:
            if test_acc.avg >= best_acc:
                cur_models = os.listdir(exp_dir)
                for cur_model in cur_models:
                    if cur_model.endswith(".pt"):
                        os.remove(os.path.join(exp_dir, cur_model))
                save_model(exp_dir, f'{id}_val{final_test_acc:.5f}_e{epoch}.pt', model)
        best_acc = max(best_acc, test_acc.avg)

    # test for predictions only
    model.eval()
    pg = tqdm(val_loader, leave=False, total=len(val_loader), disable=False)
    output_list = torch.randn((6, 10)).cuda()
    feature_list = torch.randn((6, 100)).cuda()
    label_list = []
    m = torch.nn.AvgPool2d((1, 2000), stride=(1, 1950))
    with torch.no_grad():
        test_acc_2 = AverageMeter()
        for i, (x1, x2, x3, y) in enumerate(pg):
            x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
            pred, feats = model(x, return_feat=True)
            condensed_feats = m(feats.unsqueeze(0)).squeeze(0)
            if i == 0:
                output_list = pred
                feature_list = condensed_feats
                label_list = [y]
            else:
                output_list = torch.cat([output_list, pred], dim=0)
                feature_list = torch.cat([feature_list, condensed_feats], dim=0)
                label_list.append(y)

    torch.save(output_list, "./exp_data/output_list_" + str(id) + ".pt")
    torch.save(label_list, "./exp_data/label_list_" + str(id) + ".pt")
    torch.save(feature_list, "./exp_data/feature_list_" + str(id) + ".pt")

    #     import pdb
    #     pdb.set_trace()

    # model.eval()
    # pg = tqdm(test_loader, leave=False, total=len(val_loader), disable=not tqdm_on)
    # with torch.no_grad():
    #     test_acc_2 = AverageMeter()
    #     for i, (x1, x2, x3, y) in enumerate(pg):
    #         x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
    #         pred, feats = model(x, return_feat=True)
    #         test_acc_2.update((pred.argmax(1) == y).sum().item() / len(y))

    # save checkpoint
    save_model(exp_dir, f'{id}_val{final_test_acc:.5f}_finale{epoch}.pt', model)

    print(f'Training complete after {num_epochs} epochs. Final val acc = {final_test_acc}, best val acc = {best_acc}.'
          f'Final test acc {final_test_acc}')
    logging.info(
        f'Training complete after {num_epochs} epochs. Final val acc = {final_test_acc}, best val acc = {best_acc}.'
        f'Final test acc {final_test_acc}')

    #     if return_features:
    #         return final_test_acc, final_train_preds, final_test_preds, train_feats, test_feats

    return final_test_acc, final_train_preds, final_test_preds
