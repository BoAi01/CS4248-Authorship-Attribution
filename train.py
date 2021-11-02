from typing import final
from utils import *     # bad practice, nvm

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NumpyDataset
from models import LogisticRegression
from tqdm import tqdm
import time

def train_model(model, train_loader, test_loader, criterion, scheduler, optimizer, num_epochs=5):
    # training loop
    final_test_acc = None
    final_train_preds, final_test_preds = None, None
    for epoch in range(num_epochs):
        train_acc = AverageMeter()
        train_loss = AverageMeter()
        pg = tqdm(train_loader, leave=False, total=len(train_loader))
        for i, (x, y) in enumerate(pg):
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            if epoch == num_epochs - 1: # for feature ensemble prediction
                final_train_preds = pred if final_train_preds == None else torch.cat((final_train_preds, pred), 0)
            loss = criterion(pred, y.long())      # target must be of dtype = Long!
            train_acc.update(((pred.argmax(1) == y).sum() / len(y)).item())     # .item() discard gradient
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
        with torch.no_grad():
            test_acc = AverageMeter()
            for i, (x, y) in enumerate(pg):
                x, y = x.cuda(), y.cuda()
                pred = model(x)
                if epoch == num_epochs - 1: # for feature ensemble prediction
                    final_test_preds = pred if final_test_preds == None else torch.cat((final_test_preds, pred), 0)
                test_acc.update(((pred.argmax(1) == y).sum() / len(y)).item())     # .item() discard gradient

                pg.set_postfix({
                    'test acc': '{:.6f}'.format(test_acc.avg),
                    'epoch': '{:03d}'.format(epoch + 1)
                })

        scheduler.step()

        print(f'epoch {epoch + 1}, train acc {train_acc.avg}, test acc {test_acc.avg}')
        final_test_acc = test_acc.avg
    return final_test_acc, final_train_preds, final_test_preds

def train_tf_idf(nlp_train, nlp_test):
    print("#####")
    print("Training TF-IDF")

    # data
    vectorizer = TfidfVectorizer()       # ngram_range=(1,2), max_features=3000
    train_x, train_y = vectorizer.fit_transform(nlp_train['content_tfidf']), nlp_train['Target']
    test_x, test_y = vectorizer.transform(nlp_test['content_tfidf']), nlp_test['Target']

    train_x, train_y = train_x.toarray(), train_y.to_numpy()
    test_x, test_y = test_x.toarray(), test_y.to_numpy()
    print(train_x)
    print(train_x.shape)
    return 0

    # training setup
    num_epochs, base_lr, base_bs, ngpus = 5, 5e-4, 32, 4
    in_dim, out_dim = train_x.shape[1], test_y.max()+1
    model = LogisticRegression(in_dim=in_dim, hid_dim=out_dim * 2, out_dim=out_dim, dropout=0.2)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr*ngpus, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_set, test_set = NumpyDataset(train_x, train_y), NumpyDataset(test_x, test_y)
    train_loader = DataLoader(train_set, batch_size=base_bs*ngpus, shuffle=True, num_workers=12*ngpus, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=base_bs*ngpus, shuffle=False, num_workers=12*ngpus, pin_memory=True)

    # load model into multi-gpu
    model = nn.DataParallel(model).cuda()

    return train_model(model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, scheduler=scheduler, optimizer=optimizer)


def train_bert(nlp_train, nlp_test, return_features=True):
    print("#####")
    print("Training BERT")
    from models import BertFeatExtractor, LogisticRegression
    from dataset import BertDataset
    from transformers import BertTokenizer, BertModel
    from models import BertClassifier

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    extractor = BertModel.from_pretrained("bert-base-cased")

    # freeze extractor
    for param in extractor.parameters():
        param.requires_grad = False

    train_x, train_y = nlp_train['content'].tolist(), nlp_train['Target'].tolist()
    test_x, test_y = nlp_test['content'].tolist(), nlp_test['Target'].tolist()

    # training setup
    num_epochs, base_lr, base_bs, ngpus = 5, 2e-5, 64, 1
    out_dim = max(test_y) + 1
    model = BertClassifier(extractor, LogisticRegression(768 * 128, 128, out_dim, dropout=0.3))
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr * ngpus, weight_decay=5e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_set, test_set = BertDataset(train_x, train_y, tokenizer), BertDataset(test_x, test_y, tokenizer)

    train_loader = DataLoader(train_set, batch_size=base_bs * ngpus, shuffle=True, num_workers=12 * ngpus,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=12 * ngpus,
                             pin_memory=True)

    # training loop
    final_test_acc = None
    final_train_preds, final_test_preds = None, None
    train_feats, test_feats = None, None

    for epoch in range(num_epochs):
        train_acc = AverageMeter()
        train_loss = AverageMeter()

        pg = tqdm(train_loader, leave=False, total=len(train_loader))
        for i, (x1, x2, y) in enumerate(pg):
            x, y = (x1.cuda(), x2.cuda()), y.cuda()
            pred = model(x)
            if epoch == num_epochs - 1: # for feature ensemble prediction
                p, f = model(x, return_feat=True)
                f = torch.flatten(f.last_hidden_state, start_dim=1)
                train_feats = f if (train_feats == None) else torch.cat((train_feats, f), 0)
                final_train_preds = p if (final_train_preds == None) else torch.cat((final_train_preds, p), 0)
            loss = criterion(pred, y.long())
            train_acc.update(((pred.argmax(1) == y).sum() / len(y)).item())
            train_loss.update(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pg.set_postfix({
                'train acc': '{:.6f}'.format(train_acc.avg),
                'train loss': '{:.6f}'.format(train_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

        pg = tqdm(test_loader, leave=False, total=len(test_loader))
        with torch.no_grad():
            test_acc = AverageMeter()
            for i, (x1, x2, y) in enumerate(pg):
                x, y = (x1.cuda(), x2.cuda()), y.cuda()
                pred = model(x)
                if epoch == num_epochs - 1: # for feature ensemble prediction
                    p, f = model(x, return_feat=True)
                    f = torch.flatten(f.last_hidden_state, start_dim=1)
                    test_feats = f if (test_feats == None) else torch.cat((test_feats, f), 0)
                    final_test_preds = p if (final_test_preds == None) else torch.cat((final_test_preds, p), 0)
                test_acc.update(((pred.argmax(1) == y).sum() / len(y)).item())

                pg.set_postfix({
                    'test acc': '{:.6f}'.format(test_acc.avg),
                    'epoch': '{:03d}'.format(epoch)
                })

        scheduler.step()

        print(f'epoch {epoch}, train acc {train_acc.avg}, test acc {test_acc.avg}')
        final_test_acc = test_acc.avg

    print(train_feats.shape)
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
    num_epochs, base_lr, base_bs, ngpus = 5, 5e-4, 32, 1
    in_dim, out_dim = train_x.shape[1], test_y.max()+1
    model = LogisticRegression(in_dim=in_dim, hid_dim=out_dim * 2, out_dim=out_dim, dropout=0.2)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr*ngpus, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_set, test_set = NumpyDataset(train_x, train_y), NumpyDataset(test_x, test_y)
    train_loader = DataLoader(train_set, batch_size=base_bs*ngpus, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=base_bs*ngpus, shuffle=False, num_workers=0, pin_memory=True)

    model = nn.DataParallel(model).cuda()

    score_style, style_prob_train, style_prob_test = train_model(model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, scheduler=scheduler, optimizer=optimizer)

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
    num_epochs, base_lr, base_bs, ngpus = 5, 5e-4, 32, 1
    in_dim, out_dim = train_x.shape[1], test_y.max()+1
    model = LogisticRegression(in_dim=in_dim, hid_dim=in_dim * 2, out_dim=out_dim, dropout=0.2)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr*ngpus, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_set, test_set = NumpyDataset(train_x, train_y), NumpyDataset(test_x, test_y)
    train_loader = DataLoader(train_set, batch_size=base_bs*ngpus, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=base_bs*ngpus, shuffle=False, num_workers=0, pin_memory=True)

    model = nn.DataParallel(model).cuda()

    # this sometimes works and sometimes doesn't, no idea why
    final_test_acc, final_train_preds, final_test_preds = train_model(model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, scheduler=scheduler, optimizer=optimizer)
    
    del model
    del train_loader, test_loader

    if return_features:
        return final_test_acc, final_train_preds, final_test_preds, train_x, test_x
    return final_test_acc, final_train_preds, final_test_preds


def run_iterations(source):
    # Load data and remove emails containing the sender's name
    df = load_dataset_dataframe(source)

    # list_senders = [5, 10, 25, 50, 75, 100]
    list_senders = [5]

    if source == "imdb62":
        list_senders = [62]

    # start training
    list_scores = []
    for limit in list_senders:
        print("Number of authors: ", limit)

        # Select top N senders and build Train and Test
        nlp_train, nlp_test, list_bigram, list_trigram = build_train_test(df, limit)

        # TF-IDF + LR
        # score_lr = train_tf_idf(nlp_train, nlp_test)
        # print("Training done, accuracy is : ", score_lr)

        # # Style-based classifier
        score_style, style_prob_train, style_prob_test, style_feat_train, style_feat_test = train_style_based(nlp_train, nlp_test, return_features=True)
        print("Training done, accuracy is : ", score_style)
        print(style_prob_train.shape)
        print(style_prob_test.shape)
        print(style_feat_train.shape)
        print(style_feat_test.shape)

        # Bert + Classification Layer
        score_bert, bert_prob_train, bert_prob_test, bert_feat_train, bert_feat_test = train_bert(nlp_train, nlp_test, return_features=True)
        print("Training done, accuracy is : ", score_bert)
        print(bert_prob_train.shape)
        print(bert_prob_test.shape)
        print(bert_feat_train.shape)
        print(bert_feat_test.shape)

        # # Character N-gram only
        score_char, char_prob_train, char_prob_test, char_feat_train, char_feat_test = train_char_ngram(nlp_train, nlp_test, list_bigram, list_trigram, return_features=True)
        print("Training done, accuracy is : ", score_char)
        print(char_prob_train.shape)
        print(char_prob_test.shape)
        print(char_feat_train.shape)
        print(char_feat_test.shape)

        # # BERT + Style + Char N-gram
        print("#####")
        print("BERT + Style + Char N-gram")

        bert_prob_train = bert_prob_train.cpu().detach().numpy()
        bert_feat_train = bert_feat_train.cpu().detach().numpy()
        style_prob_train = style_prob_train.cpu().detach().numpy()
        char_prob_train = char_prob_train.cpu().detach().numpy()

        bert_prob_test = bert_prob_test.cpu().detach().numpy()
        bert_feat_test = bert_feat_test.cpu().detach().numpy()
        style_prob_test = style_prob_test.cpu().detach().numpy()
        char_prob_test = char_prob_test.cpu().detach().numpy()
        
        # print(bert_prob_train.shape)
        # print(bert_feat_train.shape)

        ensemble_train_inputs = np.concatenate([bert_prob_train, bert_feat_train, style_prob_train, style_feat_train, char_prob_train, char_feat_train], axis=1)
        ensemble_test_inputs = np.concatenate([bert_prob_test, bert_feat_test, style_prob_test, style_feat_test, char_prob_test, char_feat_test], axis=1)

        print(ensemble_train_inputs.shape)
        print(ensemble_test_inputs.shape)

        train_y, test_y = nlp_train['Target'].to_numpy(), nlp_test['Target'].to_numpy()

        num_epochs, base_lr, base_bs, ngpus = 100, 5e-4, 32, 1
        in_dim, out_dim = ensemble_train_inputs.shape[1], test_y.max()+1
        model = LogisticRegression(in_dim=in_dim, hid_dim=128, out_dim=out_dim, dropout=0.2)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr*ngpus, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        train_set, test_set = NumpyDataset(ensemble_train_inputs, train_y), NumpyDataset(ensemble_test_inputs, test_y)
        train_loader = DataLoader(train_set, batch_size=base_bs*ngpus, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=base_bs*ngpus, shuffle=False, num_workers=0, pin_memory=True)

        model = nn.DataParallel(model).cuda()

        # this sometimes works and sometimes doesn't, no idea why
        final_test_acc, final_train_preds, final_test_preds = train_model(model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, scheduler=scheduler, optimizer=optimizer, num_epochs=num_epochs)

        # ensemble_train_feats = np.concatenate([bert_prob_train, style_prob_train, char_prob_train], axis=1)
        # ensemble_test_feats = np.concatenate([bert_prob_test, style_prob_test, char_prob_test], axis=1)
        # clf = LogisticRegression(random_state=0).fit(ensemble_train_feats, nlp_train['Target'])
        # y_pred = clf.predict(ensemble_test_feats)
        # score_comb_fin = accuracy_score(nlp_test['Target'], y_pred)

        # print("Training done, accuracy is : ", score_comb_fin)

        # # Store scores
        # list_scores.append([limit, score_bert, score_style, score_comb_fin])

    list_scores = np.array(list_scores)

    return list_scores
