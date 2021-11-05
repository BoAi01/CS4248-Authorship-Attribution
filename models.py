import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0):
        super().__init__()
        print(f'Logistic Regression classifier of dim ({in_dim} {hid_dim} {out_dim})')

        self.nn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dim, bias=True),
        )

    def forward(self, x):
        return self.nn(x)


class MLP2Layer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0):
        super().__init__()
        print(f'Logistic Regression classifier of dim ({in_dim} {hid_dim} {out_dim})')

        self.nn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, hid_dim / 2, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dim, bias=True),
        )

    def forward(self, x):
        return self.nn(x)


class BertFeatExtractor(nn.Module):
    FEAT_LEN = 768

    def __init__(self, raw_bert):
        super(BertFeatExtractor, self).__init__()
        self.bert = raw_bert

    def forward(self, x):
        # x is a tokenized input
        return self.bert(**x).pooler_output


class BertClassifier(nn.Module):
    FEAT_LEN = 768

    def __init__(self, raw_bert, classifier):
        super().__init__()
        self.bert = raw_bert
        self.fc = classifier

    def forward(self, x, return_feat=False):
        # x is a tokenized input
        # feature = self.bert(input_ids=x[0], token_type_ids=x[1], attention_mask=x[2])
        feature = self.bert(input_ids=x[0], attention_mask=x[2])
        # out = self.fc(feature.pooler_output.flatten(1))       # not good for our task
        out = self.fc(feature.last_hidden_state.flatten(1))
        if return_feat:
            return out, feature
        return out


class EnsembleClassifier(nn.Module):
    FEAT_LEN = 768

    def __init__(self, raw_bert, styleClassifier, charClassifier, bertClassifier, finalClassifier):
        super().__init__()
        self.bert = raw_bert
        self.styleClassifier = styleClassifier
        self.charClassifier = charClassifier
        self.bertClassifier = bertClassifier
        self.finalClassifier = finalClassifier

    def forward(self, x, return_feat=False):
        # x is a tokenized input
        # print("ENS Forward")
        
        stylePred = self.styleClassifier(x[0])

        charPred = self.charClassifier(x[1])

        bertFeature = self.bert(x[2], x[3]).last_hidden_state.flatten(1)
        bertPred = self.bertClassifier(bertFeature)
        # print(stylePred.shape)
        # print(charPred.shape)
        # print(bertFeature.shape)
        # print(bertPred.shape)
        # print(x[0].shape)
        # print(x[1].shape)
        ensembleTensor = torch.cat((stylePred, charPred, bertPred, x[0], x[1], bertFeature), dim=1)
        # out = self.fc(feature.pooler_output.flatten(1))
        out = self.finalClassifier(ensembleTensor)
        if return_feat:
            return out, bertFeature
        return out
