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
            # nn.Softmax()
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
        feature = self.bert(x[0], x[1])
        # out = self.fc(feature.pooler_output.flatten(1))
        out = self.fc(feature.last_hidden_state.flatten(1))
        if return_feat:
            return out, feature.pooler_output
        return out
