from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0):
        super().__init__()
        print(f'Logistic Regression classifier of dim ({in_dim} {hid_dim} {out_dim})')

        self.nn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(hid_dim, out_dim, bias=True),
            # nn.Softmax()
        )

    def forward(self, x):
        return self.nn(x)
