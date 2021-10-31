import torch
import torch.nn as nn
from transformers.modeling_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer


class Config():
    n_head = 12
    attn_pdrop = 0.1
    resid_pdrop = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, nx, config=Config(), scale=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=800 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    # Muti-head attention
    def _attn(self, q, k, v):
        # q*k=k，shape=[batch size, n_head=8, seq_len, seq_len]
        w = torch.matmul(q, k)
        # scale dot attention
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        # softmax
        w = nn.Softmax(dim=-1)(w)
        # dropout
        w = self.attn_dropout(w)

        # attention weight*value get attention output
        outputs = (torch.matmul(w, v),)

        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
            self,
            hidden_states
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        #  make query multi head
        #  [batch size, seq_len ,800] --> [batch size, n_head=8, seq_len ,800/8=100]
        query = self.split_heads(query)
        # make key multi head
        #  [batch size, seq_len ,768] --> [batch size, n_head=8 ,800/8=100, seq_len]
        key = self.split_heads(key, k=True)
        # make value muti head
        #  [batch size, seq_len ,768] --> [batch size, n_head=8, seq_len ,800/8=100]
        value = self.split_heads(value)

        # attn_outputs[0]: atttention输出，shape=[batch size, n_head=8, seq_len, 800/8=100]
        # attn_outputs[1]: atttention weight, shape=[batch size, n_head=8, seq_len, shape=[batch size, n_head=8, seq_len, seq_len]
        attn_outputs = self._attn(query, key, value)
        a = attn_outputs[0]

        # merge：shape=[batch size, seq_len, 800]
        a = self.merge_heads(a)
        # fully connect：shape=[batch size, seq_len, 800]
        a = self.c_proj(a)
        # dropout：shape=[batch size, seq_len, 800]
        a = self.resid_dropout(a)

        return a  # a
