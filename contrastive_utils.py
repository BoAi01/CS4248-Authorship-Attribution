import torch
import torch.nn.functional as F


def compute_sim_matrix(feats):
    """
    Takes in a batch of features of size (bs, feat_len).
    """
    # compute sim matrix. Exactly correct.
    # mod = (feats * feats).sum(1) ** 0.5
    # feats = feats / mod.unsqueeze(1).expand(-1, feats.size(1))
    # sim_matrix = (torch.matmul(feats, feats.transpose(0, 1)) / temperature).exp()

    # a more elegant implementation of computing similarity matrix
    sim_matrix = F.cosine_similarity(feats.unsqueeze(2).expand(-1, -1, feats.size(0)),
                                     feats.unsqueeze(2).expand(-1, -1, feats.size(0)).transpose(0, 2),
                                     dim=1)

    return sim_matrix


def compute_target_matrix(labels):
    """
    Takes in a label vector of size (bs)
    """
    # construct the target similarity matrix
    # target_matrix = torch.zeros(sim_matrix.shape).cuda()
    # for i in range(target_matrix.size(0)):
    #     bool_mask = (y == y[i]).type(torch.float)
    #     # target_matrix[i] = bool_mask / (bool_mask.sum() + 1e-8)      # normalize s.t. sum up to 1. Wrong, no need to norm!
    #     target_matrix[i] = bool_mask

    label_matrix = labels.unsqueeze(-1).expand((labels.shape[0], labels.shape[0]))
    trans_label_matrix = torch.transpose(label_matrix, 0, 1)
    target_matrix = (label_matrix == trans_label_matrix).type(torch.float)

    return target_matrix


def contrastive_loss(pred_sim_matrix, target_matrix, temperature, labels):
    # set class 50 to all negative
    syn_class = 50
    target_matrix[labels == syn_class] = target_matrix[labels == syn_class] / 10
    target_matrix[:, labels == syn_class] = target_matrix[:, labels == syn_class] / 10
    pred_sim_matrix[labels == syn_class] = pred_sim_matrix[labels == syn_class] / 10
    pred_sim_matrix[:, labels == syn_class] = pred_sim_matrix[:, labels == syn_class] / 10

    return F.kl_div(F.softmax(pred_sim_matrix/temperature).log(), F.softmax(target_matrix/temperature),
                    reduction="batchmean", log_target=False)
