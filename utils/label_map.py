import torch


def sem2onehot(n, labelmap):
    label_map = labelmap.long().unsqueeze(1).cuda()
    bs, _, h, w = label_map.size()
    nc = n
    input_label = torch.FloatTensor(bs, nc, h, w).zero_().cuda()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return input_semantics
