import torch.nn as nn
from GAT import GAT


class GMHAAE(nn.Module):
    def __init__(self, input_feat_dim, dropout, out_dim1=128, out_dim2=256, out_dim3=384, alpha=0.2, nheads=8):
        super(GMHAAE, self).__init__()

        self.gc1 = GAT(input_feat_dim, 128, out_dim1, dropout, alpha, nheads)
        self.gc2 = GAT(out_dim1, 256, out_dim2, dropout, alpha, nheads)
        self.gc3 = GAT(out_dim2, 512, out_dim3, dropout, alpha, nheads)
        self.dropout = dropout
        self.gc4 = GAT(out_dim3, 512, out_dim2, dropout, alpha, nheads)
        self.gc5 = GAT(out_dim2, 256, out_dim1, dropout, alpha, nheads)
        self.gc6 = GAT(out_dim1, 128, input_feat_dim, dropout, alpha, nheads)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        hidden2 = self.gc2(hidden1, adj)
        hidden3 = self.gc3(hidden2, adj)
        return hidden3

    def decode(self, x, adj):
        hidden4 = self.gc4(x, adj)
        hidden5 = self.gc5(hidden4, adj)
        hidden6 = self.gc6(hidden5, adj)
        return hidden6

    def forward(self, x, adj):
        hidden = self.encode(x, adj)
        output = self.decode(hidden, adj)
        return output

