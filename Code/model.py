from layer import *
from torch import nn

class AC_RNN(nn.Module):
    def __init__(self,num_company,d_feature,d_hidden,hidn_rnn,heads_att,hidn_att,crf_iters,alpha,pool_type):
        super(AC_RNN, self).__init__()
        self.heads_att = heads_att
        self.alpha = alpha
        self.num_company = num_company
        self.d_feature = d_feature
        self.d_hidden = d_hidden
        self.hidden_rnn = hidn_rnn
        self.hidn_att = hidn_att
        self.GRUs = GRUModel(d_feature, hidn_rnn)
        self.attentions = [Attention(hidn_rnn, hidn_att, num_company = self.num_company,alpha = self.alpha) for _ in range(heads_att)]
        self.crf = CRF_Att(hidn_rnn,hidn_rnn,crf_iters)
        self.pool_type = pool_type
        self.linear = Linear(num_company,hidn_rnn,2,bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self,x):

        x = self.GRUs(x)
        x = F.dropout(x, self.drop_out)
        ss = [att(x) for att in self.attentions]
        crf_embeddings = []
        for i in ss:
            crf_x_attention = self.crf(x,i)
            crf_embeddings.append(crf_x_attention)

        crf_x = myPool(self.pool_type,crf_embeddings)

        crf_output = F.elu(self.linear(crf_x))
        crf_output = F.softmax(crf_output, dim=1)

        return crf_output

class FocalLoss(nn.Module):

    def __init__(self,weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
