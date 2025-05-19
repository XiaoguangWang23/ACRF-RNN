
import torch.nn.functional as F
from torch import nn
import torch
from util4 import *


class CRF_Att(nn.Module):

    def __init__(self, input_dim, output_dim, num_iters):
        super(CRF_Att, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_iters = num_iters
        self.al= nn.Parameter(torch.zeros(1).type(torch.FloatTensor))
        self.beta = nn.Parameter(torch.zeros(1).type(torch.FloatTensor))

    def forward(self,inputs,similarity):

        support = similarity
        normalize = torch.sum(support, dim= 1)

        normalize = torch.Tensor.repeat(torch.unsqueeze(normalize, -1), [1, self.input_dim])

        al = torch.exp(self.al)
        beta = torch.exp(self.beta)


        output = inputs
        iters = torch.tensor(0)
        cond = lambda iters, num_iters: torch.le(iters, self.num_iters)
        while cond(iters,output):
            output = (inputs * beta + (dot(support, output) + output) * al) \
                     / (beta + normalize * al + al)
            iters = torch.add(iters, 1)

        result = output
        return result




class Linear(nn.Module):
    def __init__(self,num_nodes, input_size, hidden_size, bias=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(num_nodes,input_size,hidden_size))
        self.b = nn.Parameter(torch.zeros(num_nodes,hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x):
        output = torch.bmm(x.unsqueeze(1), self.W)
        output = output.squeeze(1)
        if self.bias:
            output = output + self.b
        return output

class GRUModel(nn.Module):
    def __init__(self,input_dim, hidden_dim, bias=True):
        super(GRUModel,self).__init__()
        self.GRU_layer1 = nn.GRU(input_size=input_dim, hidden_size=2*hidden_dim)
        self.GRU_layer2 = nn.GRU(input_size=2*hidden_dim, hidden_size=hidden_dim)
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)


    def forward(self,x):
        x,hidden1 = self.GRU_layer1(x)
        x,hidden2 = self.GRU_layer2(x)
        hidden = hidden2.squeeze(0)

        return hidden


class Attention(nn.Module):
    def __init__(self, in_features, out_features, alpha , num_company,residual=False):
        # in_features = 12 ; out_feature = 6 head_att = 2
        super(Attention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.num_company = num_company

        self.seq_transformation_r = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

        self.W_static = nn.Parameter(torch.zeros(self.num_company,self.num_company).type(torch.FloatTensor), requires_grad=True)

        self.w_1 = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.w_2 = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.coef_revise = False
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_r, relation_static = None):
        num_stock = input_r.shape[0]
        seq_r = torch.transpose(input_r, 0, 1).unsqueeze(0)
        logits = torch.zeros(num_stock, num_stock, dtype=input_r.dtype)
        seq_fts_r = self.seq_transformation_r(seq_r)
        f_1 = self.f_1(seq_fts_r)
        f_2 = self.f_2(seq_fts_r)
        logits += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
        if relation_static != None:
            logits += torch.mul(relation_static, self.W_static)
        coefs = self.leakyrelu(logits)
        if not isinstance(self.coef_revise,torch.Tensor):
            self.coef_revise = torch.zeros(self.num_company, self.num_company) + 1.0 - torch.eye(self.num_company, self.num_company)
        coefs_eye = coefs.mul(self.coef_revise)

        return coefs_eye

def myPool(type,crf_embeddings):
    head = len(crf_embeddings)
    company_num, feature_num = crf_embeddings[0].size()
    stacked_tensor = torch.stack(crf_embeddings, dim=1)
    pool_embedding = stacked_tensor.view(-1,feature_num)
    pool_embedding = pool_embedding.t()
    if type=='MAX':
        pool = nn.MaxPool1d(kernel_size=head,stride=head)
        pooled_embedding = pool(pool_embedding)
        pooled_embedding = pooled_embedding.t()
    elif type == 'AVG':
        pool = nn.AvgPool1d(kernel_size=head,stride=head)
        pooled_embedding = pool(pool_embedding)
        pooled_embedding = pooled_embedding.t()
    elif type == 'SUM':
        stacked_tensor2 = torch.stack(crf_embeddings, dim=0)
        pooled_embedding = torch.sum(stacked_tensor2, dim= 0)

    return pooled_embedding


