
import numpy as np
import pandas as pd
import torch
from model import *
from torch import optim
import argparse
# import seaborn as sns



parser = argparse.ArgumentParser()


parser.add_argument('--rnn-length', type=int, default='5', help='rnn length')
parser.add_argument('--heads-att', type=int, default='3',help='attention heads')
parser.add_argument('--hidn-att', type=int, default='32',help='attention hidden nodes')
parser.add_argument('--hidn-rnn', type=int, default='64',help='rnn hidden nodes')
parser.add_argument('--crf-iter',type=int, default='5',help='crf iter')
parser.add_argument('--max-epoch',type=int, default='800',help='max epoch')
parser.add_argument('--wait-epoch',type=int, default='150',help='wait epoch')
parser.add_argument('--device', type=str, default='0',help='GPU to use')
parser.add_argument('--alpha', type=float, default='0.5',help = 'alpha value')
parser.add_argument('--batch-train',type=int, default='4',help='train batch')
parser.add_argument('--clip', type=float, default='1',help ='clip')
parser.add_argument('--seed',type=int, default = '26541',help ='seed')
parser.add_argument('--lr', type=float, default='1e-2', help='Learning rate ')
parser.add_argument('--gamma',type = int, default='1',help= 'gamma')
parser.add_argument('--n-weight',type=float, default='0.1',help='the weight of fraud class')
parser.add_argument('--f-weight', type=float, default='0.15',help ='the weight of normal class')
parser.add_argument('--pool-type',type =int, default='3',help = 'the type of pooling layer')
parser.add_argument('--test-year',type =int, default='2017',help = 'the test year')
pool_dic = {1:'MAX',2:'AVG',3:'SUM'}

save_flag = True

args = parser.parse_args()
lr = args.lr
rnn_len = args.rnn_length
MAX_EPOCH = args.max_epoch
hidn_rnn = args.hidn_rnn
heads_att = args.heads_att
hidn_att = args.hidn_att
crf_iter = args.crf_iter
clip = args.clip
batch_train = args.batch_train
alpha = args.alpha
gamma = args.gamma
f_weight = args.f_weight
n_weight = args.n_weight
DEVICE = "cuda:" + args.device
myseed =args.seed
waits = args.wait_epoch
drop_out = args.dropout
test_year= args.test_year
pool_type = args.pool_type
pool_type = pool_dic[pool_type]

torch.manual_seed(myseed)
torch.cuda.manual_seed(myseed)
np.random.seed(myseed)
random.seed(myseed)


def load_dataset(path):

    df = pd.read_csv(path)

    gr_year = df.groupby('year')
    data =[]
    for i,y in gr_year:
        data_i = []
        data.append(data_i)
        gr_com = y.groupby('Symbol')
        for j,c in gr_com:
            c = c.iloc[:,:-2]
            data_i.append(c.values.tolist())

    data = np.array((data))

    data = np.squeeze(data,axis=2)
    data = torch.from_numpy(data).float()
    data.to()
    y = data[:,:,-1]
    y = np.expand_dims(y,axis = 2)
    y = torch.from_numpy(y).long()
    x = data[:,:,:-1]

    test_line = test_year-2021-1
    if test_line == -1:
        x_train = x[:-1]
        x_eval = x[-1 - rnn_len:]
        y_train = y[: -1]
        y_eval = y[-1 - rnn_len:]
    else:
        x_train = x[: test_line]
        x_eval = x[test_line - rnn_len: test_line+1]
        y_train = y[: test_line]
        y_eval = y[test_line - rnn_len: test_line+1]

    return x_train,x_eval,y_train,y_eval

def train(model,x_train,y_train):
    model.train()
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_len:]  # 【3……11】
    r = random.seed(myseed)
    random.shuffle(train_seq,random=r)
    total_loss = 0
    total_loss_count = 0

    preds = []
    trues = []
    plt_outputs = []

    for i in train_seq:
        y_train_i = torch.squeeze(y_train[i],1)
        output = model(x_train[i - rnn_len + 1: i + 1])

        output =output.type(torch.FloatTensor)
        loss = criterion(output, y_train_i)
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1


        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        output1 = output.detach().cpu()
        preds.append(output1.numpy())
        trues.append(y_train[i].cpu().numpy())
        plt_outputs.append(plt_output.detach().cpu().numpy())


    optimizer.step()
    optimizer.zero_grad()


    train_acc, train_recall, train_pre,train_auc = train_metrics(trues, preds)

    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    fi_loss = total_loss / total_loss_count

    return fi_loss,train_acc,train_recall,train_pre,train_auc

def evaluate(model, x_eval, y_eval):
    model.eval()

    seq_len = len(x_eval)

    seq = list(range(seq_len))[rnn_len:]

    plt_outputs = []
    preds = []
    trues = []
    for i in seq:
        output = model(x_eval[i - rnn_len + 1: i + 1])
        output = output.type(torch.FloatTensor)

        output = output.detach().cpu()
        preds.append(output.numpy())
        trues.append(y_eval[i].cpu().numpy())
        plt_outputs.append(plt_output.detach().cpu().numpy())

    l = metrics(trues, preds)

    return l

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path = 'data.csv'
    x_train, x_eval, y_train, y_eval= load_dataset(path)

    # print(x_test.shape)
    print("succes load data！")
    num_company = x_train.size(1)
    d_feature = x_train.size(2)
    d_hidden = d_feature
    best_model_file = 0
    epoch = 0
    eval_epoch_best = 0
    loss_min = 1
    wait_epoch = 0

    w1 = [n_weight,f_weight]
    class_weight = torch.tensor(w1)
    criterion = FocalLoss(gamma=gamma,weight=class_weight)

    model = AC_RNN(num_company=num_company,d_feature=d_feature,
                   d_hidden=d_hidden,hidn_rnn=hidn_rnn,heads_att=heads_att,
                   hidn_att=hidn_att,crf_iters=crf_iter,alpha = alpha,pool_type=pool_type)
    optimizer = optim.Adam(model.parameters(), lr= lr)
    train_losses = []
    train_recalls = []
    train_accs = []
    train_pres = []
    train_aucs = []
    eval_accs = []
    eval_recalls_macros =[]
    eval_kss = []
    eval_gms = []
    eval_list = [eval_accs,eval_recalls_macros,eval_kss,eval_gms]

    while epoch < MAX_EPOCH:
        # print(epoch)
        train_loss,train_acc,train_recall,train_pre,train_auc = train(model, x_train, y_train)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_recalls.append(train_recall)
        train_pres.append(train_pre)
        train_aucs.append(train_auc)

        tra_str = 'epoch{}:\ntrain_loss{:.4f}, train_acc{:.4f}, train_recall{:.4f}\n'.format(epoch, train_loss, train_acc, train_recall)

        e_list = evaluate(model, x_eval, y_eval)
        for i in range(4):
            eval_list[i].append(e_list[i])

        print('loss:{:.4f}; train_Auc:{:.4f}; train_recall:{:.4f}; eval_acc{:.4f}'.format(train_loss,train_auc,train_recall,e_list[0]))

        eval_str = 'eval_acc{:.4f},eval_recall_macro{:.4f}, eval_ks{:.4f}, eval_gm{:.4f}\n'\
            .format(e_list[0],e_list[1],e_list[2],e_list[3],e_list[4],e_list[5],e_list[6],e_list[7],e_list[8],e_list[9])
        final_Str = tra_str + eval_str

        if train_loss < loss_min:
            loss_min = train_loss
            best_epoch = epoch

            eval_best_list = [epoch,e_list[0],e_list[1],e_list[2],e_list[3]]

            eval_best_str = 'epcoch{}, eval_acc{:.4f},eval_recall_macro{:.4f}, eval_ks{:.4f}, eval_gm{:.4f}\n'\
                    .format(epoch,e_list[0],e_list[1],e_list[2],e_list[3])
            wait_epoch = 0
            if save_flag:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "savemodel/eval_acc{}.pth".\
                    format(e_list[0])
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch > waits:
            print(
                "{}epochs finished, best_epoch:{}, saved_model_result:{}".format(epoch, best_epoch, eval_best_str))

            break

        epoch += 1

    if epoch == MAX_EPOCH:
        print("{}epochs finished, saved_model_result:{}".format(MAX_EPOCH,eval_str))
        print('lowest loss epoch:{}, the result:{}'.format(best_epoch, eval_best_str))
        best_model_file = "savemodel/eval_acc{}.pth".format(e_list[0])
        torch.save(model.state_dict(), best_model_file)




