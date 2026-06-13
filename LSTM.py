import torch as t
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import wandb
wandb.login()

class pylstm(t.nn.Module):
    def __init__(self,chars,hiddensize=250,dropprob=0.2,embedsize=10,nlayers=2):
        super().__init__()
        self.hiddensize=hiddensize
        self.dropprob=dropprob
        self.embedsize=embedsize
        self.nlayers=nlayers
        self.embedding=t.nn.Embedding(len(chars), embedsize)
        self.lstm=t.nn.LSTM(input_size=embedsize, hidden_size=hiddensize, num_layers=nlayers, bias=True, batch_first=True, dropout=dropprob)
        self.h0=None
        self.c0=None
        self.dropout=t.nn.Dropout(p=dropprob, inplace=False)
        self.linear=t.nn.Linear(in_features=hiddensize, out_features=len(chars), bias=True)
    def reset_hidden(self, batch_size):
        weight = next(self.parameters())
        self.h0 = weight.new_zeros(self.nlayers, batch_size,self.hiddensize )
        self.c0 = weight.new_zeros(self.nlayers, batch_size,self.hiddensize)
    def forward(self,X):
        h_state = self.h0.detach()
        c_state = self.c0.detach()
        embedded_X = self.embedding(X)
        out,(hc,cn)=self.lstm(embedded_X,(h_state,c_state))
        out=self.dropout(out)
        out=self.linear(out)
        self.h0=hc
        self.c0=cn
        return out
    
def train(lstm,Xtr,ytr,no_epochs=24,basize=32,senlen=50,clip=5,lr=0.001,every=True):
    wandb.init(
        project="lstm-project",
        config={
            "learning_rate": lr,
            "epochs": no_epochs,
            "batch_size": basize,
            "sequence_length": senlen,
            "grad_clip": clip,
            "architecture": "Stateful LSTM"
        }
    )
    lstm.train()
    lstm.cuda()
    X=Xtr.view(-1,senlen)
    y=ytr.view(-1,senlen)
    X,y=X.cuda(),y.cuda()
    dataset=TensorDataset(X,y)
    loader = DataLoader(dataset, batch_size=basize, shuffle=False,drop_last=True)
    lossfxn = t.nn.CrossEntropyLoss()
    opt = t.optim.Adam(lstm.parameters(), lr=lr)
    for _ in range(no_epochs):
        lstm.reset_hidden(basize)
        epochloss=0
        counter=0
        for (x,y) in loader:
            counter+=1
            opt.zero_grad()
            out=lstm(x)
            loss=lossfxn(out.view(-1,out.size(-1)),y.view(-1))
            epochloss+=loss.item()
            loss.backward()
            t.nn.utils.clip_grad_norm_(lstm.parameters(), clip)
            opt.step()
        epochloss/=counter
        wandb.log({"train_loss": epochloss, "epoch": _})
        if every:
            print(epochloss)
    wandb.finish()
    

def sample(lstm,atoi,itoa,start_char='.',senlen=50,tk=5):
    g = t.Generator().manual_seed(2147483647)
    inp=atoi[start_char]
    lstm.eval()
    lstm.cpu()
    lstm.reset_hidden(1)
    ans=[]
    for i in range(senlen):
        inp=t.tensor(inp).view(1,1)
        out=lstm(inp)
        out=F.softmax(out.view(-1,out.size(-1)),dim=1).data
        #out,_=out.topk(tk)
        ix=t.multinomial(out, num_samples=1, generator=g).item()
        inp=ix
        ans.append(itoa[inp])
    ans=''.join(ans)
    print(ans)
    return ans