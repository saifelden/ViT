import torch
import torch.nn as nn
from torch.nn import Conv2d
from einops.layers.torch import Rearrange
from torchvision import datasets,transforms
from torch.optim import Adam as adam

class Pos_Embedding(nn.Module):

    def __init__(self,patch_size,num_of_patches):
        super().__init__()
        self.pos_encoding = torch.randn(1,num_of_patches*num_of_patches)
        self.embedding = Conv2d(in_channels=3, out_channels=1, kernel_size=(patch_size,patch_size), stride=(patch_size,patch_size))
    def forward(self,input):
        embeds = self.embedding(input)
        flt_layer = Rearrange('b c h w -> b (c h w)')
        flattened = flt_layer(embeds)
        encoded = flattened+self.pos_encoding
        return encoded


class Attention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.key_linear = nn.Linear(hidden_size,hidden_size)
        self.value_linear = nn.Linear(hidden_size,hidden_size)
        self.query_linear = nn.Linear(hidden_size,hidden_size)
        self.hidden_size= hidden_size

    def forward(self,inputs):
        keys = self.key_linear(inputs)
        query = self.query_linear(inputs)
        value = self.value_linear(inputs)
        weights = nn.Softmax(dim=1)(torch.matmul(query,keys.transpose(-2,-1))/torch.sqrt(torch.tensor([self.hidden_size*self.hidden_size])))
        attention  = torch.matmul(weights,value)
        return attention

class MLP(nn.Module):
    def __init__(self,hidden_size,out_size,dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size,out_size)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(out_size,out_size)
        self.activation2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self,inputs):

        out1 = self.fc1(inputs)
        act1 = self.activation1(out1)
        out2 = self.fc2(act1)
        act2 = self.activation2(out2)
        return self.dropout(act2)
    
class Encoder(nn.Module):
    def __init__(self,hidden_size,out_size,pitch_size,num_blocks=6,dropout_rate=0.8):
        super().__init__()
        self.norm_layers=[]
        self.attentions = []
        self.mlps = []
        self.num_of_blocks = num_blocks
        for i in range(num_blocks):
            norm1 = nn.LayerNorm(pitch_size*pitch_size,1e-6)
            norm2 = nn.LayerNorm(pitch_size*pitch_size,1e-6)
            self.norm_layers.append((norm1,norm2))
            self.attentions.append(Attention(pitch_size*pitch_size))
            self.mlps.append(MLP(pitch_size*pitch_size,pitch_size*pitch_size,dropout_rate))
        self.final_mlp = MLP(pitch_size*pitch_size,out_size,dropout_rate)
        self.pos_emb = Pos_Embedding(hidden_size,pitch_size)
        

    def forward(self,inputs):
        inputs = self.pos_emb(inputs)
        for i in range(self.num_of_blocks):
            norms1 = self.norm_layers[i][0](inputs)
            attention_out = self.attentions[i](norms1)
            sum1 = attention_out+inputs
            norm2 = self.norm_layers[i][0](sum1)
            mlp = self.mlps[i](norm2)
            sum2 = mlp+sum1
            inputs = sum2

        pred_output =  nn.Softmax()(self.final_mlp(sum2))
        return pred_output
        

from  torchvision.datasets import CIFAR10
train = datasets.CIFAR10('',train=True,download=True,transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.CIFAR10('',train=False,download=True,transform = transforms.Compose([transforms.ToTensor()]))
kernel_size = 4
output_size = 10
pitch_size = int(32/kernel_size)
train_set = torch.utils.data.DataLoader(train,batch_size=32,shuffle=True)
test_set = torch.utils.data.DataLoader(test,batch_size = 32,shuffle=False)
encoder = Encoder(kernel_size,output_size,pitch_size)
optimizer = adam(encoder.parameters(),lr=0.001)
loss = nn.CrossEntropyLoss()
Epochs = 100
for epoch in range(Epochs):
    for data in train_set:
        X,y = data
        encoder.zero_grad()
        pred_output = encoder(X.view(-1,3,32,32))
        loss_output = loss(pred_output,y)
        loss_output.backward()
        optimizer.step()
    print(loss_output)






