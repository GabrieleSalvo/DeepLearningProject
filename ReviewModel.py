import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle

# todo reset parameters for linear layer
class ReviewModel(nn.Module):
    def __init__(self,paramDict):
        super().__init__()
        weights_matrix = np.load("weightmatrix.npy")
        self.max_length = paramDict["max_length"]
        self.batch_size = paramDict["batch_size"]
        self.embed_size = paramDict["embedding_dim"]
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(weights_matrix), freeze=False)
        
        self.drop1 = nn.Dropout(p=paramDict["first_dropout"])
        self.conv1 = nn.Conv1d(in_channels = paramDict["conv_input_channel"],
                               out_channels = paramDict["conv_output_channel"],
                               kernel_size = paramDict["conv1_kernel_size"],
                               padding = paramDict["conv_padding"]).double()#
        self.conv2 = nn.Conv1d(in_channels=paramDict["conv_input_channel"],
                               out_channels=paramDict["conv_output_channel"],
                               kernel_size=paramDict["conv2_kernel_size"],
                               padding=paramDict["conv_padding"]).double()#    
        self.maxpool = nn.MaxPool1d(kernel_size=paramDict["maxpool_kernel_size"])
        self.drop2 = nn.Dropout(p=paramDict["second_dropout"])
        
        self.rnn = nn.GRU(input_size=paramDict["rnn_input_size"],
                          hidden_size=paramDict["rnn_hidden_size"],
                          num_layers=paramDict["rnn_num_layers"], 
                          batch_first=False) 
        self.flatten = Flatten()
        
        self.fc1 =nn.Linear(in_features=paramDict["first_dense_in"], 
                            out_features = paramDict["first_dense_out"])
        self.drop3 = nn.Dropout(p=paramDict["third_dropout"])
        self.fc2 = nn.Linear(in_features=paramDict["second_dense_in"], 
                             out_features=paramDict["second_dense_out"] )
        
    def forward(self, x):
        # x_size = (batch_size, max_seq_len)
        
        x = self.embedding(x)
        # x_size = (batch_size, max_seq_len, embed_size)
        
        x = self.drop1(x)
        
        x = x.view(-1,self.embed_size,self.max_length)  
        # x_size = (batch_size,embed_size,max_seq_len) 
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        # x1_size = (batch_size, 200, max_seq_len+1)
        # x2_size = (batch_size, 200, max_seq_len)

        x1 = self.maxpool(x1)
        x2 = self.maxpool(x2)
        # x1_size = x2_size = (batch_size,200, max_seq_len//2)

        x = torch.cat((x1,x2), 1)
        # x_size = (batch_size, 400, max_seq_len//2)

        x = self.drop2(x)
        
        x = x.view(self.max_length//2,self.batch_size, -1).float()
        # x_size = (max_seq_len//2, batch_size, 400)

        hidden = Variable(torch.cuda.FloatTensor(1, self.batch_size, 100).uniform_()) 
        output, _ = self.rnn(x,hidden)
        # output_size = (max_seq_len//2, batch_size, hidden_size)

        x = output.contiguous().view(self.batch_size,-1)
        # x_size = (batch_size, max_seq_len//2 *hidden_size)

        x = F.relu(self.fc1(x))
        # x_size = (batch_size, 400)
        
        x = self.drop3(x)
        
        x = self.fc2(x)
        # x_size = (batch_size, out_dim)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x    