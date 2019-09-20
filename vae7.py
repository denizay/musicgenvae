import torch
from torch import nn
import torch.nn.functional as F
import time

cont_len = 6
bar_len = 32
seq_len = 192
Z_dim = 100
X_dim = 156 * seq_len  
h_dim = 156

input_size = 156
num_layers = 2


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.switch = False

        self.cnt = 8
        self.modder = 20

        self.num_layers = 1
        self.hidden_size = 1024

        self.hid_1 = 512
        self.hid_2 = 512

        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=0.4)

        self.conduct1 = nn.LSTMCell(512, 512)
        self.conduct2 = nn.LSTMCell(512, 512)

        self.lstm1 = nn.LSTMCell(156, self.hid_1)
        self.lstm2 = nn.LSTMCell(self.hid_1, self.hid_2 )
        self.linear = nn.Linear(self.hid_2, 156)

        self.fc_mu = nn.Linear(self.hidden_size*2, 300)
        self.fc_var = nn.Linear(self.hidden_size*2, 300)

        self.lin_cont = nn.Linear(512, 512)

        self.lin = nn.Linear(300, 512)


    def encode(self, X):

        h0 = torch.zeros(self.num_layers*2, X.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, X.size(0), self.hidden_size).cuda()

        _, (_, final_state) = self.lstm(X, (h0, c0))
        final_state = final_state.view(-1, self.hidden_size*2)
        F.dropout(final_state, 0.3, training=True, inplace=True)
        


        z_mu = self.fc_mu(final_state)
        z_var = self.fc_var(final_state)

        return z_mu, z_var

    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, X=None, z=None, sample_new=False):
        outputs = []

        h_t = torch.zeros(X.size(0), self.hid_1).cuda()
        c_t = torch.zeros(X.size(0), self.hid_1).cuda()
        h_t2 = torch.zeros(X.size(0), self.hid_2).cuda()
        c_t2 = torch.zeros(X.size(0), self.hid_2).cuda()

        h_t_conduct = torch.zeros(X.size(0), self.hid_1).cuda()
        c_t_conduct  = torch.zeros(X.size(0), self.hid_1).cuda()
        h_t2_conduct  = torch.zeros(X.size(0), self.hid_2).cuda()
        c_t2_conduct  = torch.zeros(X.size(0), self.hid_2).cuda()

        z = z.squeeze(0)

        F.dropout(z, 0.4, training=True, inplace=True)
        z = self.lin(z)
        F.dropout(z, 0.4, training=True, inplace=True)

        start = torch.ones(X.size(0), 1, 156, dtype=torch.float).cuda()

        contexts = []

        for i in range(cont_len):
            input_t = torch.ones(X.size(0), 512, dtype=torch.float).cuda()
            if i == 0 :
                h_t_conduct, c_t_conduct = self.conduct2(input_t, (h_t_conduct, z))
                F.dropout(h_t_conduct, 0.2, training=True, inplace=True)
                h_t2_conduct, c_t2_conduct = self.conduct2(h_t_conduct, (h_t2_conduct, c_t2_conduct))
                F.dropout(h_t2_conduct, 0.1, training=True, inplace=True)
            else:
                h_t_conduct, c_t_conduct = self.conduct2(input_t, (h_t_conduct, c_t_conduct))
                F.dropout(h_t_conduct, 0.2, training=True, inplace=True)
                h_t2_conduct, c_t2_conduct = self.conduct2(h_t_conduct, (h_t2_conduct, c_t2_conduct))
                F.dropout(h_t2_conduct, 0.1, training=True, inplace=True)

            output = self.lin_cont(F.relu(h_t2)) # F.sigmoid(self.linear(h_t2)) 
            input_t = output

            contexts += [output]

        context=contexts[0]
        if sample_new:
            input_t = torch.ones(X.size(0), 156, dtype=torch.float).cuda()
            for i in range(192):
                if i % bar_len:
                    context = contexts[int(i/bar_len)]
                    c_t = context
                # input_t = input_t[:,0,:]
                if i  == 0 :
                    h_t, c_t = self.lstm1(input_t, (h_t, context))
                    F.dropout(h_t, 0.4, training=True, inplace=True)
                    h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                    F.dropout(h_t2, 0.2, training=True, inplace=True)
                else:
                    h_t, c_t = self.lstm1(input_t, (h_t, c_t))
                    F.dropout(h_t, 0.4, training=True, inplace=True)
                    h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                    F.dropout(h_t2, 0.2, training=True, inplace=True)

                output = F.sigmoid(self.linear(h_t2))
                input_t = output

                outputs += [output]
        else:
            X = torch.cat((start, X), 1)
            for i, input_t in enumerate(X.chunk(193, dim=1)[:-1]):
                if i % bar_len:
                    context = contexts[int(i/bar_len)]
                    c_t = context
                input_t = input_t[:,0,:]
                if i % self.cnt ==0 and i>0:
                    input_t = outputs[-1]
                if self.switch:
                    if i % 3 != 0:
                        input_t = outputs[-1]

                if i == 0 :
                    h_t, c_t = self.lstm1(input_t, (h_t, context))
                    F.dropout(h_t, 0.5, training=True, inplace=True)
                    h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                    F.dropout(h_t2, 0.2, training=True, inplace=True)
                else:
                    h_t, c_t = self.lstm1(input_t, (h_t, c_t))
                    F.dropout(h_t, 0.5, training=True, inplace=True)
                    h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                    F.dropout(h_t2, 0.2, training=True, inplace=True)

                output = F.sigmoid(self.linear(h_t2))
                outputs += [output]


        outputs = torch.cat(outputs,dim=1)
        return outputs

    def forward(self, x):
        x = x.view(-1, seq_len, 156)
        z_mu, z_var = self.encode(x)
        z = self.sample_z(z_mu, z_var).cuda()
        X_sample = self.decode(X=x, z=z)

        return X_sample, z_mu, z_var
