import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import time
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from midi_manipulation import midiToNoteStateMatrix, noteStateMatrixToMidi
from vae7 import VAE
from songsDataset import SongsDataset

songs = SongsDataset()

dataloader = DataLoader(songs, batch_size=128, shuffle=True, num_workers=4)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

start = time.time()

losses = []
val_losses = []

# val_songs = Variable(torch.from_numpy(val_songs))

counter = 0

l1 = []
l2 = []
l3 = []

klc = 0.2
sumc = 0.25

val1 = []

checkpoint = torch.load('ALLEVEYRTHINGNEW10')

model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


for it in range(10):
    print(it, end='  ! ')
    plt.plot(losses)
    plt.savefig('plot_losses')

    if it%1 == 0:
        samples = torch.randn(8, 300).to(device)
        samples = model.decode(X=samples, z=samples, sample_new=True).cpu()
        samples = samples.detach().numpy()

        samples[samples <= 0.5] = 0
        samples[samples > 0.5] = 1

        for i, sample in enumerate(samples):
            sample = sample.reshape(-1, 156)
            noteStateMatrixToMidi(sample, name=("samples/"+ str(it) + "new_sample_" + str(i)))

    if it > 2 and it%4 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8

    for ii, X in enumerate(dataloader):
        # X = Variable(torch.from_numpy(X))
        optimizer.zero_grad()

        X = X.type(torch.FloatTensor)
        X = X.to(device)

        # Forward
        X_sample, z_mu, z_var = model(X)

        # Loss
        try:
            recon_loss = F.binary_cross_entropy(X_sample, X, size_average=False)
        except:
            print(max(X_sample.flatten()))

        kl_loss = klc * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
        loss = recon_loss + kl_loss - sumc*X_sample.sum()

        if ii == 10:
            losses.append(float(loss.data))
            
            l1.append(float(recon_loss.data))
            l2.append(float(kl_loss.data))
            l3.append(float(X_sample.sum()))

        # Backward
        loss.backward()
        # and update

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # if it % 5 == 0 and ii == 0:
        #     model.eval()
        #     X = val_songs.type(torch.FloatTensor)
        #     X = X.to(device)

        #     X_sample, z_mu, z_var = model(X)

        #     recon_loss = F.binary_cross_entropy(X_sample, X, size_average=True)
        #     kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
        #     loss = recon_loss + kl_loss

        #     val_losses.append(loss.data)
        #     val1.append(recon_loss.data)
        #     model.train()
    if it % 2 == 0:
        torch.save({
            'epoch': it,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "NEWALLEVEYRTHINGNEW" + str(it))


end = time.time()
print("time took: " + str(end - start))

# Graph losses
plt.plot(losses)
plt.show()


# Draw some samples
model.eval()

samples = torch.randn(10, 300).to(device)
samples = model.decode(X=samples, z=samples, sample_new=True).cpu()
samples = samples.detach().numpy()

samples[samples <= 0.8] = 0
samples[samples > 0.8] = 1

for i, sample in enumerate(samples):
    sample = sample.reshape(-1, 156)
    noteStateMatrixToMidi(sample, name=("samples2/sample_0.8_" + str(i)))

