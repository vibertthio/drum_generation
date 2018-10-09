import os
from flask import Flask, request, Response
import numpy as np
import json
import pypianoroll
from pypianoroll import Multitrack, Track
import torch.utils.data as Data
from matplotlib import pyplot as plt
from vae_rnn import *


app = Flask(__name__)
app.config['ENV'] = 'development'

'''
laod model
'''
path = os.getcwd() + '/../models/'
model = [ m for m in os.listdir(path) if '.pt' in m ][0]
encoder = Encoder().to(device)
decoder = Decoder().to(device)
vae = VAE(encoder, decoder).to(device)
vae.load_state_dict(torch.load(path + model))

'''
load data
'''
genres = [x for x in os.listdir('./../../explore_data/data/') if '.npy' in x]
train_x_np = np.load('./../../explore_data/data/' + genres[2])
train_x = torch.from_numpy(train_x_np).type(torch.FloatTensor)
train_dataset = Data.TensorDataset(train_x)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=1,
)

'''
show drum roll in console
'''
def printDrumRoll(roll):
    trans = np.flip(np.transpose(roll), 0)
    for r_i, r in enumerate(trans):
        print('[{}]'.format(8 - r_i), end='')
        for i, w in enumerate(r):
            if i > 0 and i % 16 == 0:
                print('|', end='')
            if w == 0:
                print('_', end='')
            else:
                print('*', end='')
        print()



@app.route('/api/rand', methods=['POST', 'GET'])
def rand():
    with torch.no_grad():
        data = iter(train_loader).next()[0]
        # pltReducedDrumTrack(data[0])

        data = Variable(data).type(torch.float32).to(device)
        latent = vae._enc_mu(encoder(data))
        out = decoder(latent)
        out_np = out.cpu().data.numpy()
        out_np = np.where(out_np > 0.2, 128, 0)
        out_concat = np.zeros((9, 96, 9))
        for i in range(9):
            x = i % 3 - 1
            y = i // 3 - 1

            latent_shift = latent.cpu().data.numpy()
            shift = np.zeros(latent_shift.shape, dtype=np.float32)
            shift[0][2] = x * 0.2
            shift[0][3] = y * 0.2
            latent_shift = latent_shift + shift

            latent_shift = torch.from_numpy(latent_shift).to(device)
            o = decoder(latent_shift)

            o = o[0].cpu().data.numpy()
            o = np.where(o > 0.5, 1, 0)
            out_concat[i] = o
    
    out_concat = out_concat.tolist()
    response = {
        'result': out_concat
    }
    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)
