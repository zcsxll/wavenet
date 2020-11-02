import torch
import numpy as np
from scipy.io import wavfile

import dataset as d
import model as m
import save_load as sl

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

def generate():
    model = m.WaveNet(n_layers=10*3, hidden_channels=32)
    model = sl.load_model('./checkpoint', 10000, model)
    model = model.cuda()

    # samplerate, pcm = wavfile.read('./music/wav/jacob_heringman-blame_not_my_lute-03-the_bagpipes-0-29.wav')
    samplerate, pcm = wavfile.read('./music/wav/jacob_heringman-blame_not_my_lute-12-robin_hoode-0-29.wav')
    pcm = pcm[600*16:600*16+model.receptive_field]
    pcm = d.mu_law_encode(pcm)
    generated = torch.from_numpy(pcm).type(torch.long).cuda()
    
    for i in range(16000 * 3):
        input = torch.FloatTensor(1, 256, model.receptive_field).zero_().cuda()
        input.scatter_(1, generated[-model.receptive_field:].view(1, 1, model.receptive_field), 1.0)
        # print(input[0, :, -1].shape, input[0, :, -1].sum(0))
        pred = model(input)
        value = pred.argmax(dim=1)
        generated = torch.cat((generated, value), 0)
        print(i, value, generated.shape)
    generated = generated.detach().cpu().numpy()
    generated = (generated / 256) * 2 - 1
    mu_gen = mu_law_expansion(generated, 256)
    print(generated.shape, mu_gen.shape)
    # print(mu_gen[3000:3000+100])
    wavfile.write('./out.wav', 16000, (mu_gen * 30000).astype(np.int16))

if __name__ == '__main__':
    generate()
