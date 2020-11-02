import os
from scipy.io import wavfile
import numpy as np
import random
import torch

def mu_law_encode(audio):
    audio = audio / 32768.0
    mu = 256 - 1
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    encode = ((signal + 1) / 2 * mu + 0.5).astype(np.int32)
    return encode

class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_len, target_len): #input_len是模型的receptive length， target_len是模型的output length
        self.input_len = input_len
        self.target_len = target_len
        self.root_dir = './music/wav'
        self.waves = os.listdir(self.root_dir)
        assert len(self.waves) > 10

    def __getitem__(self, idx):
        wave_path = os.path.join(self.root_dir, self.waves[idx])
        samplerate, pcm = wavfile.read(wave_path)
        total_len = self.input_len + self.target_len
        '''
        训练时，对于输入的数据，不是把生成的数据追加到最后作为新的输入，而是直接使用真实的数据
        假设模型的感受野是7，我们想得到3个结果，在真正使用模型时，需要计算三次，因为后两次计算需要依赖前一次的输出
        但是在训练时，由于不需要，因为我们传入的是真实的已知的数据
        这里input_len是7，target_len是3，则需要传入10个数据，每个数据是一个256长度的onehot数据
        但是我们如果传入长度是10，则模型会计算出4个结果，分别是[0, 6]结果、[1, 7]的结果、[2, 8]的结果和[3, 9]的结果
        因此我们未给模型的是这10个数据的前9个，而和模型输出做loss的是后3个
        即[0, 6]结果和7做loss
        [1, 7]结果和8做loss
        [2, 8]结果和9做loss
        因此下边定义one_hot的时候，长度是total_len-1
        '''
        pcm = pcm[600*16:-3000*16] #掐头去尾，去除静音
        pcm = self.auto_gain(pcm)
        # print(type(pcm[0]))
        # wavfile.write('./test.wav', 16000, pcm.astype(np.int16))
        assert pcm.shape[0] >= total_len
        max_start = pcm.shape[0] - total_len
        offset = random.randint(0, max_start)
        pcm = pcm[offset:offset+total_len]
        encode = mu_law_encode(pcm)

        encode = torch.from_numpy(encode).type(torch.LongTensor)
        one_hot = torch.FloatTensor(256, total_len-1).zero_()
        one_hot.scatter_(0, encode[:total_len-1].unsqueeze(0), 1.0)
        # print(one_hot.shape)
        # print(one_hot[:, 0][100:])
        # print(encode[0])
        # print(pcm[10000:10000+100])
        target = encode[-self.target_len:]

        return one_hot, target

    def __len__(self):
        return len(self.waves)

    def auto_gain(self, pcm):
        max_point = np.max(np.abs(pcm))
        factor = 30000 / max_point
        return pcm * factor

if __name__ == '__main__':
    dataset = Dataset(3000, 16)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=None)

    for step, (batch_x, batch_y) in enumerate(dataloader):
        print(step, batch_x.shape, batch_y.shape)
        break