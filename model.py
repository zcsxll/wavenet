import torch

class WaveNet(torch.nn.Module):
    def __init__(self, n_layers, hidden_channels, skip_channels=1024):
        assert(n_layers % 10 == 0)
        self.n_layers = n_layers
        super(WaveNet, self).__init__()

        self.start_conv = torch.nn.Conv1d(in_channels=256, out_channels=hidden_channels, kernel_size=1, stride=1)

        self.filter_convs = torch.nn.ModuleList() #如果用python的list类型，则调用WaveNet().cuda()时是无效的
        self.gate_convs = torch.nn.ModuleList()
        self.skip_convs = torch.nn.ModuleList()
        self.transform_convs = torch.nn.ModuleList()
        self.receptive_field = 1 #这个值是输出一个节点需要输入的节点数量，等于所有层膨胀系数之和+1
        for i in range(self.n_layers):
            self.filter_convs.append(torch.nn.Conv1d(in_channels=hidden_channels,
                                        out_channels=hidden_channels,
                                        kernel_size=2,
                                        stride=1,
                                        dilation=2**(i%10), #膨胀系数从1到512，然后再从1到512
                                        bias=True))
            self.gate_convs.append(torch.nn.Conv1d(in_channels=hidden_channels,
                                        out_channels=hidden_channels,
                                        kernel_size=2,
                                        stride=1,
                                        dilation=2**(i%10),
                                        bias=True))

            self.skip_convs.append(torch.nn.Conv1d(in_channels=hidden_channels,
                                        out_channels=skip_channels,
                                        kernel_size=1,
                                        stride=1,
                                        bias=True))
            self.transform_convs.append(torch.nn.Conv1d(in_channels=hidden_channels,
                                        out_channels=hidden_channels,
                                        kernel_size=1,
                                        stride=1,
                                        bias=True))
            self.receptive_field += 2**(i%10)
        # print(self.receptive_field)

        self.end_conv_1 = torch.nn.Conv1d(in_channels=skip_channels,
                                        out_channels=512,
                                        kernel_size=1,
                                        stride=1,
                                        bias=True)
        self.end_conv_2 = torch.nn.Conv1d(in_channels=512,
                                        out_channels=256,
                                        kernel_size=1,
                                        stride=1,
                                        bias=True)
                

    def forward(self, x):
        x = self.start_conv(x)

        skip = None
        output_len = x.shape[-1] - self.receptive_field + 1
        # print(output_len)
        for i in range(self.n_layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.nn.Tanh()(filter)
            gate = self.gate_convs[i](x)
            gate = torch.nn.Sigmoid()(gate)
            x = filter * gate
            # print(x.shape)

            cur_skip = self.skip_convs[i](x[:, :, -output_len:])
            '''
            每次卷积都会减小长度，即每次skip的tensor都是变短的
            由于skip_conv的卷积核是1，可以不计算前边的数据
            '''
            if skip is None:
                skip = cur_skip
            else:
                skip = skip + cur_skip
                # print(skip.shape)

            x = self.transform_convs[i](x)
            x = x + residual[:, :, 2**(i%10):] #残差相加
            # print(x.shape, x.shape, residual.shape)
        # print(skip.shape)
        skip = torch.nn.ReLU()(skip)
        skip = self.end_conv_1(skip)
        skip = torch.nn.ReLU()(skip)
        skip = self.end_conv_2(skip)
        # print(skip.shape)

        n, c, l = skip.shape
        skip = skip.transpose(1, 2).contiguous()
        skip = skip.view(n*l, c)
        # print(n, c, l, skip.shape)
        return skip

if __name__ == '__main__':
    import dataset

    dataset = dataset.Dataset(3070, 16)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=4,
                                            shuffle=False)

    model = WaveNet(n_layers=10*3, hidden_channels=32)

    for step, (batch_x, batch_y) in enumerate(dataloader):
        pred = model(batch_x)
        print(step, batch_x.shape, batch_y.shape, pred.shape)
        # break