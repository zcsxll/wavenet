import torch
import dataset as d
import model as m
import save_load as sl

def train(n_epoch):
    dataset = d.Dataset(3070, 16)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=4,
                                            shuffle=True)
    model = m.WaveNet(n_layers=10*3, hidden_channels=32)
    model = model.cuda()
    model.train()

    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=0.001)

    try:
        trained_epoch = sl.find_last_checkpoint('./checkpoint')
        print('train form epoch %d' % (trained_epoch + 1)) 
    except Exception as e:
        print('train from the very begining, {}'.format(e))
        trained_epoch = -1
    model = sl.load_model('./checkpoint', -1, model)
    optim = sl.load_optimizer('./checkpoint', -1, optim)
    for epoch in range(trained_epoch+1, n_epoch):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            pred = model(batch_x)
            # print(pred.shape, batch_y.shape)
            loss = loss_func(pred, batch_y.view(pred.shape[0]))
            print(epoch, step, loss.detach().cpu().numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()
        sl.save_checkpoint('./checkpoint', epoch, model, optim)

if __name__ == '__main__':
    train(100)