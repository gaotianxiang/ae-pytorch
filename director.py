import torch
import os
from modules import AutoEncoder, CIFAR10DL
from modules import AELoss
from tqdm import tqdm, trange
from utils import RunningAverage
import torch.optim as optim


class Director:
    def __init__(self, gpu, batch_size, num_workers, hidden_size):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dl_train = CIFAR10DL(train_set=True, batch_size=batch_size, num_workers=num_workers)
        self.dl_test = CIFAR10DL(train_set=False, batch_size=batch_size, num_workers=num_workers)
        self.net = AutoEncoder(hidden_size=hidden_size).to(self.device)
        self.loss = AELoss()

    def train(self, epochs, lr, log_every):
        ravg = RunningAverage()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        for epoch in trange(epochs, desc='epochs'):
            ravg.reset()
            ite = 0
            with tqdm(total=len(self.dl_train())) as progress_bar:
                for imgs, _ in self.dl_train():
                    imgs = imgs.to(self.device)
                    optimizer.zero_grad()
                    reconstructions = self.net(imgs)
                    loss = self.loss(reconstructions, imgs)
                    loss.backward()
                    optimizer.step()
                    ite += 1
                    ravg.update(loss.item())
                    if ite % log_every == 0:
                        tqdm.write(
                            'epochs {} iterations {}, average reconstruction loss {:.4f}'.format(epoch, ite, ravg()))
                    progress_bar.set_postfix(loss_avg=ravg())
                    progress_bar.update()
            self.evaluate()

    def evaluate(self):
        ravg = RunningAverage()
        self.net.eval()
        with torch.no_grad():
            with tqdm(total=len(self.dl_test())) as progress_bar:
                for imgs, _ in self.dl_test():
                    imgs = imgs.to(self.device)
                    reconstructions = self.net(imgs)
                    loss = self.loss(reconstructions, imgs)
                    ravg.update(loss.item())
                    progress_bar.set_postfix(test_avg_loss=ravg())
                    progress_bar.update()
        tqdm.write('test reconstruction loss {:.4f}'.format(ravg()))
