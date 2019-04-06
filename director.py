import torch
import os
from modules import AutoEncoder, CIFAR10DL
from modules import AELoss
from tqdm import tqdm, trange
from utils import RunningAverage
import torch.optim as optim
import torchvision


class Director:
    def __init__(self, gpu, batch_size, num_workers, hidden_size, model_dir):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dl_train = CIFAR10DL(train_set=True, batch_size=batch_size, num_workers=num_workers)
        self.dl_test = CIFAR10DL(train_set=False, batch_size=batch_size, num_workers=num_workers)
        self.net = AutoEncoder(hidden_size=hidden_size).to(self.device)
        self.loss = AELoss()
        self.model_dir = model_dir
        self.ckpts_path = os.path.join(model_dir, 'ckpts')
        self.recon_path = os.path.join(model_dir, 'reconstructions')
        self.current_best_test_loss = 1e5
        self.start_epoch = 0
        os.makedirs(self.ckpts_path, exist_ok=True)
        os.makedirs(self.recon_path, exist_ok=True)

    def train(self, epochs, lr, log_every):
        ravg = RunningAverage()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        for epoch in trange(self.start_epoch, self.start_epoch + epochs, desc='epochs'):
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
            test_loss = self.evaluate()
            self.store_ckpts(test_loss, epoch)
            self.visualize(epoch)

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
        return ravg()

    def visualize(self, epoch):
        self.net.eval()
        for imgs, _ in self.dl_test():
            break
        imgs = imgs.to(self.device)
        with torch.no_grad():
            recon = self.net(imgs)
        whole = torch.cat((imgs, recon), dim=0)
        grid = torchvision.utils.make_grid(whole, nrow=16, padding=2, pad_value=255)
        torchvision.utils.save_image(grid, filename=os.path.join(self.recon_path,
                                                                 'cifar10_reconstruction_epoch_{}.png'.format(epoch)))

    def store_ckpts(self, loss, epoch):
        if loss > self.current_best_test_loss:
            return
        self.current_best_test_loss = loss
        state_dict = {
            'net': self.net.state_dict(),
            'epoch': epoch,
            'test_loss': loss
        }
        torch.save(state_dict, os.path.join(self.ckpts_path, 'best.pth.tar'))
        tqdm.write('new best loss is found and ckpt is saved')

    def load_ckpts(self):
        ckpts_path = os.path.join(self.ckpts_path, 'best.pth.tar')
        assert os.path.exists(ckpts_path), 'there is no ckpt file in {}'.format(self.ckpts_path)
        state_dict = torch.load(ckpts_path)
        self.net.load_state_dict(state_dict['net'])
        self.start_epoch = state_dict['epoch'] + 1
        print('ckpt after {} epochs is loaded, the test loss is {:.4f}'.format(state_dict['epoch'],
                                                                               state_dict['test_loss']))
