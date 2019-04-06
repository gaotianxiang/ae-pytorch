import argparse
from director import Director
from utils import Params
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--model_dir', default='./experiments/base_model', type=str)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

hps_path = os.path.join(args.model_dir, 'config.json')
assert os.path.exists(hps_path), 'there is no config json file'
hps = Params(hps_path)
args.__dict__.update(hps.dict)


def train():
    train_model = Director(batch_size=args.batch_size, gpu=args.gpu, num_workers=args.num_workers,
                           hidden_size=args.hidden_size, model_dir=args.model_dir)
    if args.resume:
        train_model.load_ckpts()
    train_model.train(epochs=args.num_epochs, lr=args.lr, log_every=args.log_every)


def test():
    test_model = Director(batch_size=args.batch_size, gpu=args.gpu, num_workers=args.num_workers,
                          hidden_size=args.hidden_size, model_dir=args.model_dir)
    epoch, _ = test_model.load_ckpts()
    test_model.evaluate()
    test_model.visualize(epoch, 'test')


if __name__ == '__main__':
    if args.test:
        test()
    else:
        train()
