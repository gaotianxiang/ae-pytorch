import argparse
from director import Director
from utils import Params
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--model_dir', default='./experiments/base_model', type=str)
args = parser.parse_args()


def main():
    hps_path = os.path.join(args.model_dir, 'config.json')
    assert os.path.exists(hps_path), 'there is no config json file'
    hps = Params(hps_path)
    args.__dict__.update(hps.dict)
    director = Director(batch_size=args.batch_size, gpu=args.gpu, num_workers=args.num_workers,
                        hidden_size=args.hidden_size, model_dir=args.model_dir)
    director.train(epochs=args.num_epochs, lr=args.lr, log_every=args.log_every)


if __name__ == '__main__':
    main()
