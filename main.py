import argparse
from config import Config
from easydict import EasyDict as edict

def main():
    args = parser.parse_args()
    C = Config(args.config)
    config = edict(C.config)
    print(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain")

    parser.add_argument('--config', type=str, default='')

    main()
