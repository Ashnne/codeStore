import argparse
from config import Config
from easydict import EasyDict as edict

def main():
    args = parser.parse_args()
    # C = Config(args.config)

    # # main branch
    # config = edict(C.config)
    # print(config)

    print(args.begin+args.length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain")

    parser.add_argument('--config', type=int, default='')
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--length', type=int, default=4)

    main()
