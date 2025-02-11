import argparse
from config import Config
from easydict import EasyDict as edict

def main():
    args = parser.parse_args()
    print(f'run the {args.num} task')
    print(f'use the device: CUDA:{args.cuda}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="slurm")

    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)

    main()
