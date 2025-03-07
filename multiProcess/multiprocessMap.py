import os
import torch
import numpy as np
from multiprocessing import Process, Pool


# count_all=0

def worker(i,tmp_path):
    print('index: {}'.format(i))
    torch.save(i,tmp_path.format(i))
    return None

def main():

    pool_num=32
    # clip_dirs = np.array_split(np.array(clip_dirs),pool_num)
    tmp_dir = 'tmp/{}.torch'
    os.system('rm -rf tmp')
    os.makedirs(tmp_dir,exist_ok=True)

    inputs=[]
    for i in range(pool_num):
        inputs.append([i,tmp_dir])
    
    pool = Pool(pool_num)
    pool.starmap(worker, inputs)

    for i in range(pool_num):
        tmp_path = tmp_dir.format(i)
        tmp = torch.load(tmp_path,weights_only=False)
        count_all = count_all + tmp

    os.system('rm -rf tmp')

    print('total is {} '.format(count_all))

if __name__ == "__main__":
    main()