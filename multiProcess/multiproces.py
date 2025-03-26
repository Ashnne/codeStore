from multiprocessing import Pool
import time

import torch.nn


def f(x,y):
    print(x,y)
    time.sleep(10-x)


def main():
    
    pool = Pool(processes=3)
    for i in range(10):
        pool.apply_async(f,args=(i,i-1,))
    
    print('all in pool')
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
