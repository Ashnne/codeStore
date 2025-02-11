# -*- coding: UTF-8 -*-
import os
import time
import multiprocessing
import itertools
import sys
from VNSSA import VNSSA
from insert import insert
from Data import Data

def complex_func(files, T0, cool_rate, init_capacity, N_weight, iter_num):

    data = Data()
    data.readSolomonData(files)
    # print('当前文件:',files)
    mode = 1
    start = time.time()
    oldcapacity = data.capacity
    data.capacity = int(data.capacity * init_capacity)
    p = insert(data)
    data.capacity = oldcapacity
    best_p = VNSSA(p, mode, data, T0, cool_rate, N_weight)

    end = time.time()
    obj = best_p.cal_totalObj(mode, data)
    run_time = end-start

    vehicle_num = len(best_p.routes)
    return [files, obj, vehicle_num, run_time, T0, cool_rate, init_capacity, \
        N_weight, iter_num, str(best_p.routes)]


def main(n_jobs=10):

    file_path = 'solomon_dataset/'
    file_list = ['r108.txt','r109.txt','r110.txt','r111.txt']
    # file_list = ['r201.txt','r202.txt','r203.txt','r204.txt']
    # file_list = ['r206.txt','r207.txt','r208.txt','r209.txt',\
    #     'r210.txt','r211.txt']
    T0 = [10, 50, 100, 150]
    cool_rate = [0.997, 0.9997, 0.99975]
    init_capacity = [0.3, 0.5, 0.7, 1]
    N_weight = [250, 450]
    iter_num = [i for i in range(3)]

    # test cases
    # file_list = ['r205.txt', 'r206.txt']
    # T0 = [10]
    # cool_rate = [0.997, 0.9997]
    # init_capacity = [0.3, 1]
    # N_weight = [250, 450]
    # iter_num = [i for i in range(1)]

    files = []
    for file in file_list:
        files.append(file_path + file)

    params = list(itertools.product(files, T0, cool_rate, init_capacity, \
                                    N_weight, iter_num))
    for param in params:
        print(param)
    print('num of tasks: {}'.format(len(params)))

    print('Creating pool with %d processes\n' % n_jobs)
    with multiprocessing.Pool(n_jobs) as pool:
        results = [pool.apply_async(complex_func, param) for param in params]
        for r in results:
            while 1:
                sys.stdout.flush()
                try:
                    sys.stdout.write('\n%s' % r.get())
                    break
                except multiprocessing.TimeoutError:
                    sys.stdout.write('.')


if __name__ == "__main__":
    n_jobs = int(os.getenv('SLURM_CPUS_PER_TASK'))
    # start_time = time.time()
    main(n_jobs=n_jobs)
    # end_time = time.time()
    # print(n_jobs, end_time - start_time)
