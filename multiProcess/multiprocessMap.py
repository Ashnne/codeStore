import os
import torch
import numpy as np
from multiprocessing import Process, Pool


def worker(i,tmp_path):
    print('index: {}'.format(i))
    torch.save(i,tmp_path.format(i))
    return None

def main():

    pool_num=32
    # clip_dirs = np.array_split(np.array(clip_dirs),pool_num)
    root_dir = 'tmp'
    tmp_dir = root_dir + '/{}.torch'
    os.makedirs(root_dir,exist_ok=True)

    inputs=[]
    for i in range(pool_num):
        inputs.append([i,tmp_dir])

    if os.path.exists(tmp_dir.format('all')):
        print('data has already divided!')
    else:
        dicts = {}
        keys = list(dicts.keys()) # todo:
        gap_num = len(keys) // pool_num
        for i in range(pool_num):
            if os.path.exists(tmp_dir.format(i)):
                continue
            tmp_keys = keys[i*gap_num:(i+1)*gap_num]
            tmp_dicts = {key:dicts[key] for key in tmp_keys}
            torch.save(tmp_dicts,tmp_dir.format(i))
        # divide all symbol
        torch.save('',tmp_dir.format('all'))
        print('divide all!')

    
    pool = Pool(pool_num)
    pool.starmap(worker, inputs)
    print('conceal all!')

    result_all = []
    for i in range(pool_num):
        tmp_path = tmp_dir.format(str(i)+'_record')
        tmp = torch.load(tmp_path,weights_only=False)
        result_all = result_all + tmp
    print('load result all!')

    # os.system('rm -rf tmp')
    record_path = '/public/home/group_yangych/qyzheng/data/result_top10plus/merge/record.{}'
    with open(record_path.format('txt'),'w') as f:
        f.writelines(result_all)
        f.close()
    torch.save(result_all,record_path.format('torch'))

    print('total is {} '.format(len(result_all)))

if __name__ == "__main__":
    main()