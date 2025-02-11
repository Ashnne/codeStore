index=$1
cuda=$((index/2))

CUDA_VISIBLE_DEVICES=${cuda} python3 main.py $index $cuda