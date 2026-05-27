
#!/bin/bash

mkdir -p logs

python3 main.py --model zero-masking --dataset multi-sine --type complex --activation GELU --dataset_norm true --epochs 50 --state_dim 32 --hidden 16 --layers 3 --P 128 --F 48 > logs/zm_multi.log 2>&1 
python3 main.py --model closed-loop --dataset multi-sine --type complex --activation GELU --dataset_norm true --epochs 50 --state_dim 32 --hidden 16 --layers 3 --P 128 --F 48 > logs/cl_multi.log 2>&1 
python3 main.py --model SpaceTime --dataset multi-sine --type complex --activation GELU --dataset_norm true --epochs 50 --state_dim 32 --hidden 16 --layers 3 --P 128 --F 48 > logs/st_multi.log 2>&1 


wait