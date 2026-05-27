
#!/bin/bash

mkdir -p logs

python3 main.py --model zero-masking --dataset ETTh1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 96 > logs/zm_etth1_96.log 2>&1
python3 main.py --model zero-masking --dataset ETTh2 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 96 > logs/zm_etth2_96.log 2>&1 
python3 main.py --model zero-masking --dataset ETTm1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 96 > logs/zm_ettm1_96.log 2>&1 
python3 main.py --model closed-loop --dataset ETTh1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 96 > logs/cl_etth1_96.log 2>&1 
python3 main.py --model closed-loop --dataset ETTh2 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 96 > logs/cl_etth2_96.log 2>&1 
python3 main.py --model closed-loop --dataset ETTm1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 96 > logs/cl_ettm1_96.log 2>&1
python3 main.py --model SpaceTime --dataset ETTh1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 96 > logs/st_etth1_96.log 2>&1 
python3 main.py --model SpaceTime --dataset ETTh2 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 96 > logs/st_etth2_96.log 2>&1 
python3 main.py --model SpaceTime --dataset ETTm1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 96 > logs/st_ettm1_96.log 2>&1

wait