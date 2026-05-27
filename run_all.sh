
#!/bin/bash

mkdir -p logs

python3 main.py --model zero-masking --dataset sine --type real --activation None --dataset_norm true --epochs 50 --state_dim 8 --hidden 8 --layers 3 --P 128 --F 48 > logs/zm_sine_real.log 2>&1 
python3 main.py --model zero-masking --dataset sine --type complex --activation None --dataset_norm true --epochs 50 --state_dim 6 --hidden 8 --layers 3 --P 128 --F 48 > logs/zm_sine_complex.log 2>&1 
python3 main.py --model closed-loop --dataset sine --type real --activation None --dataset_norm true --epochs 50 --state_dim 8 --hidden 8 --layers 3 --P 128 --F 48 > logs/cl_sine_real.log 2>&1 
python3 main.py --model closed-loop --dataset sine --type complex --activation None --dataset_norm true --epochs 50 --state_dim 6 --hidden 8 --layers 3 --P 128 --F 48 > logs/cl_sine_complex.log 2>&1 
python3 main.py --model SpaceTime --dataset sine --type real --activation None --dataset_norm true --epochs 50 --state_dim 8 --hidden 8 --layers 3 --P 128 --F 48 > logs/st_sine_real.log 2>&1 
python3 main.py --model SpaceTime --dataset sine --type complex --activation None --dataset_norm true --epochs 50 --state_dim 6 --hidden 8 --layers 3 --P 128 --F 48 > logs/st_sine_complex.log 2>&1 


python3 main.py --model zero-masking --dataset multi-sine --type complex --activation GELU --dataset_norm true --epochs 50 --state_dim 32 --hidden 16 --layers 3 --P 128 --F 48 > logs/zm_multi.log 2>&1 
python3 main.py --model closed-loop --dataset multi-sine --type complex --activation GELU --dataset_norm true --epochs 50 --state_dim 32 --hidden 16 --layers 3 --P 128 --F 48 > logs/cl_multi.log 2>&1 
python3 main.py --model SpaceTime --dataset multi-sine --type complex --activation GELU --dataset_norm true --epochs 50 --state_dim 32 --hidden 16 --layers 3 --P 128 --F 48 > logs/st_multi.log 2>&1 


python3 main.py --model zero-masking --dataset ETTh1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 24 > logs/zm_etth1.log 2>&1
python3 main.py --model zero-masking --dataset ETTh2 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 24 > logs/zm_etth2.log 2>&1 
python3 main.py --model zero-masking --dataset ETTm1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 24 > logs/zm_ettm1.log 2>&1 
python3 main.py --model closed-loop --dataset ETTh1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 24 > logs/cl_etth1.log 2>&1 
python3 main.py --model closed-loop --dataset ETTh2 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 24 > logs/cl_etth2.log 2>&1 
python3 main.py --model closed-loop --dataset ETTm1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 24 > logs/cl_ettm1.log 2>&1 
python3 main.py --model SpaceTime --dataset ETTh1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 24 > logs/st_etth1.log 2>&1
python3 main.py --model SpaceTime --dataset ETTh2 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 24 > logs/st_etth2.log 2>&1 
python3 main.py --model SpaceTime --dataset ETTm1 --type complex --activation GELU --dataset_norm true --epochs 10 --state_dim 128 --hidden 64 --layers 3 --P 720 --F 24 > logs/st_ettm1.log 2>&1 


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