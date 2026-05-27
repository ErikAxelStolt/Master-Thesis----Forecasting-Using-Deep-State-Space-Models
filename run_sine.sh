


#!/bin/bash

mkdir -p logs

python3 main.py --model zero-masking --dataset sine --type real --activation None --dataset_norm true --epochs 50 --state_dim 8 --hidden 8 --layers 3 --P 128 --F 48 > logs/zm_sine_real.log 2>&1 
python3 main.py --model zero-masking --dataset sine --type complex --activation None --dataset_norm true --epochs 50 --state_dim 6 --hidden 8 --layers 3 --P 128 --F 48 > logs/zm_sine_complex.log 2>&1 
python3 main.py --model closed-loop --dataset sine --type real --activation None --dataset_norm true --epochs 50 --state_dim 8 --hidden 8 --layers 3 --P 128 --F 48 > logs/cl_sine_real.log 2>&1 
python3 main.py --model closed-loop --dataset sine --type complex --activation None --dataset_norm true --epochs 50 --state_dim 6 --hidden 8 --layers 3 --P 128 --F 48 > logs/cl_sine_complex.log 2>&1 
python3 main.py --model SpaceTime --dataset sine --type real --activation None --dataset_norm true --epochs 50 --state_dim 8 --hidden 8 --layers 3 --P 128 --F 48 > logs/st_sine_real.log 2>&1 
python3 main.py --model SpaceTime --dataset sine --type complex --activation None --dataset_norm true --epochs 50 --state_dim 6 --hidden 8 --layers 3 --P 128 --F 48 > logs/st_sine_complex.log 2>&1 


wait