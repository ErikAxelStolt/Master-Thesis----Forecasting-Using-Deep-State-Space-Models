
import torch
import numpy as np  
import os
import pandas as pd

# ================================================================================================= Norm stuff

def fit_normalizer(y_train):
    mean = y_train.mean()
    std = y_train.std() + 1e-12
    return mean, std

def apply_normalizer(y, mean, std):
    return (y - mean) / std


def shift_seq(x, P):
    # Shift the sequence by the mean of the first P steps
    mean = x[:, :P, :].mean(dim=1, keepdim=True)  # (B, 1, 1)
    x_shifted = x - mean 
    return x_shifted, mean

def inverse_shift_seq(x_shifted, mean):
    # Inverse the shift by adding the mean back
    x = x_shifted + mean
    return x


# ================================================================================================= Creation stuff


def make_sine(T, amplitude, period, device="cpu"):
    t = torch.arange(T, dtype=torch.float32, device=device)
    omega = 2.0 * np.pi / float(period)
    u = amplitude * torch.sin(omega * t)
    return u  

def get_ssm_matrices(dtype=torch.float32, device="cpu"):
    A = torch.diag(torch.tensor([0.9, 0.8, 0.9, 0.8], dtype=dtype, device=device))  # (4,4)
    B = torch.tensor([0.9, 0.8, 0.9, 0.8], dtype=dtype, device=device).unsqueeze(1)  # (4,1)
    C = torch.tensor([0.1, 0.2, 0.1, 0.2], dtype=dtype, device=device).unsqueeze(0)  # (1,4)
    return A, B, C


def run_linear_ssm(u: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    """
    SSM:

    x_{t+1} = A x_t + B u_t
    y_t = C x_t

    A: (N, N)
    B: (N, 1)
    C: (1, N)
    u: (T,)
    """
    T = u.shape[0]
    N = A.shape[0]

    x = torch.zeros(N, dtype=u.dtype, device=u.device)
    y = torch.empty(T, dtype=u.dtype, device=u.device)

    for t in range(T):
        Ax = A @ x                 # (N,)
        Bu = B[:, 0] * u[t]        # (N,)
        x = Ax + Bu                # (N,)
        y[t] = C[0, :] @ x         # scalar

    return y  # (T,)


def run_nonlinear_ssm(u: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    """
    SSM:

    x_{t+1} = GELU(A x_t) + B u_t
    y_t = C x_t + noise

    A: (N, N)
    B: (N, 1)
    C: (1, N)
    u: (T,)
    """
    T = u.shape[0]
    N = A.shape[0]

    x = torch.zeros(N, dtype=u.dtype, device=u.device)
    y = torch.empty(T, dtype=u.dtype, device=u.device)

    for t in range(T):
        Ax = torch.nn.functional.gelu(A @ x)   # (N,)
        Bu = B[:, 0] * u[t]                     # (N,)
        x = Ax + Bu                             # (N,)
        y[t] = (C[0, :] @ x) + 2 * torch.randn((), device=u.device, dtype=u.dtype)

    return y  # (T,)


def mulitple_sines(T, amplitudes, periods, device="cpu"):
    t = torch.arange(T, dtype=torch.float32, device=device)
    u = torch.zeros(T, dtype=torch.float32, device=device)

    for amplitude, period in zip(amplitudes, periods):
        u += make_sine(T=T, amplitude=amplitude, period=period, device=device)

    return u  


def make_sliding_forecast_dataset(y: torch.Tensor, P: int, F: int):
    """
    X[i] = [u[i:i+P], 0..0]     length P+F  (F 0s after P)
    Y[i] = y[i:i+P+F]           length P+F

    y is output from fixed SSM
    """
    assert y.ndim == 1
    T = y.shape[0]
    N = T - (P + F) + 1 
    if N <= 0:
        raise ValueError(f"T={T} too small for P={P}, F={F}")

    X = torch.zeros((N, P + F), dtype=y.dtype, device=y.device)
    Y = torch.empty((N, P + F), dtype=y.dtype, device=y.device)

    for i in range(N): # FIIIIIIIIIIIIIIIIIIIIIZ
        X[i, :P] = y[i:i+P] # first P steps are u, last F steps are already 0 from initialization
        Y[i, :] = y[i:i+P+F] 

    return X.unsqueeze(-1), Y.unsqueeze(-1)  # add dim for features for network architectural purposes: (N, P+F, 1) for both X and Y


# ================================================================================================= Load stuff

def load_ETT(dataset, T, root_path="../Datasets",target_col="OT", device="cpu"):
    file_name=f"{dataset}.csv"
    path = os.path.join(root_path, file_name)
    df = pd.read_csv(path)
    df = df.iloc[:T]

    values = df[target_col].values.astype("float32")
    y = torch.tensor(values, dtype=torch.float32, device=device)
    return y  # raw, not normalized


def get_dataset(seed, dataset, norm, train_split, AMPLITUDES, PERIODS, device="cpu"):

    torch.manual_seed(seed)

    if dataset == 'ETTh1' or dataset == 'ETTh2' or dataset == 'ETTm1':
        T = 17420
        y = load_ETT(dataset, T)

    elif dataset == 'sine':
        T = 2000
        y = make_sine(T=T, amplitude=AMPLITUDES[1], period=PERIODS[1], device=device)
        # A, B, C = get_ssm_matrices(device=device)
        # y = run_linear_ssm(u, A, B, C)
        y = y + 0.3 * torch.randn_like(y) 

    elif dataset == 'nonlinear':
        T = 2000
        u = make_sine(T=T, amplitude=AMPLITUDES[1], period=PERIODS[1], device=device)
        A, B, C = get_ssm_matrices(device=device)
        y = run_nonlinear_ssm(u, A, B, C)
        y = y + 0.3 * torch.randn_like(y) 
    
    elif dataset == 'multi-sine':
        T = 2000
        y = mulitple_sines(T=T, amplitudes=AMPLITUDES, periods=PERIODS, device=device)
        y = y + 0.3 * torch.randn_like(y)

    else: 
        raise ValueError("Please chose avaliable dataset. Avaliable are 'sine', 'nonlinear', 'ETTh1', 'ETTh2', 'ETTm1'")
    
    if norm:
        scale_len = int(train_split * T)
        mean, std = fit_normalizer(y[:scale_len]) # get mean and std only for training dataset
        print('mean:', mean, 'var:', std ** 2)
        y = apply_normalizer(y, mean, std) # apply normalization to whole dataset using training mean and std

    return y, T  # raw, not normalized
