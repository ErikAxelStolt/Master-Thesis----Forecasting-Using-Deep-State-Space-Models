import argparse
import torch
from Experiment import run_experiments
from Create_Dataset import get_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="zero-masking",
                        choices=["zero-masking", "closed-loop", "SpaceTime"])
    parser.add_argument("--dataset", type=str, default="sine",
                        choices=["ETTh1", "ETTh2", "ETTm1", "sine", "nonlinear", "multi-sine"])
    parser.add_argument("--type", type=str, default="complex",
                        choices=["complex", "real"])
    parser.add_argument("--activation", type=str, default="None",
                        choices=["None", "GELU"])
    parser.add_argument("--dataset_norm", type=str, default="true",
                        choices=["true", "false"])

    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--P", type=int, default=128)
    parser.add_argument("--F", type=int, default=48)

    # weights for loss
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)




    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr_ssm", type=float, default=1e-3)
    parser.add_argument("--lr_others", type=float, default=1e-3)
    parser.add_argument("--ssm_w_decay", type=float, default=1e-4)
    parser.add_argument("--others_w_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--normalization", type=str, default="None",
                        choices=["None", "batchnorm", "layernorm"])

    return parser.parse_args()


def str_to_none(x):
    return None if x == "None" else x


def str_to_bool(x):
    return x.lower() == "true"


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL = args.model
    DATASET = args.dataset
    TYPE = args.type
    ACTIVATION_FNC = str_to_none(args.activation)
    DATASET_NORM = str_to_bool(args.dataset_norm)

    EPOCHS = args.epochs
    STATE_DIM = args.state_dim
    HIDDEN_NEURONS = args.hidden
    NB_LAYERS = args.layers

    P = args.P
    F = args.F

    PERIODS = [25, 50, 75, 100, 125]
    AMPLITUDES = [1, 1, 1, 1, 1]

    W0 = args.w0
    W1 = args.w1
    W2 = args.w2

    NORM = str_to_none(args.normalization)
    BATCH_SIZE = args.batch_size
    SEED = args.seed
    LR_SSM = args.lr_ssm
    LR_OTHERS = args.lr_others
    SSM_W_DECAY = args.ssm_w_decay
    OTHERS_W_DECAY = args.others_w_decay
    DROPOUT = args.dropout

    if DATASET == 'ETTh1' or DATASET == 'ETTh2' or DATASET == 'ETTm1':
        IDX = (1968, 2453)
    elif DATASET == 'sine' or DATASET == 'multi-sine':
        IDX = (0, 100)
    else:
        raise ValueError(f"IDX not defined for dataset {DATASET}")
    

    SPLITS = (0.6, 0.2, 0.2)

    y, T = get_dataset(SEED, DATASET, DATASET_NORM, SPLITS[0], AMPLITUDES, PERIODS, device="cpu")

    run_experiments(
        y=y,
        T=T,
        period=PERIODS,
        model=MODEL,
        type=TYPE,
        dataset=DATASET,
        activation_fnc=ACTIVATION_FNC,
        exp_dir=f"Experiments_{DATASET}",
        P=P,
        F=F,
        seed=SEED,
        nb_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden=HIDDEN_NEURONS,
        NB_LAYERS=NB_LAYERS,
        state_dim=STATE_DIM,
        w_decay_ssm=SSM_W_DECAY,
        w_decay_others=OTHERS_W_DECAY,
        lr_ssm=LR_SSM,
        lr_others=LR_OTHERS,
        dropout=DROPOUT,
        normalization=NORM,
        idx_example=IDX,
        splits=SPLITS,
        dataset_norm=DATASET_NORM,
        W0=W0,
        W1=W1,
        W2=W2
    )


if __name__ == "__main__":
    main()