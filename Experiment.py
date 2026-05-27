
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import csv

import matplotlib
matplotlib.use("Agg")  # headless backend (no GUI)
import matplotlib.pyplot as plt

import os
from Create_Dataset import make_sliding_forecast_dataset, shift_seq, inverse_shift_seq

from train import MIMOssm_network_and_train


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


 # ================================================================================================ PLOT Stuff

def plot_example_predictions(
    trainer,
    model,
    type,
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test,
    P, F,
    idx_list,
    save_path,
):

    device = trainer.device
    trainer.net.eval()

    # Keep only first 2 indices
    idx_list = list(idx_list)[:2]
    n_examples = len(idx_list)

    fig, axes = plt.subplots(
        n_examples, 1,
        figsize=(10, 4.6 * n_examples),
        sharex=False
    )

    title_str = "Example Predictions"

    # Add forecasting mode
    if model == "zero-masking":
        title_str += " | Zero-Masking"
    elif model == "closed-loop":
        title_str += " | Closed-Loop"
    elif model == "SpaceTime":
        title_str += " | SpaceTime"

    # Add SSM type
    if trainer.type == "real":
        title_str += " | Real $A$"
    elif trainer.type == "complex":
        title_str += " | Complex $A$"

    fig.suptitle(title_str, fontsize=27, y=0.98)
    fig.subplots_adjust(top=0.92)  # Adjust top to make room for suptitle

    if n_examples == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, idx in zip(axes, idx_list):
            if idx >= len(X_test):
                ax.set_visible(False)
                continue

            x_ex = X_test[idx:idx+1].to(device)

            t_context = np.arange(P)
            t_future = np.arange(P, P + F)

            x_context = X_test[idx, :P, 0].detach().cpu().numpy()
            y_true = Y_test[idx, :, 0].detach().cpu().numpy()

            if model == "zero-masking":
                x_shift, mean = shift_seq(x_ex, P)
                y_hat, _ = trainer.net(x_shift)
                y_hat = inverse_shift_seq(y_hat, mean)
                y_hat = y_hat[0, :, 0].detach().cpu().numpy()   # (P+F,)

                y_true_future = y_true[P:]
                y_pred_future = y_hat[P:]

            elif model == "closed-loop":
                x_ctx = x_ex[:, :P, :]
                x_ctx_shift, mean = shift_seq(x_ctx, P)
                y_hat = trainer.autoregressive_rollout(x_ctx_shift, F=F)
                y_hat = inverse_shift_seq(y_hat, mean)
                y_hat = y_hat[0, :, 0].detach().cpu().numpy()   # (F,)

                y_true_future = y_true[P:]
                y_pred_future = y_hat
            
            elif model == "SpaceTime":
                x_ctx = x_ex[:, :P, :]
                x_ctx_shift, mean = shift_seq(x_ctx, P)

                out = trainer.net.forward_spacetime_forecast(x_ctx_shift, F=F)
                y_hat = inverse_shift_seq(out["y_c"], mean)
                y_hat = y_hat[0, :, 0].detach().cpu().numpy()   # (F,)

                y_true_future = y_true[P:]
                y_pred_future = y_hat

            else:
                raise ValueError(f"Unknown model: {model}")

            # Context
            ax.plot(
                t_context,
                x_context,
                linewidth=2.2,
                label="Context",
                color='blue'
            )

            # Ground truth (now continuous with context)
            y_true_future = y_true[P-1:]
            t_future = np.arange(P-1, P + F)

            ax.plot(
                t_future,
                y_true_future,
                linewidth=2.2,
                label="Ground Truth",
                color='green'
            )

            # Prediction (still only future)
            ax.plot(
                np.arange(P, P + F),
                y_pred_future,
                linewidth=2.2,
                label="Prediction",
                color='red'
            )

            # Forecast boundary
            ax.axvline(
                P-1,
                linestyle=":",
                linewidth=1.5,
                color="black",
            )


            ax.set_title(f"Example Index {idx}", fontsize=20, pad=10)
            ax.set_ylabel("Normalized Output", fontsize=25)
            ax.grid(True, alpha=0.25)
            ax.margins(x=0.01)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.tick_params(axis='both', which='major', labelsize=20)

    axes[-1].set_xlabel("Time Step", fontsize=27)

    ax.tick_params(axis='both', which='major', labelsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=23,
    )

    fig.suptitle(title_str, fontsize=23, y=0.97)

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.08,
        top=0.82,      
        hspace=0.42    
    )


    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()



def plot_loss_curves(out, model, type, save_path, title=None):

    train_loss = np.array(out["train_loss"], dtype=float)
    val_loss = np.array(out["val_loss"], dtype=float)

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.plot(epochs, np.log10(train_loss + 1e-12), linewidth=2.2, label="Training Loss")
    ax.plot(epochs, np.log10(val_loss + 1e-12), linewidth=2.2, label="Validation Loss")

    
    
    max_xticks = 10
    n_epochs = len(epochs)

    step = max(5, int(np.ceil(n_epochs / max_xticks / 5) * 5))

    ticks = [1] + list(np.arange(step, n_epochs + 1, step))

    if n_epochs not in ticks:
        ticks.append(n_epochs)

    ax.set_xticks(ticks)




    # --- Title (bold) ---
    title_str = r"$\log_{10}(\mathrm{MSE})$ Loss vs Epoch"
    subtitle = []
    

    if model == "zero-masking":
        subtitle.append("Zero-Masking")
    elif model == "closed-loop":
        subtitle.append("Closed-Loop")
    elif model == "SpaceTime":
        subtitle.append("SpaceTime")

    if type == "real":
        subtitle.append("Real $A$")
    elif type == "complex":
        subtitle.append("Complex $A$")

    if subtitle:
        title_str += " | " + " | ".join(subtitle)

    ax.set_title(title_str, fontsize=22, pad=12)


    # --- Axis labels (bold) ---
    ax.set_xlabel("Epoch", fontsize=25)
    ax.set_ylabel(r"$\log_{10}(\mathrm{MSE})$", fontsize=25)


    # Optional: also make tick labels bold
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, fontsize=20)
    ax.margins(x=0.01)

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.12,
        top=0.88
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ============================================================================================== EXPERIMENT

def total_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def run_experiments(
    model,
    type,
    dataset,
    normalization,
    activation_fnc,
    y,
    exp_dir,
    P,
    F,
    T,
    period,
    seed,
    nb_epochs,
    batch_size,
    hidden,
    state_dim,
    w_decay_ssm,
    w_decay_others,
    NB_LAYERS,
    lr_ssm,
    lr_others,
    dropout,
    idx_example,
    splits,
    dataset_norm,
    W0,
    W1, 
    W2
):
    

    ensure_dir(exp_dir)
    set_seed(seed)

    device_train = "cuda" if torch.cuda.is_available() else "cpu"

    summary = []  # (F, final_train_mse, final_val_mse)
    all_results = []
    train_ratio, val_ratio, test_ratio = splits


    # CSV path per dataset
    csv_path = os.path.join(exp_dir, f"{dataset}_summary.csv")

    fieldnames = [
        "dataset",
        "model",
        "type",
        "P",
        "F",
        "net_params",
        "mse",
        "mae",
        "", # blanc column
        "T",
        "period",
        "epochs",
        "seed",
        "batch_size",
        "hidden",
        "state_dim",
        "layers",
        "lr_ssm",
        "lr_others",
        "dropout",
        "normalization",
        "dataset_norm",
        "activation",
        "w_decay_ssm",
        "w_decay_others",
        "w0",
        "w1",
        "w2",
    ]


    # -------------------------
    # Loop over horizons
    # -------------------------
    run_name = f"P{P}_F{F}_T{T}_period{period}_epochs{nb_epochs}_seed{seed}_B{batch_size}_hidden{hidden}_state{state_dim}_layers{NB_LAYERS}_model{model}_type{type}_lrSSM{lr_ssm}_lrOthers{lr_others}_norm{normalization}_datasetnorm{dataset_norm}_act{activation_fnc}_drop{dropout}"
    run_path = os.path.join(exp_dir, run_name)
    ensure_dir(run_path)


    y_train = y[:int(train_ratio * T)]
    y_val   = y[int(train_ratio * T):int((train_ratio + val_ratio) * T)]
    y_test  = y[int((train_ratio + val_ratio) * T):]

    X_train, Y_train = make_sliding_forecast_dataset(y_train, P=P, F=F)
    X_val, Y_val     = make_sliding_forecast_dataset(y_val, P=P, F=F)
    X_test, Y_test   = make_sliding_forecast_dataset(y_test, P=P, F=F)
    
    N = len(X_train) + len(X_val) + len(X_test)

    expected = (
        len(y_train) - (P + F) + 1
    + len(y_val)   - (P + F) + 1
    + len(y_test)  - (P + F) + 1
    )

    print("Correct window count?", N == expected)


    # -------------------------
    # Create model and train it
    # -------------------------
    trainer = MIMOssm_network_and_train(
        model=model,
        type=type,
        nb_input_neurons=1,
        nb_hidden_neurons=hidden,
        nb_output_neurons=1,
        MIMO_dim_per_neuron_for_each_layer=[(1, 1)] * NB_LAYERS,
        device=device_train,
        batch_size=batch_size,
        state_dim=state_dim,
        w_decay_ssm=w_decay_ssm,
        w_decay_others=w_decay_others,
        nb_time_steps=P + F,
        set_seed=seed,
        nb_epochs=nb_epochs,
        lr_ssm=lr_ssm,
        lr_others=lr_others,
        dropout=dropout,
        normalization=normalization,
        activation_fnc=activation_fnc,
        w0=W0,
        w1=W1,
        w2=W2,
    )

    nb_params = total_params(trainer.net)
    
    print("=" * 160)
    print(f"Starting experiment {dataset}: P={P}, F={F}, T={T}|  Model={model} | Type={type} | epochs={nb_epochs} | Net Params={nb_params} | hidden={hidden} | state_dim={state_dim} | act={activation_fnc} | norm={normalization} | dataset norm={dataset_norm} | seed={seed} | B={batch_size} | Layers={NB_LAYERS}| LR_SSM={lr_ssm} | LR_OTHERS={lr_others} | drop={dropout} | w_decay_ssm={w_decay_ssm} | w_decay_other={w_decay_others}")
    print("=" * 160)


    out = trainer.train_forecast_from_tensors(
        model=model,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        Y_train=Y_train,
        Y_val=Y_val,
        Y_test=Y_test,
        P=P,
        F=F,
        nb_epochs=nb_epochs,
        shuffle_train=True,
        num_workers=0,
        checkpoint_dir=run_path,
        dataset=dataset,
        T=T,
        period=period,
        idx_example=idx_example,
    )

    all_results.append({"dataset": dataset,"model": model,"type": type,"P": P,"F": F,"mse": out["test_mse"],"mae": out["test_mae"]})

    # -------------------------
    # Save one CSV row per experiment
    # -------------------------
    row = {
        "dataset": dataset,
        "model": model,
        "type": type,
        "P": P,
        "F": F,
        "net_params": nb_params,
        "mse": round(float(out["test_mse"]), 3),
        "mae": round(float(out["test_mae"]), 3),
        "": "", 
        "T": T,
        "period": str(period),
        "epochs": nb_epochs,
        "seed": seed,
        "batch_size": batch_size,
        "hidden": hidden,
        "state_dim": state_dim,
        "layers": NB_LAYERS,
        "lr_ssm": lr_ssm,
        "lr_others": lr_others,
        "dropout": dropout,
        "normalization": normalization,
        "dataset_norm": dataset_norm,
        "activation": activation_fnc,
        "w_decay_ssm": w_decay_ssm,
        "w_decay_others": w_decay_others,
        "w0": W0,
        "w1": W1,
        "w2": W2,
    }

    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writerow(row)

    # -------------------------
    # Save per-F plots
    # -------------------------
    plot_loss_curves(out=out, model=model, type=type, save_path=os.path.join(run_path, "log10_loss.pdf"),title=f"log10(MSE) last F steps | P={P}, F={F}",)

    plot_example_predictions(
        trainer=trainer,
        model=model,
        type=type,
        X_train=out["X_val"], Y_train=out["Y_val"],
        X_val=out["X_val"], Y_val=out["Y_val"],
        X_test=out["X_test"], Y_test=out["Y_test"],
        P=P,
        F=F,
        idx_list=idx_example,
        save_path=os.path.join(run_path, "example_prediction.pdf"),
    )

    final_train = float(out["train_loss"][-1]) if out["train_loss"] else float("nan")
    final_val = float(out["val_loss"][-1]) if out["val_loss"] else float("nan")

    print(f"Finished experiment F={F}: final train MSE={final_train:.6g} | final val MSE={final_val:.6g}")

    summary.append((F, final_train, final_val))


