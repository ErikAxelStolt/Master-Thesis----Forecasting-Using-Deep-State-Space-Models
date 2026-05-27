

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MIMOssm_network
from Create_Dataset import shift_seq, inverse_shift_seq


def build_model_from_checkpoint(ckpt, device):
    cfg = ckpt["config"]

    model_mode = cfg.get("model", "closed-loop")

    net = MIMOssm_network(
        input_shape=(cfg["batch_size"], cfg["nb_time_steps"], cfg["nb_input_neurons"]),
        layer_sizes=[cfg["nb_hidden_neurons"]] * len(cfg["MIMO_dim_per_neuron_for_each_layer"]) + [cfg["nb_output_neurons"]],
        state_dim=cfg["state_dim"],
        dropout=cfg["dropout"],
        MIMO_dim_per_neuron_for_each_layer=cfg["MIMO_dim_per_neuron_for_each_layer"],
        activation_fnc=cfg["activation_fnc"],
        normalization=cfg["normalization"],
        set_seed=cfg["set_seed"],
        ssm_type=cfg["type"],
        model=model_mode,   # IMPORTANT: needed for SpaceTime K layer
    ).to(device)

    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, cfg


@torch.no_grad()
def autoregressive_rollout(net, x, F):
    """
    x: (B, P, 1) shifted context
    returns: (B, F, 1)
    """
    _, U_list = net(x)

    last_input = x[:, -1:, :]
    preds = []

    for _ in range(F):
        y_step, U_list = net(last_input, U_list)
        next_pred = y_step[:, -1:, :]
        preds.append(next_pred)
        last_input = next_pred

    return torch.cat(preds, dim=1)


@torch.no_grad()
def plot_example_predictions_from_checkpoint(
    checkpoint_path,
    save_path=None,
    idx_list=None,
    model=None,
):
    """
    Recreates the example plot in the same style as Experiment.py, using
    X_test and Y_test stored inside the checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to saved checkpoint.
    save_path : str or None
        Where to save the plot. If None, saves next to checkpoint as
        'example_prediction_from_checkpoint.png'.
    idx_list : iterable[int] or None
        Example indices to plot. If None, uses checkpoint config['idx_example'].
        Only the first 2 indices are used to match Experiment.py.
    model : str or None
        'closed-loop' or 'zero-masking'. If None, tries checkpoint config['model'].
        If not present, defaults to 'closed-loop'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)
    net, cfg = build_model_from_checkpoint(ckpt, device)

    if "X_test" not in ckpt or "Y_test" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'X_test' and 'Y_test'. "
            "Your training code must save them to reproduce example plots directly from checkpoint."
        )

    X_test = ckpt["X_test"]
    Y_test = ckpt["Y_test"]

    P = cfg["P"]
    F = cfg["F"]
    ssm_type = cfg.get("type", None)

    if idx_list is None:
        idx_list = cfg.get("idx_example", (0, 100))
    idx_list = list(idx_list)[:2]

    if model is None:
        model = cfg.get("model", "closed-loop")

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(checkpoint_path),
            "example_prediction_from_checkpoint.png"
        )

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
    if ssm_type == "real":
        title_str += " | Real $A$"
    elif ssm_type == "complex":
        title_str += " | Complex $A$"

    fig.suptitle(title_str, fontsize=20, y=0.98)
    fig.subplots_adjust(top=0.92)

    if n_examples == 1:
        axes = [axes]

    for ax, idx in zip(axes, idx_list):
        if idx >= len(X_test):
            ax.set_visible(False)
            continue

        x_ex = X_test[idx:idx+1].to(device)

        t_context = np.arange(P)
        x_context = X_test[idx, :P, 0].detach().cpu().numpy()
        y_true = Y_test[idx, :, 0].detach().cpu().numpy()

        if model == "zero-masking":
            x_shift, mean = shift_seq(x_ex, P)
            y_hat, _ = net(x_shift)
            y_hat = inverse_shift_seq(y_hat, mean)
            y_hat = y_hat[0, :, 0].detach().cpu().numpy()

            y_true_future = y_true[P:]
            y_pred_future = y_hat[P:]

        elif model == "closed-loop":
            x_ctx = x_ex[:, :P, :]
            x_ctx_shift, mean = shift_seq(x_ctx, P)
            y_hat = autoregressive_rollout(net, x_ctx_shift, F=F)
            y_hat = inverse_shift_seq(y_hat, mean)
            y_hat = y_hat[0, :, 0].detach().cpu().numpy()

            y_true_future = y_true[P:]
            y_pred_future = y_hat
        
        elif model == "SpaceTime":
            x_ctx = x_ex[:, :P, :]
            x_ctx_shift, mean = shift_seq(x_ctx, P)

            out = net.forward_spacetime_forecast(x_ctx_shift, F=F)
            y_hat = inverse_shift_seq(out["y_c"], mean)
            y_hat = y_hat[0, :, 0].detach().cpu().numpy()

            y_true_future = y_true[P:]
            y_pred_future = y_hat

        else:
            raise ValueError(f"Unknown model mode: {model}")

        # Context
        ax.plot(
            t_context,
            x_context,
            linewidth=2.2,
            label="Context",
            color="blue"
        )

        # Ground truth continuous with context
        y_true_future = y_true[P-1:]
        t_future = np.arange(P-1, P + F)

        ax.plot(
            t_future,
            y_true_future,
            linewidth=2.2,
            label="Ground Truth",
            color="green"
        )

        # Prediction only on future
        ax.plot(
            np.arange(P, P + F),
            y_pred_future,
            linewidth=2.2,
            label="Prediction",
            color="red"
        )

        # Forecast boundary
        ax.axvline(
            P - 1,
            linestyle=":",
            linewidth=1.5,
            color="black",
        )

        ax.set_title(f"Example Index {idx}", fontsize=13, pad=10)
        ax.set_ylabel("Normalized Output", fontsize=17)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.01)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time Step", fontsize=17)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
        fontsize=20,
    )

    fig.suptitle(title_str, fontsize=20, y=0.97)

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.08,
        top=0.82,
        hspace=0.42
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {save_path}")



def plot_loss_curves_from_checkpoint(checkpoint_path, save_path=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]

    if "train_loss" not in ckpt or "val_loss" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'train_loss' and 'val_loss'. "
            "Train checkpoints must save these histories."
        )

    train_loss = np.array(ckpt["train_loss"], dtype=float)
    val_loss = np.array(ckpt["val_loss"], dtype=float)

    epochs = np.arange(1, len(train_loss) + 1)

    if save_path is None:
        save_path = "log10_loss_from_checkpoint.pdf"

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.plot(epochs, np.log10(train_loss + 1e-12), linewidth=2.2, label="Training Loss")
    ax.plot(epochs, np.log10(val_loss + 1e-12), linewidth=2.2, label="Validation Loss")




    # Keep x-axis readable for both small and large epoch counts
    max_xticks = 10
    n_epochs = len(epochs)

    step = max(5, int(np.ceil(n_epochs / max_xticks / 5) * 5))

    ticks = [1] + list(np.arange(step, n_epochs + 1, step))

    if n_epochs not in ticks:
        ticks.append(n_epochs)

    ax.set_xticks(ticks)





    title_str = r"$\log_{10}(\mathrm{MSE})$ Loss vs Epoch"
    subtitle = []

    model = cfg.get("model", None)
    ssm_type = cfg.get("type", None)

    if model == "zero-masking":
        subtitle.append("Zero-Masking")
    elif model == "closed-loop":
        subtitle.append("Closed-Loop")
    elif model == "SpaceTime":
        subtitle.append("SpaceTime")

    if ssm_type == "real":
        subtitle.append("Real $A$")
    elif ssm_type == "complex":
        subtitle.append("Complex $A$")

    if subtitle:
        title_str += " | " + " | ".join(subtitle)

    ax.set_title(title_str, fontsize=22, pad=12)

    ax.set_xlabel("Epoch", fontsize=25)
    ax.set_ylabel(r"$\log_{10}(\mathrm{MSE})$", fontsize=25)

    ax.tick_params(axis="both", which="major", labelsize=20)

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, fontsize=20)
    ax.margins(x=0.01)

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.12,
        top=0.88,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved loss plot to: {save_path}")


if __name__ == "__main__":
    epoch = "last_epoch"

    checkpoint_path = (
        f"Experiments_multi-sine/"
        "P128_F48_T2000_period[25, 50, 75, 100, 125]_epochs150_seed0_B64_hidden16_state32_layers3_modelzero-masking_typecomplex_lrSSM0.001_lrOthers0.001_normNone_datasetnormTrue_actGELU_drop0.05/"
        f"model_{epoch}.pt"
    )


    plot_example_predictions_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model="closed-loop",   # or "zero-masking"
    )

    plot_loss_curves_from_checkpoint(
        checkpoint_path=checkpoint_path,
        save_path="log10_loss_from_checkpoint.pdf",
    )

