
import csv
import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from Create_Dataset import shift_seq, inverse_shift_seq 
from model import MIMOssm_network


class MIMOssm_network_and_train:
    def __init__(
        self,
        model,
        type,
        nb_input_neurons,
        nb_hidden_neurons,
        nb_output_neurons,
        MIMO_dim_per_neuron_for_each_layer,
        device,
        batch_size,
        state_dim,
        nb_time_steps,
        set_seed,
        w_decay_ssm,
        w_decay_others,
        lr_ssm,
        lr_others,
        nb_epochs,
        activation_fnc,
        dropout,
        normalization,
        w0,
        w1,
        w2
    ):
        self.device = device
        self.nb_input_neurons = nb_input_neurons
        self.nb_output_neurons = nb_output_neurons
        self.nb_hidden_neurons = nb_hidden_neurons
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.nb_time_steps = nb_time_steps
        self.nb_epochs = nb_epochs
        self.set_seed = set_seed

        self.type = type
        self.activation_fnc = activation_fnc
        self.dropout = dropout
        self.normalization = normalization
        self.w_decay_ssm = w_decay_ssm
        self.w_decay_others = w_decay_others
        self.lr_ssm = lr_ssm
        self.lr_others = lr_others

        self.w0 = w0
        self.w1 = w1
        self.w2 = w2

        self.MIMO_dim_per_neuron_for_each_layer = MIMO_dim_per_neuron_for_each_layer
        self.nb_hidden_layers = len(self.MIMO_dim_per_neuron_for_each_layer)
            

        self.net = MIMOssm_network(
            input_shape=(self.batch_size, self.nb_time_steps, self.nb_input_neurons),
            layer_sizes=[self.nb_hidden_neurons] * self.nb_hidden_layers + [self.nb_output_neurons],
            MIMO_dim_per_neuron_for_each_layer=self.MIMO_dim_per_neuron_for_each_layer,
            state_dim=self.state_dim,
            set_seed=self.set_seed,
            activation_fnc=activation_fnc,
            dropout=dropout,
            normalization=normalization,
            ssm_type=type,
            model=model
        ).to(self.device)

        # Loss (you can override from outside)
        self.loss_fn = torch.nn.MSELoss()

        # Optimizer: separate SSM params vs others
        ssm_params = []
        other_params = []
        ssm_keys = ["log_A_real_half", "log_dt_half", "A_imag_half", "B_half", "C_half"]

        for name, p in self.net.named_parameters(): # consider ssm parameter if its name contains "A_diag" or ".B" or ".C" 
            if any(k in name for k in ssm_keys):
                ssm_params.append(p)
            else:
                other_params.append(p)


        self.opt = torch.optim.AdamW([{"params": ssm_params, "lr": lr_ssm, "weight_decay": w_decay_ssm}, {"params": other_params, "lr": lr_others, "weight_decay": w_decay_others},])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(1, self.nb_epochs), eta_min=1e-5)


    def autoregressive_rollout(self, x, F):
        """
        x: (B, P, 1) past observed signal
        returns: (B, F, 1) future forecast
        """
        _, U_list = self.net(x)              # consume context and build hidden state

        last_input = x[:, -1:, :]            # last observed y
        preds = []

        for _ in range(F):
            y_step, U_list = self.net(last_input, U_list)
            next_pred = y_step[:, -1:, :]
            preds.append(next_pred)
            last_input = next_pred           # feedback prediction

        return torch.cat(preds, dim=1)



    def train_one_epoch(self, model, train_loader, F: int, P: int):
        self.net.train()

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)


            if model == 'zero-masking':
                x_shift, mean = shift_seq(x, P)
                y_hat, _ = self.net(x_shift)
                y_hat = inverse_shift_seq(y_hat, mean)
                loss_val = self.loss_fn(y_hat[:, -F:, :], y[:, -F:, :])

            elif model == 'closed-loop':
                x_ctx = x[:, :P, :]
                x_ctx_shift, mean = shift_seq(x_ctx, P)
                y_hat = self.autoregressive_rollout(x_ctx_shift, F=F)
                y_hat = inverse_shift_seq(y_hat, mean)
                loss_val = self.loss_fn(y_hat[:, -F:, :], y[:, -F:, :])

            elif model == 'SpaceTime':
                x_ctx = x[:, :P, :]
                x_ctx_shift, mean = shift_seq(x_ctx, P)

                out = self.net.forward_spacetime_forecast(x_ctx_shift, F=F)
                y_hat = out["y_c"]  
                y_hat = inverse_shift_seq(y_hat, mean)
                loss_val, loss_parts = self.compute_spacetime_loss(out, y, P=P, F=F, mean=mean)

            else:
                raise ValueError(f"Unknown model: {model}")
            

            self.opt.zero_grad()
            loss_val.backward()


            for p in self.net.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    p.grad = torch.where(torch.isnan(p.grad), torch.zeros_like(p.grad), p.grad)

            self.opt.step()

        self.scheduler.step()


    @torch.no_grad()
    def test(self, model, loader, P, F):
        self.net.eval()

        total_abs_error = 0.0
        total_sq_error = 0.0
        total_elements = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)


            if model == 'zero-masking':
                x_shift, mean = shift_seq(x, P)
                y_hat, _ = self.net(x_shift)

                loss_val = self.loss_fn(y_hat[:, -F:, :], y[:, -F:, :])

            elif model == 'closed-loop':
                x_ctx = x[:, :P, :]
                x_ctx_shift, mean = shift_seq(x_ctx, P)
                y_hat = self.autoregressive_rollout(x_ctx_shift, F=F)

                loss_val = self.loss_fn(y_hat[:, -F:, :], y[:, -F:, :])

            elif model == 'SpaceTime':
                x_ctx = x[:, :P, :]
                x_ctx_shift, mean = shift_seq(x_ctx, P)
                out = self.net.forward_spacetime_forecast(x_ctx_shift, F=F)
                y_hat = out["y_c"]  
        

            y_hat = inverse_shift_seq(y_hat, mean)


            y_hat = y_hat[:, -F:, :]
            y = y[:, -F:, :]

            abs_error = torch.abs(y_hat - y)
            sq_error = (y_hat - y) ** 2

            total_abs_error += abs_error.sum().item()
            total_sq_error += sq_error.sum().item()

            total_elements += y.numel()

        if total_elements == 0:
            return float("nan"), float("nan")

        mae = total_abs_error / total_elements
        mse = total_sq_error / total_elements

        return mse, mae




    def compute_spacetime_loss(self, out, y_full, P, F, mean):

        assert out["y_c"].shape[1] == F
        assert out["y_o"].shape[1] == P
        assert out["z_t"].shape[1] == P
        assert out["z_p"].shape[1] == P
        assert y_full.shape[1] == P + F


        y_c = inverse_shift_seq(out["y_c"], mean)   # future signal prediction
        y_o = inverse_shift_seq(out["y_o"], mean)   # lag-region signal prediction

        z_t = out["z_t"]   # true decoder inputs
        z_p = out["z_p"]   # predicted next decoder inputs

        y_target_lag = y_full[:, :P, :]
        y_target_future = y_full[:, P:P+F, :]

        L_horizon = self.loss_fn(y_c, y_target_future)

        L_lag_open = self.loss_fn(
            y_o[:, :-1, :],
            y_target_lag[:, 1:, :]
        )

        L_lag_closed = self.loss_fn(
            z_p[:, :-1, :],
            z_t[:, 1:, :]
        )

        total = self.w0 * L_horizon + self.w1 * L_lag_open + self.w2 * L_lag_closed

        return total, {
            "L_horizon": L_horizon.detach(),
            "L_lag_open": L_lag_open.detach(),
            "L_lag_closed": L_lag_closed.detach(),
        }





    def train_forecast_from_tensors(
        self,
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        F,
        P,
        nb_epochs,
        checkpoint_dir,
        dataset=None,
        T=None,
        period=None,
        idx_example=None,
        shuffle_train=True,
        num_workers=0,
    ):
        if nb_epochs is None:
            nb_epochs = self.nb_epochs

        train_loader = DataLoader(TensorDataset(X_train, Y_train),batch_size=self.batch_size, shuffle=shuffle_train, drop_last=True, pin_memory=True, num_workers=num_workers)
        val_loader = DataLoader(TensorDataset(X_val, Y_val),batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=num_workers)
        test_loader = DataLoader(TensorDataset(X_test, Y_test),batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=num_workers)

        print(f"Samples: {len(X_train)}, {len(X_val)}, {len(X_test)}")


        train_loss_hist, val_loss_hist = [], []
        for e in range(nb_epochs):
            self.train_one_epoch(model, train_loader, F=F, P=P)
            tr_mse, tr_mae = self.test(model, train_loader, F=F, P=P)
            va_mse, va_mae = self.test(model, val_loader, F=F, P=P)
            train_loss_hist.append(tr_mse)
            val_loss_hist.append(va_mse)

            if (e % 5 == 0) or (e == nb_epochs - 1):
                print(f"Epoch {e:4d} | train MSE={tr_mse:.6g} MAE={tr_mae:.6g} | val MSE={va_mse:.6g} MAE={va_mae:.6g}")


            if e == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, "model_epoch_0.pt"),
                    model=model,
                    epoch=e,
                    P=P,
                    F=F,
                    train_loss_hist=train_loss_hist,
                    val_loss_hist=val_loss_hist,
                    dataset=dataset,
                    T=T,
                    period=period,
                    idx_example=idx_example,
                    X_test=X_test,
                    Y_test=Y_test,
                )


            if e == nb_epochs - 1:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, "model_last_epoch.pt"),
                    model=model,
                    epoch=e,
                    P=P,
                    F=F,
                    train_loss_hist=train_loss_hist,
                    val_loss_hist=val_loss_hist,
                    dataset=dataset,
                    T=T,
                    period=period,
                    idx_example=idx_example,
                    X_test=X_test,
                    Y_test=Y_test,
                )
        

        
        tt_mse, tt_mae = self.test(model, test_loader, F=F, P=P)
        print(f"Final Test MSE={tt_mse:.6g} MAE={tt_mae:.6g}")


        csv_path = os.path.join(checkpoint_dir, "final_test_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["test_mse", "test_mae"])
            writer.writerow([tt_mse, tt_mae])

        return {"train_loss": train_loss_hist, "val_loss": val_loss_hist, "test_mse": tt_mse, "test_mae": tt_mae, "X_val": X_val.cpu(), "Y_val": Y_val.cpu(), "X_test": X_test.cpu(), "Y_test": Y_test.cpu()}
    

    def _save_checkpoint(
            self,
            checkpoint_path,
            model,
            epoch,
            P,
            F,
            train_loss_hist,
            val_loss_hist,
            dataset=None,
            T=None,
            period=None,
            idx_example=None,
            X_test=None,
            Y_test=None,
        ):
            
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss_hist,
            "val_loss": val_loss_hist,
            "config": {
                "nb_input_neurons": self.nb_input_neurons,
                "nb_hidden_neurons": self.nb_hidden_neurons,
                "nb_output_neurons": self.nb_output_neurons,
                "MIMO_dim_per_neuron_for_each_layer": self.MIMO_dim_per_neuron_for_each_layer,
                "device": self.device,
                "batch_size": self.batch_size,
                "state_dim": self.state_dim,
                "nb_time_steps": self.nb_time_steps,
                "set_seed": self.set_seed,
                "nb_epochs": self.nb_epochs,
                "P": P,
                "F": F,
                "type": self.type,
                "model": model,   # add this
                "activation_fnc": self.activation_fnc,
                "dropout": self.dropout,
                "normalization": self.normalization,
                "lr_ssm": self.lr_ssm,
                "lr_others": self.lr_others,
                "w_decay_ssm": self.w_decay_ssm,
                "w_decay_others": self.w_decay_others,
                "dataset": dataset,
                "T": T,
                "period": period,
                "idx_example": idx_example,
            },
        }

        if X_test is not None:
            checkpoint["X_test"] = X_test.detach().cpu()
        if Y_test is not None:
            checkpoint["Y_test"] = Y_test.detach().cpu()

        torch.save(checkpoint, checkpoint_path)