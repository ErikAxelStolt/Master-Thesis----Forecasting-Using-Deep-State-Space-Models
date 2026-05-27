
import torch
import torch.nn as nn
import numpy as np
import math



def s4d_init(N, dt_min=1e-3, dt_max=1e-1, device=None, dtype=torch.float32):
    Nh = N // 2

    n = torch.arange(Nh, device=device, dtype=dtype)

    # Continuous-time diagonal A
    A_real = torch.full((Nh,), -0.5, device=device, dtype=dtype)
    A_imag = math.pi * n

    # Log-uniform initialization of dt
    log_dt = torch.rand(Nh, device=device, dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

    return A_real, A_imag, log_dt


class MIMOssm_network(nn.Module):
    """
    Time-invariant SSM network.

    Input:  x (B, T, in_dim)
    Output: y (B, T, out_dim)
    """

    def __init__(
        self,
        input_shape,
        layer_sizes,
        state_dim,
        dropout,
        MIMO_dim_per_neuron_for_each_layer,
        activation_fnc,
        normalization,
        set_seed,
        ssm_type,
        model
    ):
        super().__init__()

        if set_seed is not False:
            torch.manual_seed(set_seed)

        self.reshape = True if len(input_shape) > 3 else False
        self.input_size = int(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]

        self.layer_sizes = layer_sizes
        self.state_dim = int(state_dim)
        self.num_layers = len(layer_sizes)
        self.num_outputs = layer_sizes[-1]

        self.MIMO_dim_per_neuron_for_each_layer = MIMO_dim_per_neuron_for_each_layer
        self.activation_fnc = activation_fnc
        self.dropout = dropout
        self.normalization = normalization
        self.ssm_type = ssm_type

        self.use_readout_layer = True

        self.snn = self._init_layers()


        decoder_out_dim = self.layer_sizes[-2] * self.MIMO_dim_per_neuron_for_each_layer[-1][1]
        decoder_in_dim = self.layer_sizes[-2]

        if model=="SpaceTime":
            self.K = nn.Linear(decoder_out_dim, decoder_in_dim, bias=False)


    def _init_layers(self):
        snn = nn.ModuleList([])
        input_size = self.input_size

        # With readout: all but last are stateful layers
        num_hidden_layers = self.num_layers - 1

        # Number of hidden layers with SSM neurons
        for i in range(num_hidden_layers):
            # Hidden layers with all SSM neurons
            snn.append( 
                StatefulLayer(
                    input_size=input_size,
                    hidden_size=self.layer_sizes[i],
                    state_dim=self.state_dim,
                    MIMO_dim_per_neuron_for_each_layer=self.MIMO_dim_per_neuron_for_each_layer,
                    activation_fnc=self.activation_fnc,
                    layer_index=i,
                    batch_size=self.batch_size,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    ssm_type=self.ssm_type
                )
            )
            input_size = self.layer_sizes[i] 

        # Readout layer (output projection)
        snn.append(
            ReadoutLayer(
                input_size=input_size,
                hidden_size=self.layer_sizes[-1],  # out_dim
                batch_size=self.batch_size,
                MIMO_dim_per_neuron_for_each_layer=self.MIMO_dim_per_neuron_for_each_layer,
                dropout=self.dropout,
                normalization=self.normalization,
            )
        )

        return snn

    def forward(self, x, U_list=None):
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            else:
                raise NotImplementedError("Only 4D reshape supported when reshape=True")

        if U_list is None:
            U_list = [None] * (len(self.snn) - 1)   # all layers except readout

        new_U_list = []
        state_idx = 0

        for i, layer in enumerate(self.snn):
            if isinstance(layer, StatefulLayer):
                x, U = layer(x, U_list[state_idx])
                new_U_list.append(U)
                state_idx += 1
            else:
                x = layer(x)   # readout layer

        return x, new_U_list
    




    def _stateful_layers(self):
        return [layer for layer in self.snn if isinstance(layer, StatefulLayer)]

    def _encoder_layers(self):
        stateful = self._stateful_layers()
        if len(stateful) < 2:
            raise ValueError("spacetime-like mode requires at least 2 stateful layers")
        return stateful[:-1]

    def _decoder_layer(self):
        stateful = self._stateful_layers()
        if len(stateful) < 2:
            raise ValueError("spacetime-like mode requires at least 2 stateful layers")
        return stateful[-1]

    def _readout_layer(self):
        return self.snn[-1]
    

    def encode(self, x, U_list=None):
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            else:
                raise NotImplementedError("Only 4D reshape supported when reshape=True")

        encoder_layers = self._encoder_layers()

        if U_list is None:
            U_list = [None] * len(encoder_layers)

        z = x
        new_U_list = []

        for i, layer in enumerate(encoder_layers):
            z, U = layer(z, U_list[i])
            new_U_list.append(U)

        return z, new_U_list
    

    def decode_open_loop(self, z_enc, U_dec=None):
        decoder = self._decoder_layer()
        y_head = self._readout_layer()

        h_dec, U_dec = decoder(z_enc, U_dec)     # decoder hidden sequence
        y_o = y_head(h_dec)                      # open-loop signal predictions
        z_p = self.K(h_dec)         # predicted next decoder inputs

        return {
            "z_t": z_enc,        # true decoder inputs over lag region
            "h_dec": h_dec,
            "y_o": y_o,
            "z_p": z_p,
            "U_dec": U_dec,
        }
    
    def decode_closed_loop(self, z_last, U_dec, F):
        decoder = self._decoder_layer()
        y_head = self._readout_layer()

        z_curr = z_last
        y_preds = []
        z_preds = []

        for _ in range(F):
            h_step, U_dec = decoder(z_curr, U_dec)   # z_curr must be (B,1,D)
            y_step = y_head(h_step)                  # (B,1,1)
            z_next = self.K(h_step)     # (B,1,D)

            y_preds.append(y_step)
            z_preds.append(z_next)

            z_curr = z_next

        return {
            "y_c": torch.cat(y_preds, dim=1),
            "z_c": torch.cat(z_preds, dim=1),
            "U_dec": U_dec,
        }
    
    def forward_spacetime_forecast(self, x_ctx, F):
        z_enc, U_enc = self.encode(x_ctx) # U_enc 

        open_out = self.decode_open_loop(z_enc)
        z_last = open_out["z_t"][:, -1:, :]

        closed_out = self.decode_closed_loop(
            z_last=z_last,
            U_dec=open_out["U_dec"],
            F=F,
        )

        return {
            "z_t": open_out["z_t"],
            "y_o": open_out["y_o"],
            "z_p": open_out["z_p"],
            "y_c": closed_out["y_c"],
        }





class StatefulLayer(nn.Module):
    """
    One time-invariant stateful layer.

    - Uses diagonal A per neuron/state dim: A_diag (h, n)
    - Uses B per neuron: B (h, n, h_in)
    - Uses C per neuron: C (h, h_out, n)

    Outputs (B, T, h*h_out) to feed next layer / readout.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        state_dim,
        layer_index,
        MIMO_dim_per_neuron_for_each_layer,
        activation_fnc,
        dropout,
        normalization,
        ssm_type,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = int(batch_size)
        self.state_dim = int(state_dim)

        self.layer_index = int(layer_index)
        self.dropout = float(dropout)
        self.normalization = normalization
        self.ssm_type = ssm_type    

        # MIMO dims per neuron
        self.h_in, self.h_out = MIMO_dim_per_neuron_for_each_layer[self.layer_index]
        if self.layer_index == 0:
            h_before_out = 1
        else:
            _, h_before_out = MIMO_dim_per_neuron_for_each_layer[self.layer_index - 1]

        # Input and output map for next layer
        in_dim = self.input_size * h_before_out # nb features * nb of neurons in previous layer
        out_dim = self.hidden_size * self.h_in # nb features out * nb of neurons in this layer 
        self.W = nn.Identity() if (in_dim, out_dim) == (1, 1) else nn.Linear(in_dim, out_dim, bias=True) 


        A_real, A_imag, log_dt = s4d_init(self.state_dim)

        Nh = self.state_dim // 2 # number of complex conjugate pairs

        self.log_A_real_half = nn.Parameter(torch.log(-A_real).repeat(self.hidden_size, 1))

        if self.ssm_type == "complex":
            self.A_imag_half = nn.Parameter(A_imag.repeat(self.hidden_size, 1))
        elif self.ssm_type == "real":
            self.A_imag_half = nn.Parameter(A_imag.repeat(self.hidden_size, 1), requires_grad=False)  
       
        self.log_dt_half = nn.Parameter(log_dt.repeat(self.hidden_size, 1), requires_grad=False)  # Freezed, not trained
        self.B_half = nn.Parameter(torch.ones(self.hidden_size, Nh, self.h_in))
        self.C_half = nn.Parameter(torch.randn(self.hidden_size, self.h_out, Nh))

        # Norm
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(out_dim, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(out_dim)
            self.normalize = True

        self.drop = nn.Dropout(p=self.dropout)

        if activation_fnc == "GELU":
            self.activation = nn.GELU()
        elif activation_fnc == None:
            self.activation = nn.Identity()
        else:
            raise ValueError("This cleaned version keeps only GELU.")

    def forward(self, x, U=None):
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        Wx = self.W(x)

        if self.normalize:
            Wx2 = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = Wx2.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        s, U = self._cell(Wx, U)
        s = self.drop(s)
        return s, U


    def _cell(self, Wx, U=None):
        device = Wx.device
        dtype = Wx.dtype  # float dtype for inputs/params

        Nh = self.state_dim // 2

        if self.ssm_type == "complex":
            if U is None:
                U = torch.zeros(self.batch_size, Nh, self.hidden_size, device=device, dtype=torch.cfloat)
            A = -torch.exp(self.log_A_real_half) + 1j * self.A_imag_half
            dt = torch.exp(self.log_dt_half).to(torch.cfloat)
        
        elif self.ssm_type == "real":
            if U is None:
                U = torch.zeros(self.batch_size, Nh, self.hidden_size, device=device, dtype=torch.cfloat)
            A = -torch.exp(self.log_A_real_half)
            dt = torch.exp(self.log_dt_half)


        # Bilinear form 
        dtA = dt * A

        dA = (1 + dtA / 2) / (1 - dtA / 2)
        dB = (dt / (1 - dtA / 2))[:, :, None] * self.B_half.to(dtype=torch.cfloat)
                        
        # Cast B and C once per call (saves time vs casting each step)
        Cc = self.C_half.to(dtype=torch.cfloat)

        outputs = []
        T = Wx.shape[1]

        dA_T = dA.transpose(0, 1)

        for t in range(T):
            I = Wx[:, t, :].to(dtype).reshape(self.batch_size, self.hidden_size, self.h_in)

            BI = torch.einsum("hni,bhi->bnh", dB, I.to(dtype=torch.cfloat))
            U = torch.einsum("nh,bnh->bnh", dA_T, U) + BI
            y = 2 * torch.einsum("hon,bnh->bho", Cc, U).real
            y = self.activation(y)

            outputs.append(y.reshape(self.batch_size, -1))

        return torch.stack(outputs, dim=1), U # (B, Nh, hidden_size) -> to feed next time step and keep track of state across time steps


class ReadoutLayer(nn.Module):
    """
    Stateless readout layer: maps last hidden layer features -> output features, per time step.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        MIMO_dim_per_neuron_for_each_layer,
        normalization,
        dropout,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)  # out_dim
        self.batch_size = int(batch_size)

        self.dropout = float(dropout)
        self.normalization = normalization

        # MIMO dims from previous layer
        _, h_last_out = MIMO_dim_per_neuron_for_each_layer[-1]
        in_dim = self.input_size * h_last_out

        self.W = nn.Identity() if (in_dim, self.hidden_size) == (1, 1) else nn.Linear(in_dim, self.hidden_size, bias=True)

        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        self.drop = nn.Dropout(p=self.dropout)

    def forward(self, x):
        # x: (B, T, in_dim)
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        Wx = self.W(x)  # (B, T, out_dim)

        if self.normalize:
            Wx2 = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = Wx2.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        return self.drop(Wx)
    
