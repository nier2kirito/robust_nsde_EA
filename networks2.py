import torch
import torch.nn as nn

class Net_Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(Net_Transformer, self).__init__()
        self.model_dim = model_dim
        
        # Project input to model_dim if different from input_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.activation = nn.Softplus()
        
    def forward(self, x):
        # Ensure x has three dimensions: (batch_size, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension if missing

        # Project the inputs
        x = self.input_proj(x)  # (batch, seq_len, model_dim)
        # PyTorch's transformer expects (seq_len, batch, model_dim)
        x = x.permute(1, 0, 2)
        
        transformer_out = self.transformer_encoder(x)  # (seq_len, batch, model_dim)
        # For prediction, take the output of the last sequence element.
        final_step = transformer_out[-1, :, :]
        out = self.fc_out(final_step)
        out = self.activation(out)
        return out
    def freeze(self):
        """
        Freeze all parameters in the model.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze all parameters in the model.
        """
        for param in self.parameters():
            param.requires_grad = True
class Net_LSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, output_dim, bidirectional=False):
        super(Net_LSTM, self).__init__()
        self.bidirectional = bidirectional
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        final_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        
        self.fc = nn.Linear(final_dim, output_dim)
        self.activation = nn.Softplus()
        
    def forward(self, x):
        # Ensure x has three dimensions: (batch_size, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension if missing

        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        final_step = lstm_out[:, -1, :]  # Take the output from the last time step
        out = self.fc(final_step)
        out = self.activation(out)
        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
class Net_Transformer_Grid(nn.Module):
    """
    A wrapper class for multiple Transformer networks, one for each maturity.
    """
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, n_maturities, dropout=0.1):
        super(Net_Transformer_Grid, self).__init__()
        self.nets = nn.ModuleList([
            Net_Transformer(input_dim, model_dim, num_heads, num_layers, output_dim, dropout)
            for _ in range(n_maturities)
        ])
    
    def forward_idx(self, idx, x):
        """
        Forward pass for a specific network based on the index.
        """
        return self.nets[idx](x)

    def freeze(self):
        """
        Freeze all networks in the grid.
        """
        for net in self.nets:
            for param in net.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze all networks in the grid.
        """
        for net in self.nets:
            for param in net.parameters():
                param.requires_grad = True

class Net_LSTM_Grid(nn.Module):
    """
    A wrapper class for multiple LSTM networks, one for each maturity.
    """
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, output_dim, n_maturities):
        super(Net_LSTM_Grid, self).__init__()
        self.nets = nn.ModuleList([
            Net_LSTM(input_dim, lstm_hidden_dim, lstm_layers, output_dim)
            for _ in range(n_maturities)
        ])
    
    def forward_idx(self, idx, x):
        """
        Forward pass for a specific network based on the index.
        """
        return self.nets[idx](x)

    def freeze(self):
        """
        Freeze all networks in the grid.
        """
        for net in self.nets:
            for param in net.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze all networks in the grid.
        """
        for net in self.nets:
            for param in net.parameters():
                param.requires_grad = True

class Net_FFN(nn.Module):
    def __init__(self, dim, nOut, n_layers, vNetWidth, activation="relu", activation_output="id", batchnorm=False):
        super(Net_FFN, self).__init__()
        self.dim = dim
        self.nOut = nOut
        self.batchnorm = batchnorm
        
        # Set activation functions
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("unknown activation function {}".format(activation))

        # Set output activation function
        if activation_output == "id":
            self.activation_output = nn.Identity()
        elif activation_output == "softplus":
            self.activation_output = nn.Softplus()
        else:
            raise ValueError("unknown output activation function {}".format(activation_output))
        
        # Define the layers
        self.i_h = self.hiddenLayerT1(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for _ in range(n_layers-1)])
        self.h_o = self.outputLayer(vNetWidth, nOut)
    
    def hiddenLayerT1(self, nIn, nOut):
        if self.batchnorm:
            return nn.Sequential(
                nn.Linear(nIn, nOut, bias=True),
                nn.BatchNorm1d(nOut),
                self.activation
            )   
        else:
            return nn.Sequential(
                nn.Linear(nIn, nOut, bias=True),
                self.activation
            )
    
    def outputLayer(self, nIn, nOut):
        return nn.Sequential(
            nn.Linear(nIn, nOut, bias=True),
            self.activation_output
        )
    
    def forward(self, S):
        h = self.i_h(S)  # Initial layer
        
        for l in range(len(self.h_h)):
            residual = h  # Save the input to this layer for residual connection
            h = self.h_h[l](h)
            h = h + residual  # Add the residual (skip connection)
        
        output = self.h_o(h)  # Final output layer
        return output

class Net_timegrid(nn.Module):
    """One feedforward network per timestep!"""
    def __init__(self, dim, nOut, n_layers, vNetWidth, n_maturities, activation="relu", activation_output="id"):
        super().__init__()
        self.dim = dim
        self.nOut = nOut

        # Use the updated Net_FFN class with residuals
        self.net_t = nn.ModuleList([Net_FFN(dim, nOut, n_layers, vNetWidth, activation=activation, activation_output=activation_output) for _ in range(n_maturities)])
        
    def forward_idx(self, idnet, x):
        y = self.net_t[idnet](x)
        return y

    def freeze(self, *args):
        if not args:
            for p in self.net_t.parameters():
                p.requires_grad = False
        else:
            self.unfreeze()
            for idx in args:
                for p in self.net_t[idx].parameters():
                    p.requires_grad_(False)

    def unfreeze(self, *args):
        if not args:
            for p in self.net_t.parameters():
                p.requires_grad = True
        else:
            # we just unfreeze the parameters between [last_T,T]
            self.freeze()
            for idx in args:
                for p in self.net_t[idx].parameters():
                    p.requires_grad = True


class TransformerDiffusion(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(TransformerDiffusion, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        return x

    def freeze(self):
        # Freeze all parameters of the model by setting requires_grad to False
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # Unfreeze all parameters of the model by setting requires_grad to True
        for param in self.parameters():
            param.requires_grad = True

    
class TransformerNet(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1, nOut=1):
        """
        A Transformer-based network to process the entire time series.
        
        Args:
            input_dim (int): Dimension of the input features.
            model_dim (int): Dimension for the internal representation (d_model).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout rate.
            nOut (int): Output dimension.
        """
        super(TransformerNet, self).__init__()
        # Project input features to the model dimension
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # Create a transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final output layer
        self.output_layer = nn.Linear(model_dim, nOut)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, input_dim)
        
        Returns:
            output: Tensor of shape (batch_size, nOut)
        """
        # Project the input
        x = self.input_proj(x)  # shape: (batch_size, seq_length, model_dim)
        
        # Transformer expects inputs in shape (seq_length, batch_size, model_dim)
        x = x.transpose(0, 1)  # shape: (seq_length, batch_size, model_dim)
        
        # Process with transformer encoder
        x = self.transformer_encoder(x)  # shape remains: (seq_length, batch_size, model_dim)
        
        # Option 1: Use the last time step output as a summary
        x_last = x[-1, :, :]  # shape: (batch_size, model_dim)
        output = self.output_layer(x_last)  # shape: (batch_size, nOut)
        
        # Option 2: Use pooling (e.g., mean pooling) over the sequence
        # x_pooled = x.mean(dim=0)
        # output = self.output_layer(x_pooled)
        
        return output