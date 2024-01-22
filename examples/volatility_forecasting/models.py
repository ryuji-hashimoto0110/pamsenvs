import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from typing import Optional

class PositionalEncoding(Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        """_summary_

        References:
            - https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Args:
            hidden_dim (int): _description_
            dropout (float, optional): _description_. Defaults to 0.1.
            max_len (int, optional): _description_. Defaults to 5000.
        """
        super().__init__()
        self.dropout: Module = nn.Dropout(p=dropout)
        position: Tensor = torch.arange(max_len).unsqueeze(1)
        div_term: Tensor = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe: Tensor = torch.zeros(1, max_len, hidden_dim)
        pe[0,:,0::2] = torch.sin(position * div_term)
        pe[0,:,1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape ``[batch_size, seq_len, hidden_dim]``
        """
        self.pe.device = x.device
        x = x + self.pe[:,:x.shape[1],:]
        return self.dropout(x)

class FFN(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_ff: int,
        d_output: int
    ) -> None:
        """_summary_

        Args:
            d_input (int): _description_
            d_ff (int): _description_
            d_output (int): _description_
        """
        super().__init__()
        self.linear1: Module = nn.Linear(d_input, d_ff)
        self.linear2: Module = nn.Linear(d_ff, d_output)
        self.dropout: Module = nn.Dropout(0.1)

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class RVPredictor(nn.Module):
    def __init__(
        self,
        encoder_type: str,
        input_dim: int,
        hidden_dim: int,
        nhead: Optional[int] = None,
    ) -> None:
        super(RVPredictor, self).__init__()
        self.ffn1: Module = FFN(input_dim, hidden_dim, hidden_dim)
        if encoder_type == "LSTM":
            self.encoder: Module = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim,
                num_layers=2, dropout=0.1, batch_first=True
            )
        elif encoder_type == "Transformer":
            assert nhead is not None
            pos: Module = PositionalEncoding(hidden_dim)
            encoder_layer: Module = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, batch_first=True
            )
            transformer_encoder: Module = nn.TransformerEncoder(
                encoder_layer, num_layers=1
            )
            self.encoder: Module = nn.Sequential(pos, transformer_encoder)
        else:
            raise NotImplementedError
        self.ffn2: Module = FFN(hidden_dim, hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.ffn1(x)
        x: Tensor | tuple[Tensor] = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0][:,-1,:]
        elif isinstance(x, Tensor):
            x = x[:,-1,:]
        x: Tensor = self.ffn2(x)
        return x