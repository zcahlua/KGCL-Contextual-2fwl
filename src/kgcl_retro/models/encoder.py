from typing import Tuple  # Explanation: imports selected names needed to encode molecular graphs with D-MPNN and optional attention
import math  # Explanation: imports math for encode molecular graphs with D-MPNN and optional attention
import torch  # Explanation: imports torch for encode molecular graphs with D-MPNN and optional attention
import torch.nn as nn  # Explanation: imports torch.nn as nn for encode molecular graphs with D-MPNN and optional attention

from kgcl_retro.models.utils import index_select_ND  # Explanation: imports packaged tensor indexing used by the message-passing encoder.


class MPNEncoder(nn.Module):  # Explanation: defines MPNEncoder, directed message passing graph encoder

    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_size: int,  # Explanation: defines __init__, which encode molecular graphs with D-MPNN and optional attention
                 depth: int, dropout: float = 0.15, atom_message: bool = False):  # Explanation: assigns an intermediate value used by later computation
        """
        Parameters
        ----------
        atom_fdim: Atom feature vector dimension.
        bond_fdim: Bond feature vector dimension.
        hidden_size: Hidden layers dimension
        depth: Number of message passing steps
        droupout: the droupout rate
       """
        super(MPNEncoder, self).__init__()  # Explanation: executes this statement as part of encode molecular graphs with D-MPNN and optional attention
        self.atom_fdim = atom_fdim  # Explanation: stores this value on the object for later model operations
        self.bond_fdim = bond_fdim  # Explanation: stores this value on the object for later model operations
        self.hidden_size = hidden_size  # Explanation: stores this value on the object for later model operations
        self.depth = depth  # Explanation: stores this value on the object for later model operations
        self.dropout = dropout  # Explanation: stores this value on the object for later model operations
        # self.atom_message = atom_message

        # Input
        # input_dim = self.atom_fdim if self.atom_message else self.bond_fdim
        input_dim = self.bond_fdim  # Explanation: assigns an intermediate value used by later computation
        self.w_i = nn.Linear(input_dim, self.hidden_size, bias=False)  # Explanation: creates a learned linear projection

        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)  # Explanation: updates D-MPNN messages with a GRU cell

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)  # Explanation: adds dropout regularization
        # Output
        self.W_o = nn.Sequential(  # Explanation: stores this value on the object for later model operations
            nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size), nn.SELU())  # Explanation: creates a learned linear projection

    def forward(self, graph_tensors: Tuple[torch.Tensor], mask: torch.Tensor) -> torch.FloatTensor:  # Explanation: defines forward, which encode molecular graphs with D-MPNN and optional attention
        """
        Forward pass of the graph encoder. Encodes a batch of molecular graphs.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details, the incoming bond indices of atoms
            the index of the atom the bond is coming from, the index of the reverse bond and the undirected bond index
            to the beginindex and endindex of the atoms.
        mask: torch.Tensor,
            Masks on nodes
        """
        f_atoms, f_bonds, f_fgs, atom_num, n_mols_tensor, a2b, b2a, b2revb, undirected_b2a = graph_tensors  # Explanation: assigns an intermediate value used by later computation
        # Input
        input = self.w_i(f_bonds)  # num_bonds x hidden  # Explanation: assigns an intermediate value used by later computation

        # Message passing
        message = input  # Explanation: assigns an intermediate value used by later computation
        message_mask = torch.ones(message.size(0), 1, device=message.device)  # Explanation: assigns an intermediate value used by later computation
        message_mask[0, 0] = 0  # first message is padding  # Explanation: assigns an intermediate value used by later computation

        for depth in range(self.depth - 1):  # Explanation: iterates over this collection to process each item
            # num_atoms x max_num_bonds x hidden
            nei_a_message = index_select_ND(message, a2b)  # Explanation: assigns an intermediate value used by later computation
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden  # Explanation: assigns an intermediate value used by later computation
            rev_message = message[b2revb]  # num_bonds x hidden  # Explanation: assigns an intermediate value used by later computation
            message = a_message[b2a] - rev_message  # num_bonds x hidden  # Explanation: assigns an intermediate value used by later computation

            message = self.gru(input, message)  # num_bonds x hidden_size  # Explanation: assigns an intermediate value used by later computation
            message = message * message_mask  # Explanation: assigns an intermediate value used by later computation
            message = self.dropout_layer(message)  # num_bonds x hidden  # Explanation: assigns an intermediate value used by later computation

        # num_atoms x max_num_bonds x hidden
        nei_a_message = index_select_ND(message, a2b)  # Explanation: assigns an intermediate value used by later computation

        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden  # Explanation: assigns an intermediate value used by later computation
        # num_atoms x (atom_fdim + hidden)
        a_input = torch.cat([f_atoms, a_message], dim=1)  # Explanation: concatenates tensors along an existing dimension
        atom_hiddens = self.W_o(a_input)  # num_atoms x hidden  # Explanation: computes an intermediate value for molecular graph editing

        if mask is None:  # Explanation: checks this condition to choose the next execution path
            mask = torch.ones(atom_hiddens.size(0), 1, device=f_atoms.device)  # Explanation: assigns an intermediate value used by later computation
            mask[0, 0] = 0  # first node is padding  # Explanation: assigns an intermediate value used by later computation

        return atom_hiddens * mask  # Explanation: returns this computed result to the caller


class MultiHeadAttention(nn.Module):  # Explanation: defines MultiHeadAttention, multi-head scaled dot-product attention layer
    def __init__(self, heads, d_model, dropout=0.1):  # Explanation: defines __init__, which encode molecular graphs with D-MPNN and optional attention
        super(MultiHeadAttention, self).__init__()  # Explanation: executes this statement as part of encode molecular graphs with D-MPNN and optional attention
        self.d_model = d_model  # Explanation: stores this value on the object for later model operations
        self.d_k = d_model // heads  # Explanation: stores this value on the object for later model operations
        self.h = heads  # Explanation: stores this value on the object for later model operations
        self.q_linear = nn.Linear(d_model, d_model, bias=False)  # Explanation: creates a learned linear projection
        self.v_linear = nn.Linear(d_model, d_model, bias=False)  # Explanation: creates a learned linear projection
        self.k_linear = nn.Linear(d_model, d_model, bias=False)  # Explanation: creates a learned linear projection
        self.dropout = nn.Dropout(dropout)  # Explanation: adds dropout regularization
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # Explanation: stores this value on the object for later model operations
        self.reset_parameters()  # Explanation: uses or updates this object state during computation

    def reset_parameters(self):  # Explanation: defines reset_parameters, which encode molecular graphs with D-MPNN and optional attention
        for p in self.parameters():  # Explanation: iterates over this collection to process each item
            if p.dim() > 1:  # Explanation: checks this condition to choose the next execution path
                nn.init.xavier_uniform_(p)  # Explanation: executes this statement as part of encode molecular graphs with D-MPNN and optional attention

    def attention(self, q, k, v, mask=None):  # Explanation: defines attention, which fuses atom features with functional-group embeddings
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # Explanation: assigns an intermediate value used by later computation
        if mask is not None:  # Explanation: checks this condition to choose the next execution path
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1)  # Explanation: assigns an intermediate value used by later computation
            mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1, 1)  # Explanation: assigns an intermediate value used by later computation
            scores[~mask.bool()] = float(-9e15)  # Explanation: assigns an intermediate value used by later computation
        scores = torch.softmax(scores, dim=-1)  # Explanation: assigns an intermediate value used by later computation
        scores = self.dropout(scores)  # Explanation: assigns an intermediate value used by later computation
        output = torch.matmul(scores, v)  # Explanation: assigns an intermediate value used by later computation
        return scores, output  # Explanation: returns this computed result to the caller

    def forward(self, x, mask=None):  # Explanation: defines forward, which encode molecular graphs with D-MPNN and optional attention
        bs = x.size(0)  # Explanation: assigns an intermediate value used by later computation
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)  # Explanation: assigns an intermediate value used by later computation
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)  # Explanation: assigns an intermediate value used by later computation
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)  # Explanation: assigns an intermediate value used by later computation
        k = k.transpose(1, 2)  # Explanation: assigns an intermediate value used by later computation
        q = q.transpose(1, 2)  # Explanation: assigns an intermediate value used by later computation
        v = v.transpose(1, 2)  # Explanation: assigns an intermediate value used by later computation
        scores, output = self.attention(q, k, v, mask)  # Explanation: assigns an intermediate value used by later computation
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)  # Explanation: assigns an intermediate value used by later computation
        output = output + x  # Explanation: assigns an intermediate value used by later computation
        output = self.layer_norm(output)  # Explanation: assigns an intermediate value used by later computation
        return scores, output.squeeze(-1)  # Explanation: returns this computed result to the caller


class FeedForward(nn.Module):  # Explanation: defines FeedForward, residual feed-forward attention block
    def __init__(self, d_model, dropout=0.1):  # Explanation: defines __init__, which encode molecular graphs with D-MPNN and optional attention
        super(FeedForward, self).__init__()  # Explanation: executes this statement as part of encode molecular graphs with D-MPNN and optional attention
        self.net = nn.Sequential(  # Explanation: stores this value on the object for later model operations
            nn.Linear(d_model, d_model*2),  # Explanation: creates a learned linear projection
            nn.SELU(),  # Explanation: adds SELU nonlinearity
            nn.Linear(d_model*2, d_model),  # Explanation: creates a learned linear projection
            nn.Dropout(dropout)  # Explanation: adds dropout regularization
        )  # Explanation: closes the current multi-line expression
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # Explanation: stores this value on the object for later model operations

    def forward(self, x):  # Explanation: defines forward, which encode molecular graphs with D-MPNN and optional attention
        output = self.net(x)  # Explanation: assigns an intermediate value used by later computation
        return self.layer_norm(x + output)  # Explanation: returns this computed result to the caller


class Global_Attention(nn.Module):  # Explanation: defines Global_Attention, optional global attention stack over atom embeddings
    def __init__(self, d_model, heads, n_layers=1, dropout=0.1):  # Explanation: defines __init__, which encode molecular graphs with D-MPNN and optional attention
        super(Global_Attention, self).__init__()  # Explanation: executes this statement as part of encode molecular graphs with D-MPNN and optional attention
        self.n_layers = n_layers  # Explanation: stores this value on the object for later model operations
        att_stack = []  # Explanation: assigns an intermediate value used by later computation
        pff_stack = []  # Explanation: assigns an intermediate value used by later computation
        for _ in range(n_layers):  # Explanation: iterates over this collection to process each item
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))  # Explanation: executes this statement as part of encode molecular graphs with D-MPNN and optional attention
            pff_stack.append(FeedForward(d_model, dropout))  # Explanation: executes this statement as part of encode molecular graphs with D-MPNN and optional attention
        self.att_stack = nn.ModuleList(att_stack)  # Explanation: stores this value on the object for later model operations
        self.pff_stack = nn.ModuleList(pff_stack)  # Explanation: stores this value on the object for later model operations

    def forward(self, x, mask):  # Explanation: defines forward, which encode molecular graphs with D-MPNN and optional attention
        scores = []  # Explanation: assigns an intermediate value used by later computation
        for n in range(self.n_layers):  # Explanation: iterates over this collection to process each item
            score, x = self.att_stack[n](x, mask)  # Explanation: assigns an intermediate value used by later computation
            x = self.pff_stack[n](x)  # Explanation: assigns an intermediate value used by later computation
            scores.append(score)  # Explanation: executes this statement as part of encode molecular graphs with D-MPNN and optional attention
        return scores, x  # Explanation: returns this computed result to the caller
