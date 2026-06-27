import torch  # Explanation: imports torch for compute adaptive dynamic contrastive loss
import torch.nn.functional as F  # Explanation: imports torch.nn.functional as F for compute adaptive dynamic contrastive loss
from torch import nn  # Explanation: imports selected names needed to compute adaptive dynamic contrastive loss
import math  # Explanation: imports math for compute adaptive dynamic contrastive loss

__all__ = ['ADNCE', 'adnce']  # Explanation: assigns an intermediate value used by later computation


class ADNCE(nn.Module):  # Explanation: defines ADNCE, PyTorch module wrapper for adaptive contrastive loss

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', mu=0.2, sigma=1.0):  # Explanation: defines __init__, which compute adaptive dynamic contrastive loss
        super().__init__()  # Explanation: executes this statement as part of compute adaptive dynamic contrastive loss
        self.temperature = temperature  # Explanation: stores this value on the object for later model operations
        self.reduction = reduction  # Explanation: stores this value on the object for later model operations
        self.negative_mode = negative_mode  # Explanation: stores this value on the object for later model operations
        self.mu = mu  # Explanation: stores this value on the object for later model operations
        self.sigma = sigma  # Explanation: stores this value on the object for later model operations

    def forward(self, query, positive_key, negative_keys=None):  # Explanation: defines forward, which compute adaptive dynamic contrastive loss
        return adnce(query, positive_key, negative_keys,  # Explanation: returns this computed result to the caller
                     temperature=self.temperature,  # Explanation: assigns an intermediate value used by later computation
                     reduction=self.reduction,  # Explanation: assigns an intermediate value used by later computation
                     negative_mode=self.negative_mode,  # Explanation: assigns an intermediate value used by later computation
                     mu=self.mu,  # Explanation: assigns an intermediate value used by later computation
                     sigma=self.sigma)  # Explanation: assigns an intermediate value used by later computation


def adnce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired', mu=0.1,  # Explanation: defines adnce, which computes weighted contrastive cross-entropy
          sigma=1.0):  # Explanation: assigns an intermediate value used by later computation
    # Check input dimensionality
    if query.dim() != 2:  # Explanation: checks this condition to choose the next execution path
        raise ValueError('<query> must have 2 dimensions')  # Explanation: raises an error when unsupported input is encountered
    if positive_key.dim() != 2:  # Explanation: checks this condition to choose the next execution path
        raise ValueError('<positive_key> must have 2 dimensions')  # Explanation: raises an error when unsupported input is encountered
    if negative_keys is not None:  # Explanation: checks this condition to choose the next execution path
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:  # Explanation: checks this condition to choose the next execution path
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'。")  # Explanation: raises an error when unsupported input is encountered
        if negative_mode == 'paired' and negative_keys.dim() != 3:  # Explanation: checks this condition to choose the next execution path
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'paired'。")  # Explanation: raises an error when unsupported input is encountered

    # Check sample number matching
    if len(query) != len(positive_key):  # Explanation: checks this condition to choose the next execution path
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')  # Explanation: raises an error when unsupported input is encountered
    if negative_keys is not None:  # Explanation: checks this condition to choose the next execution path
        if negative_mode == 'paired' and len(query) != len(negative_keys):  # Explanation: checks this condition to choose the next execution path
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")  # Explanation: raises an error when unsupported input is encountered

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:  # Explanation: checks this condition to choose the next execution path
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')  # Explanation: raises an error when unsupported input is encountered
    if negative_keys is not None:  # Explanation: checks this condition to choose the next execution path
        if query.shape[-1] != negative_keys.shape[-1]:  # Explanation: checks this condition to choose the next execution path
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')  # Explanation: raises an error when unsupported input is encountered

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)  # Explanation: assigns an intermediate value used by later computation
    if negative_keys is not None:  # Explanation: checks this condition to choose the next execution path

        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)  # Explanation: assigns an intermediate value used by later computation

        if negative_mode == 'unpaired':  # Explanation: checks this condition to choose the next execution path
            negative_logits = query @ transpose(negative_keys)  # Explanation: assigns an intermediate value used by later computation

        elif negative_mode == 'paired':  # Explanation: checks an alternate condition after the previous branch failed
            query = query.unsqueeze(1)  # Explanation: assigns an intermediate value used by later computation
            negative_logits = query @ transpose(negative_keys)  # Explanation: assigns an intermediate value used by later computation
            negative_logits = negative_logits.squeeze(1)  # Explanation: assigns an intermediate value used by later computation

        # apply weight
        weight = (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((negative_logits - mu) / sigma) ** 2)  # Explanation: assigns an intermediate value used by later computation
        weight = weight / weight.mean(dim=-1, keepdim=True)  # Explanation: assigns an intermediate value used by later computation
        negative_logits = negative_logits * weight.detach()  # Explanation: assigns an intermediate value used by later computation

        logits = torch.cat([positive_logit, negative_logits], dim=1)  # Explanation: concatenates tensors along an existing dimension
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)  # Explanation: assigns an intermediate value used by later computation

    else:  # Explanation: handles the fallback branch for the preceding condition
        logits = query @ transpose(positive_key)  # Explanation: assigns an intermediate value used by later computation

        labels = torch.arange(len(query), device=query.device)  # Explanation: assigns an intermediate value used by later computation

        temp_logits = logits.clone()  # Explanation: assigns an intermediate value used by later computation
        neg_logits = temp_logits.fill_diagonal_(0)  # Explanation: assigns an intermediate value used by later computation

        # apply weight
        weight = (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((neg_logits - mu) / sigma) ** 2)  # Explanation: assigns an intermediate value used by later computation
        weight = weight / weight.mean(dim=-1, keepdim=True)  # Explanation: assigns an intermediate value used by later computation

        adjusted_logits = logits * weight.detach()  # Explanation: assigns an intermediate value used by later computation

        diagonal = torch.diag(logits)  # Explanation: assigns an intermediate value used by later computation
        adjusted_logits = adjusted_logits.clone()  # Explanation: assigns an intermediate value used by later computation
        diag_matrix = torch.diag_embed(diagonal)  # Explanation: assigns an intermediate value used by later computation
        adjusted_logits.fill_diagonal_(0)  # Explanation: executes this statement as part of compute adaptive dynamic contrastive loss
        adjusted_logits += diag_matrix  # Explanation: assigns an intermediate value used by later computation

        logits = adjusted_logits  # Explanation: assigns an intermediate value used by later computation

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)  # Explanation: returns this computed result to the caller


def transpose(x):  # Explanation: defines transpose, which transposes tensor dimensions for similarity computation
    return x.transpose(-2, -1)  # Explanation: returns this computed result to the caller


def normalize(*xs):  # Explanation: defines normalize, which normalizes embeddings before contrastive loss
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]  # Explanation: returns this computed result to the caller
