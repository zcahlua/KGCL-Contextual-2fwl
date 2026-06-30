import csv  # Explanation: imports csv for provide tensor utilities and logging helpers
from dataclasses import fields, is_dataclass, replace
import torch  # Explanation: imports torch for provide tensor utilities and logging helpers
from torch.nn.utils.rnn import pad_sequence  # Explanation: imports selected names needed to provide tensor utilities and logging helpers


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:  # Explanation: defines index_select_ND, which selects neighbor/message tensors by index
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.
    Parameters
    ----------
    source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)  # Explanation: assigns an intermediate value used by later computation
    suffix_dim = source.size()[1:]  # (hidden_size,)  # Explanation: assigns an intermediate value used by later computation
    # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    final_size = index_size + suffix_dim  # Explanation: assigns an intermediate value used by later computation

    # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = source.index_select(dim=0, index=index.view(-1))  # Explanation: assigns an intermediate value used by later computation
    # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    target = target.view(final_size)  # Explanation: assigns an intermediate value used by later computation

    return target  # Explanation: returns this computed result to the caller


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        updates = {field.name: move_to_device(getattr(obj, field.name), device) for field in fields(obj)}
        return replace(obj, **updates)
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    return obj


def creat_edits_feats(atom_feats, atom_scope):  # Explanation: defines creat_edits_feats, which pads atom embeddings for optional attention
    a_feats = []  # Explanation: assigns an intermediate value used by later computation
    masks = []  # Explanation: assigns an intermediate value used by later computation

    for idx, (st_a, le_a) in enumerate(atom_scope):  # Explanation: iterates over this collection to process each item
        feats = atom_feats[st_a: st_a + le_a]  # Explanation: assigns an intermediate value used by later computation
        mask = torch.ones(feats.size(0), dtype=torch.uint8)  # Explanation: assigns an intermediate value used by later computation
        a_feats.append(feats)  # Explanation: executes this statement as part of provide tensor utilities and logging helpers
        masks.append(mask)  # Explanation: executes this statement as part of provide tensor utilities and logging helpers

    a_feats = pad_sequence(a_feats, batch_first=True, padding_value=0)  # Explanation: assigns an intermediate value used by later computation
    masks = pad_sequence(masks, batch_first=True, padding_value=0)  # Explanation: assigns an intermediate value used by later computation

    return a_feats, masks  # Explanation: returns this computed result to the caller


def unbatch_feats(feats, atom_scope):  # Explanation: defines unbatch_feats, which restores padded atom embeddings to flat format
    atom_feats = []  # Explanation: computes an intermediate value for molecular graph editing

    for idx, (st_a, le_a) in enumerate(atom_scope):  # Explanation: iterates over this collection to process each item
        atom_feats.append(feats[idx][:le_a])  # Explanation: executes this statement as part of provide tensor utilities and logging helpers

    a_feats = torch.cat(atom_feats, dim=0)  # Explanation: concatenates tensors along an existing dimension

    pad_tensor = torch.zeros(1, a_feats.size(1), device=a_feats.device)  # Explanation: assigns an intermediate value used by later computation
    return torch.cat((pad_tensor, a_feats), dim=0)  # Explanation: returns this computed result to the caller


def get_seq_edit_accuracy(seq_edit_scores, seq_labels, seq_mask):  # Explanation: defines get_seq_edit_accuracy, which measures full edit-sequence accuracy
    max_seq_len = seq_mask.shape[0]  # Explanation: assigns an intermediate value used by later computation
    batch_size = seq_mask.shape[1]  # Explanation: assigns an intermediate value used by later computation
    assert len(seq_edit_scores) == max_seq_len  # Explanation: checks an invariant expected by the model pipeline
    assert len(seq_labels) == max_seq_len  # Explanation: checks an invariant expected by the model pipeline
    assert len(seq_edit_scores[0]) == batch_size  # Explanation: checks an invariant expected by the model pipeline
    lengths = seq_mask.sum(dim=0).flatten()  # Explanation: assigns an intermediate value used by later computation

    def check_equals(x, y): return torch.argmax(x) == torch.argmax(y)  # Explanation: defines check_equals, which provide tensor utilities and logging helpers

    all_acc = 0  # Explanation: assigns an intermediate value used by later computation
    for batch_id in range(batch_size):  # Explanation: iterates over this collection to process each item
        step_acc = 0  # Explanation: assigns an intermediate value used by later computation
        seq_length = lengths[batch_id]  # Explanation: computes an intermediate value for molecular graph editing
        for idx in range(seq_length):  # Explanation: iterates over this collection to process each item
            if check_equals(seq_edit_scores[idx][batch_id], seq_labels[idx][batch_id]):  # Explanation: checks this condition to choose the next execution path
                step_acc += 1  # Explanation: assigns an intermediate value used by later computation

        if step_acc == seq_length:  # Explanation: checks this condition to choose the next execution path
            all_acc += 1  # Explanation: assigns an intermediate value used by later computation

    accuracy = all_acc / batch_size  # Explanation: assigns an intermediate value used by later computation
    return accuracy  # Explanation: returns this computed result to the caller


class CSVLogger():  # Explanation: defines CSVLogger, CSV logger for training metrics
    def __init__(self, args, fieldnames, filename='log.csv'):  # Explanation: defines __init__, which provide tensor utilities and logging helpers

        self.filename = filename  # Explanation: stores this value on the object for later model operations
        self.csv_file = open(filename, 'w')  # Explanation: stores this value on the object for later model operations

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)  # Explanation: assigns an intermediate value used by later computation
        for arg, arg_val in args.items():  # Explanation: iterates over this collection to process each item
            writer.writerow([arg, arg_val])  # Explanation: executes this statement as part of provide tensor utilities and logging helpers
        # for arg in vars(args):
        #     writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])  # Explanation: executes this statement as part of provide tensor utilities and logging helpers

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)  # Explanation: stores this value on the object for later model operations
        self.writer.writeheader()  # Explanation: uses or updates this object state during computation

        self.csv_file.flush()  # Explanation: uses or updates this object state during computation

    def writerow(self, row):  # Explanation: defines writerow, which provide tensor utilities and logging helpers
        self.writer.writerow(row)  # Explanation: uses or updates this object state during computation
        self.csv_file.flush()  # Explanation: uses or updates this object state during computation

    def close(self):  # Explanation: defines close, which provide tensor utilities and logging helpers
        self.csv_file.close()  # Explanation: uses or updates this object state during computation
