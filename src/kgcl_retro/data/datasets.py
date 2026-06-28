import os  # Explanation: imports os for load processed training and evaluation data
from typing import List, Optional, Tuple  # Explanation: imports selected names needed to load processed training and evaluation data

import joblib  # Explanation: imports joblib for load processed training and evaluation data

import torch  # Explanation: imports torch for load processed training and evaluation data
from torch.utils.data import DataLoader, Dataset  # Explanation: imports selected names needed to load processed training and evaluation data
from kgcl_retro.chemistry.edits import ReactionData  # Explanation: imports the packaged reaction data tuple used by processed datasets.
from kgcl_retro.config.schema import normalize_model_variant
from kgcl_retro.data.collate import ContextualEditTarget, GraphBatch


class RetroEditDataset(Dataset):  # Explanation: defines RetroEditDataset, dataset for saved training tensor batches
    def __init__(self, data_dir: str, **kwargs):  # Explanation: defines __init__, which load processed training and evaluation data
        self.data_dir = data_dir  # Explanation: stores this value on the object for later model operations
        self.model_variant = normalize_model_variant(kwargs.pop("model_variant", "kgcl"))
        self.data_files = [  # Explanation: stores this value on the object for later model operations
            os.path.join(self.data_dir, file)  # Explanation: builds a filesystem path
            for file in os.listdir(self.data_dir)  # Explanation: iterates over this collection to process each item
            if "batch-" in file  # Explanation: checks this condition to choose the next execution path
        ]  # Explanation: closes the current multi-line expression
        self.__dict__.update(**kwargs)  # Explanation: uses or updates this object state during computation

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:  # Explanation: defines __getitem__, which load processed training and evaluation data
        """Retrieves a particular batch of tensors.

        Parameters
        ----------
        idx: int,
            Batch index
        """
        batch_tensors = torch.load(self.data_files[idx], map_location='cpu')  # Explanation: loads saved tensor batches or checkpoints

        return batch_tensors  # Explanation: returns this computed result to the caller

    def __len__(self) -> int:  # Explanation: defines __len__, which load processed training and evaluation data
        """Returns length of the Dataset."""
        return len(self.data_files)  # Explanation: returns this computed result to the caller

    def collate(self, attributes: List[Tuple[torch.tensor]]) -> Tuple[torch.Tensor]:  # Explanation: defines collate, which load processed training and evaluation data
        """Processes the batch of tensors to yield corresponding inputs."""
        assert isinstance(attributes, list)  # Explanation: checks an invariant expected by the model pipeline
        assert len(attributes) == 1  # Explanation: checks an invariant expected by the model pipeline

        attributes = attributes[0]  # Explanation: assigns an intermediate value used by later computation
        graph_seq_tensors, edit_seq_labels, seq_mask = attributes  # Explanation: computes an intermediate value for molecular graph editing
        if self.model_variant == "contextual_2fwl" and (
            not graph_seq_tensors or not isinstance(graph_seq_tensors[0], GraphBatch)
        ):
            raise ValueError(
                "Prepared data lacks sparse pair metadata / gold action metadata. Re-run prepare_data.py "
                "with --model_variant contextual_2fwl."
            )
        if self.model_variant == "contextual_2fwl":
            first_step_labels = edit_seq_labels[0] if edit_seq_labels else []
            if not first_step_labels or not isinstance(first_step_labels[0], ContextualEditTarget):
                raise ValueError(
                    "Prepared data lacks sparse pair metadata / gold action metadata. Re-run prepare_data.py "
                    "with --model_variant contextual_2fwl."
                )
        return graph_seq_tensors, edit_seq_labels, seq_mask  # Explanation: returns this computed result to the caller

    def loader(self, batch_size: int, num_workers: int = 6, shuffle: bool = False) -> torch.utils.data.DataLoader:  # Explanation: defines loader, which load processed training and evaluation data
        """Creates a DataLoader from given batches."""
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate)  # Explanation: returns this computed result to the caller


class RetroEvalDataset(Dataset):  # Explanation: defines RetroEvalDataset, dataset for processed evaluation reactions
    def __init__(self, data_dir: str, data_file: str, use_rxn_class: bool = False):  # Explanation: defines __init__, which load processed training and evaluation data
        self.data_dir = data_dir  # Explanation: stores this value on the object for later model operations
        self.data_file = os.path.join(data_dir, data_file)  # Explanation: builds a filesystem path
        self.use_rxn_class = use_rxn_class  # Explanation: stores this value on the object for later model operations
        self.dataset = joblib.load(self.data_file)  # Explanation: loads processed dataset or vocabulary objects

    def __getitem__(self, idx: int) -> ReactionData:  # Explanation: defines __getitem__, which load processed training and evaluation data
        """Retrieves the corresponding ReactionData

        Parameters
        ----------
        idx: int,
        Index of particular element
        """
        return self.dataset[idx]  # Explanation: returns this computed result to the caller

    def __len__(self) -> int:  # Explanation: defines __len__, which load processed training and evaluation data
        """Returns length of the Dataset."""
        return len(self.dataset)  # Explanation: returns this computed result to the caller

    def collate(self, attributes: List[ReactionData]) -> Tuple[str, List[Tuple], List[List], Optional[List[int]]]:  # Explanation: defines collate, which load processed training and evaluation data
        """Processes the batch of tensors to yield corresponding inputs."""
        rxns_batch = attributes  # Explanation: assigns an intermediate value used by later computation
        prod_smi = [rxn_data.rxn_smi.split(">>")[-1]  # Explanation: computes an intermediate value for molecular graph editing
                    for rxn_data in rxns_batch]  # Explanation: iterates over this collection to process each item
        edits = [rxn_data.edits for rxn_data in rxns_batch]  # Explanation: assigns an intermediate value used by later computation
        edits_atom = [rxn_data.edits_atom for rxn_data in rxns_batch]  # Explanation: assigns an intermediate value used by later computation

        if self.use_rxn_class:  # Explanation: checks this condition to choose the next execution path
            rxn_classes = [rxn_data.rxn_class for rxn_data in rxns_batch]  # Explanation: computes an intermediate value for molecular graph editing
            return prod_smi, edits, edits_atom, rxn_classes  # Explanation: returns this computed result to the caller
        else:  # Explanation: handles the fallback branch for the preceding condition
            return prod_smi, edits, edits_atom, None  # Explanation: returns this computed result to the caller

    def loader(self, batch_size: int, num_workers: int = 6, shuffle: bool = False) -> DataLoader:  # Explanation: defines loader, which load processed training and evaluation data
        """Creates a DataLoader from given batches."""
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate)  # Explanation: returns this computed result to the caller
