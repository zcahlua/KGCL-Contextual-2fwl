from __future__ import annotations  # Explanation: postpones annotation evaluation for external chemistry types.

from typing import Any, Tuple  # Explanation: imports generic edit tuple typing used by model and preprocessing code.

from rdkit import Chem  # Explanation: imports RDKit molecule types for edit application.

from kgcl_retro.chemistry.actions import (  # Explanation: imports packaged graph edit action classes.
    AddGroupAction,  # Explanation: applies leaving-group attachment edits.
    AtomEditAction,  # Explanation: applies atom hydrogen and chirality edits.
    BondEditAction,  # Explanation: applies bond deletion, addition, and type changes.
)  # Explanation: closes the grouped edit action import.


def apply_edit_to_mol(mol: Chem.Mol, edit: Tuple, edit_atom: Any) -> Chem.Mol:  # Explanation: applies one predicted or labeled graph edit to an RDKit molecule.
    if edit[0] == "Change Atom":  # Explanation: handles atom-feature edits such as hydrogen or chirality changes.
        return AtomEditAction(edit_atom, *edit[1], action_vocab="Change Atom").apply(mol)  # Explanation: executes the atom edit and returns the edited molecule.
    if edit[0] == "Delete Bond":  # Explanation: handles edits that remove an existing product bond.
        return BondEditAction(*edit_atom, *edit[1], action_vocab="Delete Bond").apply(mol)  # Explanation: executes the bond deletion and returns the edited molecule.
    if edit[0] == "Change Bond":  # Explanation: handles edits that change bond order or stereochemistry.
        return BondEditAction(*edit_atom, *edit[1], action_vocab="Change Bond").apply(mol)  # Explanation: executes the bond update and returns the edited molecule.
    if edit[0] == "Add Bond":  # Explanation: handles edits that add a reactant-side bond between mapped atoms.
        return BondEditAction(*edit_atom, *edit[1], action_vocab="Add Bond").apply(mol)  # Explanation: executes the bond addition and returns the edited molecule.
    if edit[0] == "Attaching LG":  # Explanation: handles edits that attach a leaving group fragment.
        return AddGroupAction(edit_atom, edit[1], action_vocab="Attaching LG").apply(mol)  # Explanation: executes leaving-group attachment and returns the edited molecule.
    raise ValueError(f"Unsupported edit action: {edit}")  # Explanation: fails clearly when an unknown edit tuple reaches this shared helper.
