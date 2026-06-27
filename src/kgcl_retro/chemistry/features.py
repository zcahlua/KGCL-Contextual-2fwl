from typing import Any, List, Set, Union  # Explanation: imports selected names needed to construct atom and bond feature vectors
from rdkit import Chem  # Explanation: imports selected names needed to construct atom and bond feature vectors

ATOM_SYMBOL_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', 'Si', 'P', 'B', 'I', 'Li', 'Na', 'K', 'Ca',  # Explanation: defines a module-level constant used by the pipeline
                    'Mg', 'Al', 'Cu', 'Zn', 'Sn', 'Se', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'As', 'Bi', 'Te', 'Sb',  # Explanation: continues the current multi-line argument or data structure
                    'Ba', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Pt', 'Au', 'Pb', 'Cs', 'Sm', 'Os', 'Ir', '*', 'unk']  # Explanation: executes this statement as part of construct atom and bond feature vectors

DEGREES = list(range(10))  # Explanation: defines a module-level constant used by the pipeline
FORMAL_CHARGE = [-1, -2, 1, 2, 0]  # Explanation: defines a module-level constant used by the pipeline
VALENCE = [0, 1, 2, 3, 4, 5, 6]  # Explanation: defines a module-level constant used by the pipeline
NUM_Hs = [0, 1, 2, 3, 4]  # Explanation: assigns an intermediate value used by later computation
CHIRALTAG = [0, 1, 2, 3]  # Explanation: defines a module-level constant used by the pipeline
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,  # Explanation: defines a module-level constant used by the pipeline
                 Chem.rdchem.HybridizationType.SP2,  # Explanation: continues the current multi-line argument or data structure
                 Chem.rdchem.HybridizationType.SP3,  # Explanation: continues the current multi-line argument or data structure
                 Chem.rdchem.HybridizationType.SP3D,  # Explanation: continues the current multi-line argument or data structure
                 Chem.rdchem.HybridizationType.SP3D2]  # Explanation: executes this statement as part of construct atom and bond feature vectors

BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,  # Explanation: defines a module-level constant used by the pipeline
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]  # Explanation: executes this statement as part of construct atom and bond feature vectors
BONDSTEREO = list(range(6))  # Explanation: defines a module-level constant used by the pipeline

RXN_CLASSES = list(range(10))  # Explanation: defines a module-level constant used by the pipeline
ATOM_FDIM = (  # Explanation: starts the full atom feature dimension calculation.
    len(ATOM_SYMBOL_LIST) + len(DEGREES) + len(FORMAL_CHARGE) +  # Explanation: counts symbol, degree, and charge feature slots.
    len(VALENCE) + len(NUM_Hs) + len(CHIRALTAG) + len(HYBRIDIZATION) + 1  # Explanation: adds valence, hydrogen, chirality, hybridization, and aromatic slots.
)  # Explanation: closes the atom feature dimension calculation.
BOND_FDIM = len(BOND_TYPES) + len(BONDSTEREO) + 2  # Explanation: defines a module-level constant used by the pipeline


def one_of_k_encoding(x: Any, allowable_set: Union[List, Set]) -> List:  # Explanation: defines one_of_k_encoding, which builds one-hot chemical features
    """Converts x to one hot encoding.

    Parameters
    ----------
    x: Any,
        An element of any type
    allowable_set: Union[List, Set]
        Allowable element collection
    """
    if x not in allowable_set:  # Explanation: checks this condition to choose the next execution path
        x = allowable_set[-1]  # Explanation: assigns an intermediate value used by later computation
    return list(map(lambda s: float(x == s), allowable_set))  # Explanation: returns this computed result to the caller


def get_atom_features(atom: Chem.Atom, rxn_class: int = None, use_rxn_class: bool = False) -> List[  # Explanation: defines get_atom_features, which constructs atom feature vectors
    Union[bool, int, float]]:  # Explanation: executes this statement as part of construct atom and bond feature vectors
    """Get atom features.

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit
    rxn_class: int, None
        Reaction class the molecule was part of
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    """
    if use_rxn_class:  # Explanation: checks this condition to choose the next execution path
        atom_features = (  # Explanation: starts the reaction-class-aware atom feature vector.
            one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL_LIST) +  # Explanation: encodes the atom element symbol.
            one_of_k_encoding(atom.GetDegree(), DEGREES) +  # Explanation: encodes the atom degree.
            one_of_k_encoding(atom.GetFormalCharge(), FORMAL_CHARGE) +  # Explanation: encodes the formal charge.
            one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION) +  # Explanation: encodes the RDKit hybridization state.
            one_of_k_encoding(atom.GetTotalValence(), VALENCE) +  # Explanation: encodes total valence.
            one_of_k_encoding(atom.GetTotalNumHs(), NUM_Hs) +  # Explanation: encodes attached hydrogens.
            one_of_k_encoding(int(atom.GetChiralTag()), CHIRALTAG) +  # Explanation: encodes atom chirality.
            [atom.GetIsAromatic()] + one_of_k_encoding(rxn_class, RXN_CLASSES)  # Explanation: appends aromaticity and reaction-class features.
        )  # Explanation: closes the reaction-class-aware atom feature vector.

        return atom_features  # Explanation: returns this computed result to the caller

    else:  # Explanation: handles the fallback branch for the preceding condition
        atom_features = (  # Explanation: starts the base atom feature vector without reaction class.
            one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL_LIST) +  # Explanation: encodes the atom element symbol.
            one_of_k_encoding(atom.GetDegree(), DEGREES) +  # Explanation: encodes the atom degree.
            one_of_k_encoding(atom.GetFormalCharge(), FORMAL_CHARGE) +  # Explanation: encodes the formal charge.
            one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION) +  # Explanation: encodes the RDKit hybridization state.
            one_of_k_encoding(atom.GetTotalValence(), VALENCE) +  # Explanation: encodes total valence.
            one_of_k_encoding(atom.GetTotalNumHs(), NUM_Hs) +  # Explanation: encodes attached hydrogens.
            one_of_k_encoding(int(atom.GetChiralTag()), CHIRALTAG) +  # Explanation: encodes atom chirality.
            [atom.GetIsAromatic()]  # Explanation: appends the aromaticity flag.
        )  # Explanation: closes the base atom feature vector.

        return atom_features  # Explanation: returns this computed result to the caller


def get_bond_features(bond: Chem.Bond) -> List[Union[bool, int, float]]:  # Explanation: defines get_bond_features, which constructs bond feature vectors
    """
    Get bond features.
    """
    bond_features = (  # Explanation: starts the bond feature vector.
        one_of_k_encoding(bond.GetBondType(), BOND_TYPES) +  # Explanation: encodes the RDKit bond type.
        one_of_k_encoding(int(bond.GetStereo()), BONDSTEREO) +  # Explanation: encodes the bond stereo state.
        [bond.GetIsConjugated()] + [bond.IsInRing()]  # Explanation: appends conjugation and ring membership flags.
    )  # Explanation: closes the bond feature vector.

    return bond_features  # Explanation: returns this computed result to the caller
