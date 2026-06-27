from importlib import import_module  # Explanation: imports import_module for lazy public API resolution.

_EXPORT_MODULES = {  # Explanation: maps public chemistry names to the submodule that defines each name.
    "AddGroupAction": "kgcl_retro.chemistry.actions",  # Explanation: resolves leaving-group attachment action lazily.
    "AtomEditAction": "kgcl_retro.chemistry.actions",  # Explanation: resolves atom edit action lazily.
    "BondEditAction": "kgcl_retro.chemistry.actions",  # Explanation: resolves bond edit action lazily.
    "ReactionAction": "kgcl_retro.chemistry.actions",  # Explanation: resolves the edit action base class lazily.
    "Termination": "kgcl_retro.chemistry.actions",  # Explanation: resolves the edit sequence termination action lazily.
    "ATOM_FDIM": "kgcl_retro.chemistry.features",  # Explanation: resolves atom feature dimension lazily.
    "BOND_FDIM": "kgcl_retro.chemistry.features",  # Explanation: resolves bond feature dimension lazily.
    "get_atom_features": "kgcl_retro.chemistry.features",  # Explanation: resolves atom featurization lazily.
    "get_bond_features": "kgcl_retro.chemistry.features",  # Explanation: resolves bond featurization lazily.
}  # Explanation: closes the lazy export mapping.

__all__ = list(_EXPORT_MODULES)  # Explanation: exposes the same public chemistry names for wildcard imports.


def __getattr__(name):  # Explanation: loads a public chemistry symbol only when a caller asks for it.
    if name not in _EXPORT_MODULES:  # Explanation: rejects unknown symbols with Python's normal attribute error behavior.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # Explanation: reports invalid chemistry package attributes.
    module = import_module(_EXPORT_MODULES[name])  # Explanation: imports the defining submodule lazily.
    value = getattr(module, name)  # Explanation: retrieves the requested public symbol from that submodule.
    globals()[name] = value  # Explanation: caches the resolved symbol on this package module.
    return value  # Explanation: returns the resolved public chemistry symbol.
