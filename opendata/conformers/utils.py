import sys
import types
import functools
import pandas as pd
from collections.abc import MutableMapping
from typing import Sequence, Optional, Any


def rmsd_filtering(mol, threshold=0.1):
    """ Filter conformers based on RMSD threshold.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol or openff.toolkit.topology.Molecule
        The molecule which conformers will be filtered.

    threshold : float
        The RMSD threshold. unit is Angstrom.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol or openff.toolkit.topology.Molecule
        The molecule with filtered conformers.
    """

    is_opemm_mol = isinstance(mol, Molecule)
    if is_opemm_mol:
        mol = mol.to_rdkit()

    if mol.GetNumConformers() <= 1:
        raise ValueError(
            "The molecule has 0 or 1 conformer. You can generate conformers with `dm.conformers.generate(mol)`."
        )

    n_confs = mol.GetNumConformers()
    remaining_confs, keep_confs = list(range(n_confs)), []
    while len(remaining_confs) > 1:
        i = remaining_confs[0]
        tmp = np.array([rdMolAlign.AlignMol(prbMol=mol, refMol=mol, prbCid=i, refCid=j) 
                        for j in remaining_confs])
        remaining_confs = np.argwhere(tmp > threshold).flatten()
        keep_confs.append(i)

    mol =dm.conformers.keep_conformers(mol,indices_to_keep=keep_confs)
        
    assert np.all(dm.conformers.rmsd(mol) < threshold)

    if is_opemm_mol:
        mol = Molecule.from_rdkit(mol, allow_undefined_stereo=False)

    return mol
