import os
import sys
import types
import fsspec
import functools
import numpy as np
import pandas as pd
import pickle as pkl
import datamol as dm
from collections import Counter
from collections.abc import MutableMapping
from typing import Sequence, Optional, Any


def flatten(seq: Sequence[Optional[Any]]):
    return [item for subseq in seq for item in subseq]


def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

def is_callable(func):
    """Check if the object is callable."""
    FUNCTYPES = (types.FunctionType, types.MethodType, functools.partial)
    return func and (isinstance(func, FUNCTYPES) or callable(func))



def merge_counters(counters, step=5, verbose=0):
    """Merge a list of counters into a single counter in parallel using a step size."""
    def merge_n_counters(x):
        return sum(x, Counter())

    
    if len(counters) == 0:
        return Counter()
    
    if len(counters) < step:
        return merge_n_counters(counters)
    
    n = len(counters)
    res = dm.parallelized(merge_n_counters, 
                          [counters[i:i+step] for i in range(0, n, step)], 
                          n_jobs=-1, progress=verbose, batch_size=1, )
    
    res = merge_counters(res)

    return res


def get_local_cache():
    fname = os.path.abspath(__file__)
    base ='/'.join(fname.split('/')[:-2])
    cache_dir=os.path.join(base, 'cache')
    return cache_dir


def get_misato_folder():
    fname = os.path.abspath(__file__)
    base ='/'.join(fname.split('/')[:-3])
    d = os.path.join(base, 'misato')
    if not os.path.exists(d):
        d = "gs://opendatasets/misato"
    return d


def get_remote_cache(cname, return_filesystem=False):
    if cname == "small_molecules":
        remote_cache = "gs://opendatasets/small_molecule_collections"
        rfs = fsspec.filesystem("gs")
    elif cname == "peptides":
        remote_cache = "gs://opendatasets/peptides"
        rfs = fsspec.filesystem("gs")
    else:
        raise NotImplementedError()
    
    if return_filesystem:
        return remote_cache, rfs
    return remote_cache 


def caching_wrapper(f, *args, local_path=None, remote_path=None, 
                    remote_filesystem=None, **kwargs):
    local_specified = local_path is not None
    remote_specified = remote_path is not None
    if local_specified and os.path.exists(local_path):
        with fsspec.open(local_path, "rb") as fd:
            res = pkl.load(fd)
    elif remote_specified and remote_filesystem.exists(remote_path):
        with fsspec.open(remote_path, "rb") as fd:
            res = pkl.load(fd)
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        res = f(*args, **kwargs)
        if local_path is not None:
            with fsspec.open(local_path, "wb") as fd:
                pkl.dump(res, fd)

        if remote_path is not None:
            with fsspec.open(remote_path, "wb") as fd:
                pkl.dump(res, fd)
    return res


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