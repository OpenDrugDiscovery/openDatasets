import os
import json
import tqdm
import fsspec
import pandas as pd
import numpy as np
import datamol as dm
from loguru import logger
from rdkit import Chem
from typing import Union
from openff.toolkit.typing.engines.smirnoff import ForceField
from odd_datasets.collections.base import ConformersDataset
from openff.toolkit.topology import Molecule, Topology


class NaturalPeptides(ConformersDataset):
    """Natural peptides dataset.

    This collection contains peptides from the UniProt database.
    The peptides are all at most 10 amino acids long and contain no
    disallowed amino acids (B, J, O, U, X, Z).

    """

    NAME = "odd_natural_peptides"
    TAGLINE = "Natural peptides from the UniProt database."
    DESCRIPTION = """This collection contains peptides from the UniProt database.
        The peptides are all at most 10 amino acids long and contain no
        disallowed amino acids (B, J, O, U, X, Z).
        """
    
    def __init__(self, cache_folder, reviewed_only=True, min_count=5, server_info_file: Union[str, None] = None):
        super().__init__(server_info_file)
        basename = f"peptides_set_{'reviewed' if reviewed_only else 'unreviewed'}_min_occ_{min_count}.json"
        self.input_file = os.path.join(cache_folder, basename)

    def sample_peptides(self, n=None):
        """
        Samples peptides from the dataset.
        
        Parameters
        ----------
        n: int or float or dict
            If int, the number of peptides to sample.
            If float, the fraction of peptides to sample.
            If dict, the number of peptides to sample per length.
            If None, all peptides are sampled.
            
        Returns
        -------
        peptides: list
            The sampled peptides.
        """
        with fsspec.open(self.input_file, "r") as fd:
            peptides = json.load(fd)

        n_peptides_per_length = {int(k): v for k, v in peptides["n_kmers"].items()}
        if n is None:
            n_peptides_per_length = {int(k): v for k, v in peptides["n_kmers"].items()}
        elif isinstance(n, float):
            n_peptides_per_length = {int(k): (int*v) for k, v in peptides["n_kmers"].items()}
        elif isinstance(n, int):
            ks = np.array(list(peptides["n_kmers"].keys()))
            vs = np.array(list(peptides["n_kmers"].values()))
            cumsum = np.cumsum(vs)
            idxs = np.argwhere(cumsum > n).flatten()
            vs = np.array(peptides["n_kmers"].values())
            vs[idxs] = 0
            vs[idxs[0]] = n - cumsum[idxs[0]-1]
            n_peptides_per_length = dict(zip(ks, vs))
        elif isinstance(n, dict):
            pass
        else:
            raise ValueError(f"Invalid value for n: {n}")
        
        selected_peptides = []
        for i in range(11): # 11 is the max length of a peptide
            selected_peptides += np.random.choice(a=peptides["kmers"][str(i)].keys(), 
                                                  p=peptides["kmers"][str(i)].values(),
                                                  size=n_peptides_per_length[i], 
                                                  replace=False).tolist()
        return selected_peptides
    

    def _generate_conformers_(self, n_peptides):
        """
        Generates conformers for the peptides.
        """
        def from_sequence(sequence):
            mol = Chem.MolFromSequence(sequence)
            if mol is not None:
                smi= dm.to_smiles(mol)
                mol = Molecule.from_smiles(smi, allow_undefined_stereo=False)
                nr = (len(mol.find_rotatable_bonds) + 3)**3
                mol.generate_conformers(n_conformers=nr)

            return mol
        
        peptides = self.sample_peptides(n=n_peptides)
        mols = dm.parallelized(
            from_sequence, 
            peptides, 
            n_jobs=-1, 
            progress=True)
        molecules = dict(zip(peptides, mols))
        logger.info(f"N molecules: {len(molecules)} .")
        return molecules
