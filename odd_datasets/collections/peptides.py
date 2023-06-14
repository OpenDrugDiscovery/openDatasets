import os
import json
import time
import tqdm
import fsspec
import logging
import pandas as pd
import numpy as np
import datamol as dm
from loguru import logger
from rdkit import Chem
from typing import Union
import qcengine as qcng
import qcelemental as qcel
import openmm.unit as unit
from rdkit.Chem import rdMolAlign
from odd_datasets.collections.utils import flatten_dict
from openff.toolkit.typing.engines.smirnoff import ForceField
from odd_datasets.collections.base import QCFractalOptDataset
from openff.toolkit.topology import Molecule, Topology
from rdkit.Chem.MolStandardize import rdMolStandardize
from openff.qcsubmit.common_structures import QCSpec, SCFProperties

logging.getLogger('openff').setLevel(logging.ERROR)

def peptide_to_mol(sequence):
    mol = Chem.MolFromSequence(sequence)
    mol = dm.standardize_mol(mol)
    if "R" in sequence:
        t0 = time.time()
        mols = dm.enumerate_tautomers(
            mol, 
            n_variants=max(100, 4**len(sequence)),
            remove_bond_stereo=False, 
            remove_sp3_stereo=False, 
            reassign_stereo=True)
        mols = [dm.standardize_mol(mol) for mol in mols]
        mols = [dm.add_hs(mol) for mol in mols]
        mol = None
        for mol in mols:
            try:
                Molecule.from_rdkit(mol, allow_undefined_stereo=False)
                break
            except:
                pass
        logger.info(f"Tautomers enumeration took {time.time() - t0} seconds.") 

    if mol is not None:     
        nr = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        nc = int((nr + 2)**2.5)
        logger.info(f"Generating {nc} conformers for {sequence} that has {nr} rotatable bonds.")
        t0 = time.time()
        mol = dm.conformers.generate(mol, n_confs=nc, minimize_energy=False, sort_by_energy=False)
        mol = Molecule.from_rdkit(mol, allow_undefined_stereo=False)
        logger.info(f"Conformer generation took {time.time() - t0} seconds to generate {mol.n_conformers}.")
    
    return mol





class NaturalPeptides(QCFractalOptDataset):
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
    
    def __init__(self, cache_folder, reviewed_only=True, min_count=5, server_info_file: Union[str, None] = None, debug: bool = False):
        super().__init__(server_info_file)
        basename = f"peptides_set_{'reviewed' if reviewed_only else 'unreviewed'}_min_occ_{min_count}.json"
        self.input_file = os.path.join(cache_folder, "uniprotkb", basename)
        self.debug = debug

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

        if n is None:
            n_peptides_per_length = {int(k): v for k, v in peptides["n_kmers"].items()}
        elif isinstance(n, float):
            n_peptides_per_length = {int(k): (int*v) for k, v in peptides["n_kmers"].items()}
        elif isinstance(n, int):
            ks = np.array(list(peptides["n_kmers"].keys()))
            vs = np.array(list(peptides["n_kmers"].values()))
            cumsum = np.cumsum(vs)
            idxs = np.argwhere(cumsum > n).flatten()
            vs = np.array(list(peptides["n_kmers"].values()))
            vs[idxs] = 0
            vs[idxs[0]] = n - cumsum[idxs[0]-1]
            n_peptides_per_length = dict(zip(ks, vs))
        elif isinstance(n, dict):
            n_peptides_per_length = {int(k): v for k, v in peptides["n_kmers"].items()}
        else:
            raise ValueError(f"Invalid value for n: {n}")
        print(n_peptides_per_length)
        
        selected_peptides = []
        for i in range(10): # 11 is the max length of a peptide
            key = str(i+1)
            probs = np.array(list(peptides[key].values())) / sum(list(peptides[key].values()))
            selected_peptides += np.random.choice(a=list(peptides[key].keys()), 
                                                  p=probs,
                                                  size=n_peptides_per_length[key], 
                                                  replace=False).tolist()
        return selected_peptides
    

    def generate_conformers(self, n_peptides, return_peptides=False):
        """
        Generates conformers for the peptides.
        """
        
        peptides = self.sample_peptides(n=n_peptides)
        mols = dm.parallelized(
            peptide_to_mol, 
            peptides, 
            n_jobs=1, 
            progress=True)
        molecules = dict(zip(peptides, mols))
        logger.info(f"N molecules: {len(molecules)} .")
        if return_peptides:
            return molecules, peptides
        return molecules


if __name__ == "__main__":
    w = "N=C(N)NCCC[C@H](N)C(=O)O"
    mol = peptide_to_mol("K")
    molecule_geometry_optimization(mol)
    exit()

    fpath = os.path.abspath(__file__)
    cache_folder = "/".join(fpath.split("/")[:-3]) + "/cache"
    dataset = NaturalPeptides(cache_folder=cache_folder, debug=True)
    molecules, peptides = dataset.generate_conformers(n_peptides=100, return_peptides=True)
    print(peptides)
    exit()
    res = dataset.submit(
        molecules, 
        collection_name="odd_natural_peptides_test", 
        tagline="Natural peptides from the UniProt database.", 
        description="This collection contains peptides from the UniProt database."
    )
    print(res)
