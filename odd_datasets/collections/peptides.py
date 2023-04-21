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
from odd_datasets.collections.base import QCFractalOptDataset
from openff.toolkit.topology import Molecule, Topology


def peptide_to_mol(sequence):
    aa_smart = Chem.MolFromSmarts('NCC(=O)')

    mol = Chem.MolFromSequence(sequence)
    print(dm.to_smiles(mol))
    if mol is not None:
        # smi= dm.to_smiles(mol, canonical=True, isomeric=True)
        # stereo_candidates = dm.enumerate_stereoisomers(mol, n_variants=10000)
        # print(sequence, smi, len(stereo_candidates))
        # for candidate in stereo_candidates:
        #     m = dm.to_mol(candidate)
        #     alphas =  [m.GetAtomWithIdx(aa[1]) for aa in m.GetSubstructMatches(aa_smart)]
        #     alphas = [alpha for alpha in alphas if alpha.HasProp('_CIPCode')]
        #     print([alpha.GetProp('_CIPCode') for alpha in alphas])
        #     print(len(alphas))
        #     if all([(alpha.GetProp('_CIPCode') == "S") for alpha in alphas]):
        #         Chem.rdCIPLabeler.AssignCIPLabels(mol)
        #         print(smi)
        #         smi= dm.to_smiles(dm.standardize_mol(mol), canonical=True, isomeric=True, explicit_bonds=True)
        #         # dm.standardize_smiles(smi)

        #         break
        Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True)
        C
        mol = Molecule.from_rdkit(mol, allow_undefined_stereo=False)
        nr = (len(mol.find_rotatable_bonds()) + 3)**3
        mol.generate_conformers(n_conformers=nr)

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
    peptide_to_mol("AR")
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
