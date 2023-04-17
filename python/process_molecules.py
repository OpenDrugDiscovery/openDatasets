import csv

import datamol as dm

from typing import List
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from argparse import ArgumentParser
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers, 
    StereoEnumerationOptions
)
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def get_mol_frags(mol: Mol, discard_gt_eq: int=300) -> List[Mol]:
    """
    Generates a list of fragments from the given molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule to generate fragments for.
        discard_gt_eq (int): Discard fragments with a weight greater than or equal to this value.
    """
    frags = Chem.GetMolFrags(mol, asMols=True)
    return [x for x in frags if MolWt(x) < discard_gt_eq]

def generate_tautomers(mol: Mol, max_tautomers: int=5) -> List[Mol]:
    """
    Generates a list of tautomers, with the canonical one first.

    Code from http://rdkit.blogspot.com/2020/01/trying-out-new-tautomer.html

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule to generate tautomers for.
        max_tautomers (int): Maximum number of tautomers to generate.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    canon = enumerator.Canonicalize(mol)
    csmi = Chem.MolToSmiles(canon)
    res = [canon]
    tauts = enumerator.Enumerate(mol)
    smis = [Chem.MolToSmiles(x) for x in tauts]
    stpl = sorted((x, y) for x, y in zip(smis, tauts) if x != csmi)
    res += [y for x, y in stpl]
    return res[:max_tautomers]

def count_lines(fpath: str) -> int:
    """
    Counts the lines in the given file.

    Args:
        fpath (str): Path to file.
    """
    return sum(1 for i in open(fpath, 'rb'))

def process_single_molecule(line: str,
                            max_mol_weight: float=1000.0,
                            max_frag_weight: float=300.0,
                            max_tautomers: int=5,
                            max_isomers: int=10,
                            opts=StereoEnumerationOptions(tryEmbedding=True, unique=True, maxIsomers=10)) -> List[Mol]:
                            
    mol_id, smiles, std_inchi, std_inchi_key = line
    mol = Chem.MolFromSmiles(smiles)
    all_variants = []

    if MolWt(mol) > max_mol_weight:
        return
    
    isomers = EnumerateStereoisomers(mol, options=opts)

    sorted_tautomers = generate_tautomers(mol, max_tautomers=max_tautomers)

    all_variants += get_mol_frags(mol, discard_gt_eq=max_frag_weight)

    for isomer in isomers:
        frags = get_mol_frags(isomer, discard_gt_eq=max_frag_weight)
        all_variants += frags

    for taut in sorted_tautomers:
        frags = get_mol_frags(taut, discard_gt_eq=max_frag_weight)
        all_variants += frags

    return all_variants

def process_molecules(
        in_fname: str, 
        out_fname: str, 
        max_mol_weight: float=1000.0, 
        max_frag_weight: float=300.0,
        max_isomers: int=10,
        max_tautomers: int=5,
        opts=StereoEnumerationOptions(tryEmbedding=True, unique=True, maxIsomers=10)) -> None:
    """
    Process a .tsv file of molecules and write out a new one with 
    fragments generated from selected stereoisomers and tautomers of
    the molecules.

    Args:
        in_fname (str): Path to input file.
        out_fname (str): Path to output file.
        max_mol_weight (float): Maximum molecular weight.
        max_frag_weight (float): Maximum fragment weight.
        max_isomers (int): Maximum number of isomers to enumerate.
        max_tautomers (int): Maximum number of tautomers to enumerate.
    """
    with open(in_fname, "r") as infile, open(out_fname, "w") as outfile:
        line_count = len(open(in_fname, "r").readlines()) - 1
        header = infile.readline().split("\t")
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(["count", "smiles", "inchi_key"])
        seen_fragments = defaultdict(int)
        results = dm.parallelized(process_single_molecule, reader, n_jobs=-1, progress=True, total=line_count)

        print("Counting fragments and writing output...")
        def count_frags(list_of_var_list):
            for var_list in list_of_var_list:
                if var_list is None:
                    continue
                for variant in var_list:
                    var_smiles = Chem.MolToSmiles(variant)
                    seen_fragments[var_smiles] += 1

        count_frags(results)
        for var_smiles, count in seen_fragments.items():
            mol = Chem.MolFromSmiles(var_smiles)
            var_std_inchi_key = Chem.MolToInchiKey(mol)
            writer.writerow([str(count), var_smiles, var_std_inchi_key])

if __name__ == "__main__":
    parser = ArgumentParser(description="""Process a .tsv file of molecules by: 
1. Removing all molecules with a molecular weight above the specified threshold.
2. Enumerating stereoisomers and tautomers.
3. Generating fragments for all the above variants.
4. Removing all fragments with a fragment weight above the specified threshold.
""")
    parser.add_argument("-i", "--in-fname", type=str, help="Path to input file.")
    parser.add_argument("-o", "--out-fname", type=str, help="Path to output file.")
    parser.add_argument("-mw", "--max-mol-weight", type=float, default=1000.0, help="Maximum molecular weight (default: 1000.0 dA).")
    parser.add_argument("-fw", "--max-frag-weight", type=float, default=300.0, help="Maximum fragment weight (default: 300.0 dA).")
    parser.add_argument("-mi", "--max-isomers", type=int, default=10, help="Maximum number of isomers to enumerate (default: 10).")
    parser.add_argument("-mt", "--max-tautomers", type=int, default=5, help="Maximum number of tautomers to enumerate (default: 5).")
    
    args = vars(parser.parse_args())
    process_molecules(**args)
