import csv

from tqdm import tqdm
from rdkit import Chem
from argparse import ArgumentParser
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers, 
    StereoEnumerationOptions
)
from rdkit.Chem.MolStandardize import rdMolStandardize

def reorder_tautomers(mol):
    """
    Generates a list of tautomers, with the canonical one first.

    Code from http://rdkit.blogspot.com/2020/01/trying-out-new-tautomer.html
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    canon = enumerator.Canonicalize(mol)
    csmi = Chem.MolToSmiles(canon)
    res = [canon]
    tauts = enumerator.Enumerate(mol)
    smis = [Chem.MolToSmiles(x) for x in tauts]
    stpl = sorted((x, y) for x, y in zip(smis, tauts) if x != csmi)
    res += [y for x, y in stpl]
    return res

def count_lines(fpath: str) -> int:
    """
    Counts the lines in the given file.
    """
    return sum(1 for i in open(fpath, 'rb'))

def process_molecules(
        in_fname: str, 
        out_fname: str, 
        max_mol_weight: float=1000.0, 
        max_frag_weight: float=300.0,
        max_isomers: int=10) -> None:
    """Process ChEMBL file and write out a new one with only the molecules that are below the
    specified maximum molecular weight and maximum fragment weight.

    Args:
        in_fname (str): Path to input file.
        out_fname (str): Path to output file.
        max_mol_weight (float): Maximum molecular weight.
        max_frag_weight (float): Maximum fragment weight.
    """
    frags = []
    opts = StereoEnumerationOptions(tryEmbedding=True, unique=True, maxIsomers=max_isomers)
    with open(in_fname, "r") as infile, open(out_fname, "w") as outfile:
        reader = csv.reader(infile, delimiter="\t")
        for i, line in enumerate(reader):
            if i == 0:
                # just write the header here
                joined = "\t".join(line)
                outfile.write(joined + "\n")
                continue
            
            mol_id, smiles, std_inchi, std_inchi_key = line
            mol = Chem.MolFromSmiles(smiles)
            isomers = EnumerateStereoisomers(mol, options=opts)
            sorted_isomers = sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers)


            # for isomer in sorted_isomers:


if __name__ == "__main__":
    parser = ArgumentParser(description="""Process a tsv file of molecules by 
1. Removing all molecules with a molecular weight above the specified threshold.
2. Enumerating stereoisomers, tautomers, and protomers.
3. Generating fragments for all the above variants.
4. Removing all molecules with a fragment weight above the specified threshold.
""")
    parser.add_argument("-i", "--in-fname", type=str, help="Path to input file.")
    parser.add_argument("-o", "--out-fname", type=str, help="Path to output file.")
    parser.add_argument("-mw", "--max-mol-weight", type=float, help="Maximum molecular weight (default: 1000.0 dA).")
    parser.add_argument("-fw", "--max-frag-weight", type=float, help="Maximum fragment weight (default: 300.0 dA).")
    parser.add_argument("-mi", "--max-isomers", type=int, default=10, help="Maximum number of isomers to enumerate (default: 10).")
    
    args = parser.parse_args()
    process_molecules(*args)
