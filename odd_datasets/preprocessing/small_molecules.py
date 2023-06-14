import os
import csv
import gzip
import fsspec
import pandas as pd
import datamol as dm
import pickle as pkl
from typing import List
from loguru import logger
from collections import Counter, defaultdict
from rdkit import Chem
import fsspec.implementations.ftp
from rdkit.Chem.Descriptors import MolWt
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def _load_one_surechembl_(f_url, cache_dir, verbose=0):
    with fsspec.open(f"filecache::{f_url}", 
                        filecache={'cache_storage':cache_dir}) as fd:
        with dm.without_rdkit_log():
            df = dm.read_csv(fd, sep="\t", compression="gzip")
    if verbose:
        logger.info(f"Nb molecules in {f_url.split('/')[-1]}: {df.shape}")
    return df.SMILES.to_list()


def load_surechembl(cache_dir, verbose=0, n_jobs=-1):
    ftp_url = "ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/data/SureChEMBL*.txt.gz"
    base, tail = ftp_url.split('/')[0], '/'.join(ftp_url.split('/')[1:])
    ftpfs = fsspec.implementations.ftp.FTPFileSystem(base)
    f_tails = ftpfs.glob(tail)
    all_smiles = set()
    f_urls = ["https://" + base + f_tail for f_tail in f_tails] #[-2:]
    f = lambda x: _load_one_surechembl_(x, cache_dir=cache_dir, verbose=verbose)
    res = dm.parallelized(f, f_urls, n_jobs=n_jobs)

    all_smiles = set()
    for x in res:
        all_smiles.update(x)
    all_smiles.update()
    return all_smiles


def _load_one_pubchem_(f_url, cache_dir, verbose=0):
    with fsspec.open(f"filecache::{f_url}", 
                filecache={'cache_storage':cache_dir}, compression="gzip") as fd:
        df = dm.read_sdf(fd, as_df=True)
    if verbose:
        logger.info(f"Nb molecules in {f_url.split('/')[-1]}: {df.shape}")
    return df["PUBCHEM_OPENEYE_CAN_SMILES"].to_list()
    

def load_pubchem(cache_dir, verbose=0, n_jobs=-1):
    ftp_url = "ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/*sdf.gz"
    base, tail = ftp_url.split('/')[0], '/'.join(ftp_url.split('/')[1:])
    ftpfs = fsspec.implementations.ftp.FTPFileSystem(base)
    f_tails = ftpfs.glob(tail)
    f_urls = ["https://" + base + f_tail for f_tail in f_tails] #[:2]
    f = lambda x: _load_one_pubchem_(x, cache_dir=cache_dir, verbose=verbose)
    res = dm.parallelized(f, f_urls, n_jobs=n_jobs)

    all_smiles = set()
    for x in res:
        all_smiles.update(x)

    return all_smiles


def load_mcule(cache_dir, verbose=0, n_jobs=-1):
    url = "https://mcule.s3.amazonaws.com/database/mcule_purchasable_full_230323.smi.gz"
    with fsspec.open(f"filecache::{url}", 
                        filecache={'cache_storage':cache_dir}) as fd:
        df = dm.read_csv(fd, sep="\t", compression="gzip")
    if verbose:
        print(df.head())
        print(df.shape)
    res = set(df.iloc[:, 0].to_list())
    return res


def load_chembl(cache_dir, verbose=0, n_jobs=-1):
    url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_32_chemreps.txt.gz"
    with fsspec.open(f"filecache::{url}", filecache={'cache_storage':cache_dir}) as fd:
        df = dm.read_csv(fd, sep="\t", compression="gzip")
    return set(df.canonical_smiles.to_list())


def load_datasets(cache_dir, flatten=False, verbose=0, n_jobs=-1):
    unichem_db_urls = dict( 
        mcule=load_mcule,
        # surecheml=load_surechembl,
        pubchem=load_pubchem,
        chembl=load_chembl,
    )
    os.makedirs(cache_dir, exist_ok=True)
    all_smiles = set() if flatten else dict()
    for db_name, fn_loader in unichem_db_urls.items():
        local_cache = os.path.join(cache_dir, f"{db_name}.pkl")
        if os.path.exists(local_cache):
            with fsspec.open(local_cache, "rb") as fd:
                smiles = pkl.load(fd)
        else:
            smiles =  fn_loader(cache_dir=os.path.join(cache_dir, db_name), 
                                verbose=verbose, n_jobs=n_jobs)
            with fsspec.open(local_cache, "wb") as fd:
                pkl.dump(smiles, fd)
        if verbose:
            logger.info(f"Nb molecules in {db_name}: {len(smiles)}")

        if flatten:
            all_smiles.update(smiles)
        else:
            all_smiles[db_name] = smiles

    return all_smiles


def expansion_by_fragment_decomposition(list_smiles: List[str],
                                        max_mol_weight: float=1000.0,
                                        max_frag_weight: float=300.0) -> List[str]:
    res = Counter()
    for smi in list_smiles:
        mol = dm.to_mol(smi)
        
        if MolWt(mol) < max_mol_weight:
            with dm.without_rdkit_log():
                frags = dm.fragment.brics(mol, remove_parent=False)
                frags = [dm.to_smiles(x) for x in frags if MolWt(x) < max_frag_weight]
                res.update(Counter(frags))
            
    return res


def expanasion_by_iso_tautomerization(list_smiles: List[str], 
                                      max_isomers: int=10, 
                                      max_tautomers: int=10):
    res = Counter()
    for smi in list_smiles:
        mol = dm.to_mol(smi)
        
        with dm.without_rdkit_log():
            isomers = dm.enumerate_stereoisomers(mol, n_variants=max_isomers, 
                                                undefined_only=False, rationalise=True, 
                                                timeout_seconds=30)
            tautomers = dm.enumerate_tautomers(mol,n_variants=max_tautomers)
            it_smiles = [dm.to_smiles(x) for x in isomers+tautomers]
            res.update(Counter(it_smiles))
            
    return res    


def process_molecules(
        in_fname: str, 
        out_fname: str, 
        max_mol_weight: float=1000.0, 
        max_frag_weight: float=300.0,
        max_isomers: int=10,
        max_tautomers: int=5) -> None:
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
        results = dm.parallelized(process_molecule, reader, n_jobs=-1, progress=True, total=line_count)

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
    load_datasets(cache_dir="/storage/shared_data/prudencio/odd", verbose=1, n_jobs=32)
#     parser = ArgumentParser(description="""Process a .tsv file of molecules by: 
# 1. Removing all molecules with a molecular weight above the specified threshold.
# 2. Enumerating stereoisomers and tautomers.
# 3. Generating fragments for all the above variants.
# 4. Removing all fragments with a fragment weight above the specified threshold.
# """)
#     parser.add_argument("-i", "--in-fname", type=str, help="Path to input file.")
#     parser.add_argument("-o", "--out-fname", type=str, help="Path to output file.")
#     parser.add_argument("-mw", "--max-mol-weight", type=float, default=1000.0, help="Maximum molecular weight (default: 1000.0 dA).")
#     parser.add_argument("-fw", "--max-frag-weight", type=float, default=300.0, help="Maximum fragment weight (default: 300.0 dA).")
#     parser.add_argument("-mi", "--max-isomers", type=int, default=10, help="Maximum number of isomers to enumerate (default: 10).")
#     parser.add_argument("-mt", "--max-tautomers", type=int, default=5, help="Maximum number of tautomers to enumerate (default: 5).")
    
#     args = vars(parser.parse_args())
#     process_molecules(**args)