import os
import csv
import wget
import click
import fsspec
import numpy as np
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
from opendata.preprocessing.utils import merge_counters

RDLogger.DisableLog('rdApp.*')


@click.group()
def cli():
    pass


remote_cache = "gs://opendatasets/small_molecule_collections"
gcs = fsspec.filesystem("gs")

def _load_one_(f_url, cache_dir, verbose=0):
    basename, dname = f_url.split('/')[-1], cache_dir.split('/')[-1]
    basebase = basename.split('.')[0]

    data_local_cache_path = os.path.join(cache_dir, basename)
    result_local_cache_path = os.path.join(cache_dir, basebase)
    gcs_path = os.path.join(remote_cache, basebase)
    if os.path.exists(result_local_cache_path):
        with fsspec.open(result_local_cache_path, "rb") as fd:
            res = pkl.load(fd)
    elif gcs.exists(gcs_path):
        with fsspec.open(result_local_cache_path, "rb") as fd:
            res = pkl.load(fd)
    else:
        if not os.path.exists(data_local_cache_path):
            try:
                wget.download(f_url, data_local_cache_path)
                # ftpfs.download(f_url, data_local_cache_path)
            except Exception as e:
                logger.warning(f"Error {e} for {f_url}")
                return []

        with fsspec.open(data_local_cache_path, compression="gzip") as fd:
            with dm.without_rdkit_log():
                if dname == "surechembl":
                    df = dm.read_csv(fd, sep="\t")
                    res = df.SMILES.to_list()
                elif dname == "mcule":
                    df = dm.read_csv(fd, sep="\t")
                    res = df.iloc[:, 0].to_list()
                elif dname == "chembl":
                    df = dm.read_csv(fd, sep="\t")
                    res = df.canonical_smiles.to_list()
                elif dname == "pubchem":
                    df = dm.read_sdf(fd, as_df=True)
                    print(df.head())
                    res = df["smiles"].to_list()
                else:
                    raise NotImplementedError()
        with fsspec.open(result_local_cache_path, "wb") as fd:
            pkl.dump(res, fd)

        with fsspec.open(gcs_path, "wb") as fd:
            pkl.dump(res, fd)

    if verbose:
        logger.info(f"Nb molecules in {f_url.split('/')[-1]}: {len(res)}")
    return res


def load_surechembl(cache_dir, verbose=0, n_jobs=-1):
    os.makedirs(cache_dir, exist_ok=True)
    ftp_url = "ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/data/SureChEMBL*.txt.gz"
    base, tail = ftp_url.split('/')[0], '/'.join(ftp_url.split('/')[1:])
    ftpfs = fsspec.implementations.ftp.FTPFileSystem(base, timeout=300)
    f_tails = ftpfs.glob(tail)
    all_smiles = set()
    f_urls = ["https://" + base + f_tail for f_tail in f_tails] #[-2:]
    f = lambda x: _load_one_(x, cache_dir=cache_dir, verbose=verbose)
    res = dm.parallelized(f, f_urls, n_jobs=1)

    all_smiles = set()
    for x in res:
        all_smiles.update(x)
    return list(all_smiles)


def load_pubchem(cache_dir, verbose=0, n_jobs=-1):
    os.makedirs(cache_dir, exist_ok=True)
    ftp_url = "ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/*sdf.gz"
    base, tail = ftp_url.split('/')[0], '/'.join(ftp_url.split('/')[1:])
    ftpfs = fsspec.implementations.ftp.FTPFileSystem(base, timeout=300)
    f_tails = ftpfs.glob(tail)
    f_urls = ["https://" + base + f_tail for f_tail in f_tails] #[:2]
    f = lambda x: _load_one_(x, cache_dir=cache_dir, verbose=verbose)
    res = dm.parallelized(f, f_urls, n_jobs=4)
    all_smiles = set()
    for x in res:
        all_smiles.update(x)
    return list(all_smiles)


def load_mcule(cache_dir, verbose=0, n_jobs=-1):
    url = "https://mcule.s3.amazonaws.com/database/mcule_purchasable_full_230323.smi.gz"
    return _load_one_(url, cache_dir=cache_dir, verbose=verbose)



def load_chembl(cache_dir, verbose=0, n_jobs=-1):
    url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_chemreps.txt.gz"
    return _load_one_(url, cache_dir=cache_dir, verbose=verbose)


def load_datasets(cache_dir, flatten=False, verbose=0, n_jobs=-1):
    unichem_db_urls = dict( 
        pubchem=load_pubchem,
        surecheml=load_surechembl,
        mcule=load_mcule,
        chembl=load_chembl,
    )
    os.makedirs(cache_dir, exist_ok=True)
    basename = f"unichem_{'non' if flatten else ''}flatten.pkl"
    local_cache_path = os.path.join(cache_dir, basename)
    gcs_path = os.path.join(remote_cache, basename)

    if os.path.exists(local_cache_path):
        with fsspec.open(local_cache_path, "rb") as fd:
            all_smiles = pkl.load(fd)
    elif gcs.exists(gcs_path):
        with fsspec.open(local_cache_path, "rb") as fd:
            all_smiles = pkl.load(fd)
    else:
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
                if len(all_smiles) == 0:
                    all_smiles = set(smiles)
                else:
                    all_smiles.update(smiles)
            else:
                all_smiles[db_name] = smiles
        with fsspec.open(local_cache_path, "wb") as fd:
            pkl.dump(all_smiles, fd)

        with fsspec.open(gcs_path, "wb") as fd:
            pkl.dump(all_smiles, fd)
    return all_smiles


def expansion_by_fragment_decomposition(list_smiles: List[str],
                                        max_mol_weight: float=1000.0,
                                        max_frag_weight: float=300.0) -> List[str]:
    res = Counter()
    for smi in list_smiles:
        try:
            mol = dm.to_mol(smi)
            mol = dm.standardize_mol(dm.remove_stereochemistry(mol))
            if mol is None:
                continue
            w = MolWt(mol)
            
            with dm.without_rdkit_log():
                frags = dm.fragment.brics(mol, remove_parent=True)
                frags = [dm.to_smiles(x) for x in frags if MolWt(x) < max_frag_weight]

            if w < max_mol_weight:
                frags.append(dm.to_smiles(mol))

            res.update(Counter(frags))
        except:
            pass
    return res


def expansion_by_iso_tautomerization(list_smiles: List[str], 
                                      max_isomers: int=10, 
                                      max_tautomers: int=10):
    res = Counter()
    for smi in list_smiles:
        try:
            mol = dm.to_mol(smi)
            if mol is None:
                continue

            with dm.without_rdkit_log():
                isomers = dm.enumerate_stereoisomers(mol, n_variants=max_isomers, 
                                                    undefined_only=False, rationalise=True, 
                                                    timeout_seconds=30)
                tautomers = dm.enumerate_tautomers(mol,n_variants=max_tautomers)
                it_smiles = [dm.to_smiles(x) for x in isomers+tautomers]
                res.update(Counter(it_smiles))
        except:
            pass        
    return res    

@click.command()
@click.option("--cache_dir", type=str, help="Path to cache directory.")
@click.option("--max-mol-weight", type=float, default=1000.0, help="Maximum molecular weight (default: 1000.0 dA).")
@click.option("--max-frag-weight", type=float, default=300.0, help="Maximum fragment weight (default: 300.0 dA).")
@click.option("--max-isomers", type=int, default=10, help="Maximum number of isomers to enumerate (default: 10).")
@click.option("--max-tautomers", type=int, default=5, help="Maximum number of tautomers to enumerate (default: 5).")
def preprocess_datasets(
        cache_dir,
        max_mol_weight: float=500.0, 
        max_frag_weight: float=250.0,
        max_isomers: int=10,
        max_tautomers: int=5) -> None:
    """
    Load unichem molecules and expand them by fragment decomposition and
    stereoisomer + tautomer enumeration.

    Args:
        in_fname (str): Path to input file.
        out_fname (str): Path to output file.
        max_mol_weight (float): Maximum molecular weight.
        max_frag_weight (float): Maximum fragment weight.
        max_isomers (int): Maximum number of isomers to enumerate.
        max_tautomers (int): Maximum number of tautomers to enumerate.
    """
    all_smiles = load_datasets(cache_dir, flatten=True, verbose=1, n_jobs=-1)
    logger.info(f"Nb molecules: {len(all_smiles)}")

    all_smiles = list(all_smiles)
    logger.info("Expanding molecules...")
    f = lambda x: [expansion_by_fragment_decomposition(x, max_mol_weight, max_frag_weight)]
    res = dm.parallelized_with_batches(f, all_smiles, batch_size=256, n_jobs=-1, progress=True)
    res = merge_counters(res, step=5, verbose=0)
    logger.info(f"Nb fragments: {len(res)}")
    with fsspec.open(os.path.join(cache_dir, "raw_fragments.pkl"), "wb") as fd:
        pkl.dump(res, fd)

    logger.info("Expanding fragments...")
    f = lambda x: [expansion_by_iso_tautomerization(x, max_isomers, max_tautomers)]
    res = dm.parallelized_with_batches(f, res.keys(), batch_size=64, n_jobs=-1, progress=True)
    res = merge_counters(res, step=5, verbose=0)
    logger.info(f"Nb extended fragments: {len(res)}")
    with fsspec.open(os.path.join(cache_dir, "fragments_with_stereo_and_tauto.pkl"), "wb") as fd:
        pkl.dump(res, fd)



if __name__ == "__main__":
    preprocess_datasets(cache_dir="/storage/shared_data/prudencio/odd/cache")
