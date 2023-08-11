import os
import csv
import wget
import glob
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
from opendata.utils import merge_counters, get_local_cache, get_remote_cache, caching_wrapper
from opendata.preprocessing.utils import mols_fragmentation, mols_iso_tautomerization
RDLogger.DisableLog('rdApp.*')


@click.group()
def cli():
    pass

cache_dir = get_local_cache()
remote_dir, remote_fs = get_remote_cache("small_molecules", return_filesystem=True)


def _load_one_(dname, f_url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        try:
            wget.download(f_url, local_path)
            # ftpfs.download(f_url, data_local_cache_path)
        except Exception as e:
            logger.warning(f"Error {e} for {f_url}")
            return []

    with fsspec.open(local_path, compression="gzip") as fd:
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
    return res
            

def _load_one_cache_(f_url, local_cache_dir, verbose=0):
    basename, dname = f_url.split('/')[-1], local_cache_dir.split('/')[-1]
    basebase = basename.split('.')[0]

    data_local_path = os.path.join(local_cache_dir, basename)
    result_local_path = os.path.join(local_cache_dir, basebase)
    gcs_path = os.path.join(remote_dir, basebase)

    res = caching_wrapper(_load_one_, dname, f_url, data_local_path,
                          local_path=result_local_path, 
                          remote_path=gcs_path, remote_filesystem=remote_fs)

    if verbose:
        logger.info(f"Nb molecules in {f_url.split('/')[-1]}: {len(res)}")
    return res


def load_surechembl(local_cache_dir, verbose=0, n_jobs=-1):
    ftp_url = "ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/data/SureChEMBL*.txt.gz"
    base, tail = ftp_url.split('/')[0], '/'.join(ftp_url.split('/')[1:])
    ftpfs = fsspec.implementations.ftp.FTPFileSystem(base, timeout=300)
    f_tails = ftpfs.glob(tail)
    all_smiles = set()
    f_urls = ["https://" + base + f_tail for f_tail in f_tails] #[-2:]
    f = lambda x: _load_one_cache_(x, local_cache_dir=local_cache_dir, verbose=verbose)
    res = dm.parallelized(f, f_urls, n_jobs=1)

    all_smiles = set()
    for x in res:
        all_smiles.update(x)
    return list(all_smiles)


def load_pubchem(local_cache_dir, verbose=0, n_jobs=-1):
    ftp_url = "ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/*sdf.gz"
    base, tail = ftp_url.split('/')[0], '/'.join(ftp_url.split('/')[1:])
    ftpfs = fsspec.implementations.ftp.FTPFileSystem(base, timeout=300)
    f_tails = ftpfs.glob(tail)
    f_urls = ["https://" + base + f_tail for f_tail in f_tails] #[:2]
    f = lambda x: _load_one_cache_(x, local_cache_dir=local_cache_dir, verbose=verbose)
    res = dm.parallelized(f, f_urls, n_jobs=4)
    all_smiles = set()
    for x in res:
        all_smiles.update(x)
    return list(all_smiles)


def load_mcule(local_cache_dir, verbose=0, n_jobs=-1):
    url = "https://mcule.s3.amazonaws.com/database/mcule_purchasable_full_230323.smi.gz"
    return _load_one_cache_(url, local_cache_dir=local_cache_dir, verbose=verbose)



def load_chembl(local_cache_dir, verbose=0, n_jobs=-1):
    url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_chemreps.txt.gz"
    return _load_one_cache_(url, local_cache_dir=local_cache_dir, verbose=verbose)


def _load_all_datasets_(cache_dir, flatten=False, verbose=0, n_jobs=-1):
    unichem_db_urls = dict( 
        pubchem=load_pubchem,
        surecheml=load_surechembl,
        mcule=load_mcule,
        chembl=load_chembl,
    )
    all_smiles = set() if flatten else dict()
    for db_name, fn_loader in unichem_db_urls.items():
        db_local_cache = os.path.join(cache_dir, f"{db_name}.pkl")
        db_remote_path = os.path.join(remote_dir, f"{db_name}.pkl")

        smiles = caching_wrapper(fn_loader, 
                                 os.path.join(cache_dir, db_name),
                                 local_path=db_local_cache, 
                                 remote_path=db_remote_path,
                                 remote_filesystem=remote_fs,
                                 verbose=verbose, n_jobs=n_jobs)

        if verbose:
            logger.info(f"Nb molecules in {db_name}: {len(smiles)}")

        if flatten:
            if len(all_smiles) == 0:
                all_smiles = set(smiles)
            else:
                all_smiles.update(smiles)
        else:
            all_smiles[db_name] = smiles

    return all_smiles


def load_datasets(flatten=False, verbose=0, n_jobs=-1):

    basename = f"unichem_{'non' if flatten else ''}_flatten.pkl"
    local_path = os.path.join(cache_dir, basename)
    remote_path = os.path.join(remote_dir, basename)

    return caching_wrapper(_load_all_datasets_, cache_dir, 
                           flatten=flatten, 
                           verbose=verbose, 
                           n_jobs=n_jobs,
                           remote_path=remote_path, 
                           local_path=local_path,
                           remote_filesystem=remote_fs)


def expansion_by_fragment_decomposition(smiles_list, max_frag_weight: float=300.0):
    f = lambda x: [mols_fragmentation(x, max_frag_weight)]
    res = dm.parallelized_with_batches(f, smiles_list, batch_size=256, n_jobs=-1, progress=True)
    frags = merge_counters(res, step=5, verbose=0)
    logger.info(f"Nb fragments: {len(frags)}")
    return frags


@cli.command()
@click.option("--chunk-id", "-i", type=int,  help="chunk id starting at 0")
@click.option("--chunk-size", "-s", type=int, default=1000000, help="Chunk size to divide and conquer.")
@click.option("--max-frag-weight", "-w", type=float, default=300.0, help="Maximum fragment weight (default: 300.0 dA).")
def fragmentation(
        chunk_id,
        chunk_size,
        max_frag_weight: float=250.0) -> None:
    """
    Load unichem molecules and expand them by fragment decomposition and
    stereoisomer + tautomer enumeration.

    Args:
        in_fname (str): Path to input file.
        out_fname (str): Path to output file.
        max_frag_weight (float): Maximum fragment weight.
    """

    all_smiles = load_datasets(flatten=True, verbose=1, n_jobs=-1)
    n_total = len(all_smiles)
    logger.info(f"Nb molecules: {n_total}")
    
    start_idx, end_idx = chunk_id * chunk_size, (chunk_id+1) * chunk_size
    if start_idx > n_total:
        logger.info("Chunk ID exceed the maximum number of samples")
    else:
        fname = f"raw_fragments_{start_idx}_{end_idx}.pkl"
        local_file = os.path.join(cache_dir, "unichem", fname)
        remote_file = os.path.join(remote_dir, "unichem", fname)

        all_smiles = list(all_smiles)[start_idx:end_idx]
        logger.info(f"Expanding molecules from {start_idx}_{end_idx}...")
        caching_wrapper(expansion_by_fragment_decomposition,
                        all_smiles,
                        max_frag_weight = max_frag_weight,
                        local_path=local_file,
                        remote_path=remote_file,
                        remote_filesystem=remote_fs)


def _load_fragment_collection_(include_iso_tauto=False):
    fname = f"fragments_with_stereo_and_tauto_*.pkl" if include_iso_tauto else "raw_fragments*.pkl"
    local_files = glob.glob(os.path.join(cache_dir, "unichem", fname))
    remote_files = remote_fs.glob(os.path.join(remote_dir, "unichem", fname))
    fnames = local_files if len(local_files) else remote_files
    all_res = []

    if len(fnames):
        for fname in fnames:
            with fsspec.open(fname, "rb") as fd:
                res = pkl.load(fd)
            all_res.append(res)
    res = merge_counters(all_res, step=5)
    return res

def load_fragment_collection(include_iso_tauto=False):
    fname = f"all_iso_tauto_fragments.pkl" if include_iso_tauto else f"all_raw_fragments.pkl" 
    local_file = os.path.join(cache_dir, fname)
    remote_file = os.path.join(remote_dir, fname)
    res = caching_wrapper(_load_fragment_collection_,
                    local_path=local_file,
                    remote_path=remote_file,
                    remote_filesystem=remote_fs,
                    include_iso_tauto=include_iso_tauto
    )
    logger.info(f"Fragment collection size: {len(res)}")

    return res


def expansion_by_iso_tautomerization(list_smiles: List[str]):
    f = lambda x: [mols_iso_tautomerization(x)]
    res = dm.parallelized_with_batches(f, list_smiles, 
                                       batch_size=256, n_jobs=-1, progress=True)
    print(len(res))
    print(type(res[0]))
    frags = merge_counters(res, step=5, verbose=0)
    logger.info(f"Nb extended fragments: {len(frags)}")
    return frags


@cli.command()
@click.option("--chunk-id", "-i", type=int,  help="chunk id starting at 0")
@click.option("--chunk-size", "-s", type=int, default=1000000, help="Chunk size to divide and conquer.")
def iso_and_tauto(
        chunk_id,
        chunk_size) -> None:
    """
    Load unichem molecules and expand them by fragment decomposition and
    stereoisomer + tautomer enumeration.

    Args:
        in_fname (str): Path to input file.
        out_fname (str): Path to output file.
        max_isomers (int): Maximum number of isomers to enumerate.
        max_tautomers (int): Maximum number of tautomers to enumerate.
    """

    res = load_fragment_collection(include_iso_tauto=False)

    all_smiles = list(res.keys())
    n_total = len(all_smiles)
    logger.info(f"Nb molecules: {n_total}")
    
    start_idx, end_idx = chunk_id * chunk_size, (chunk_id+1) * chunk_size
    if start_idx > n_total:
        logger.info("Chunk ID exceed the maximum number of samples")
    else:
        fname = f"fragments_with_stereo_and_tauto_{start_idx}_{end_idx}.pkl"
        local_file = os.path.join(cache_dir, "unichem", fname)
        remote_file = os.path.join(remote_dir, "unichem", fname)

        logger.info(f"Expanding fragments by isomerization and tautomerization...")
        all_smiles = all_smiles[start_idx:end_idx]

        caching_wrapper(expansion_by_iso_tautomerization,
                        all_smiles,
                        local_path=local_file,
                        remote_path=remote_file,
                        remote_filesystem=remote_fs)


@cli.command()
def collect_frags():
    load_fragment_collection(include_iso_tauto=False)

@cli.command()
def collect_iso_tauto_frags():
    load_fragment_collection(include_iso_tauto=True)
    
if __name__ == "__main__":
    cli()
