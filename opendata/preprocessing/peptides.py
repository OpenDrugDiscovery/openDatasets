
import os
import wget
import gzip
import json
import click
import fsspec
import itertools
import pandas as pd
import datamol as dm
import pickle as pkl
import urllib.parse as urlparse
from loguru import logger
from urllib import request
from Bio import SeqIO
from collections import Counter
from opendata.utils import merge_counters, get_local_cache, get_remote_cache

@click.group()
def cli():
    pass

file_urls = [
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz",
]


def create_kmers_frequency(peptides_with_count, k=10):
    """Create counts of kmers of size k from a list of pairs (peptide/protein and count) from a corpus."""
    def all_natural(x):
        return len(set("BJOUXZ").intersection(x)) == 0
    
    kmers_with_count = [(peptide[i:i+k], count) 
                        for peptide, count in peptides_with_count 
                        for i in range(len(peptide) - k + 1)]
    kmers_with_count = [(kmer, count) for kmer, count in kmers_with_count if all_natural(kmer)]
    if len(kmers_with_count) == 0:
        res = Counter()
    else:
        peptides, _ = zip(*kmers_with_count)
        kmers = dict(zip(peptides, [0]*len(peptides)))  
        for kmer, count in kmers_with_count:
            kmers[kmer] += count
        res = Counter(kmers)
    return res 


def filter_by_count(x, min_count=1):
    def _filter_by_count_(x):
        return [(k, v) for k, v in x if v >= min_count]
    
    res = x
    if min_count > 1:
        res = dm.parallelized_with_batches(_filter_by_count_, list(res.items()), n_jobs=-1, progress=True, batch_size=1000, flatten_results=True)
        res = Counter(dict(res))

    return res


@cli.command()
@click.option("--reviewed-only", is_flag=True, help="Only use reviewed proteins (SwissProt) or not (TrEmbl).")
@click.option("--min-count", default=1, help="Minimum count of a kmer to be kept.")
def uniprotkb(reviewed_only=True, min_count=1):
    cache_folder = get_local_cache()
    cache_folder = os.path.join(cache_folder, "uniprotkb")
    os.makedirs(cache_folder, exist_ok=True)

    remote_cache, _ = get_remote_cache("peptides", return_filesystem=True)
    remote_cache = os.path.join(remote_cache, "uniprotkb")

    if reviewed_only:
        urls = file_urls[:1]
    else:
        urls = file_urls

    # download and read files
    peptides = set()
    for url in urls:
        local_file = os.path.join(cache_folder, url.split("/")[-1])

        # download and save file into the cache folder
        if os.path.exists(local_file):
            logger.info("File already downloaded")
        else:
            logger.info("Downloading", url)
            wget.download(url, local_file)

        with gzip.open(local_file, "rt") as f:
            peptides_file = [str(record.seq) for record in SeqIO.parse(f, "fasta")]
        peptides.update(peptides_file)
    
    peptides = list(peptides)
    logger.info(f"n={len(peptides)}")

    # create kmers
    kmers = dict(total_count=dict(), n_kmers=dict())
    last_kmers = None
    for k in range(10, 0, -1):
        if last_kmers is None:
            n = 5000 #None
            last_kmers = dict(zip(peptides[:n], [1] * len(peptides[:n])))
        last_kmers = list(last_kmers.items())

        logger.info(f"Creating kmers of size {k}")
        f = lambda x: create_kmers_frequency(x, k=k)

        res = dm.utils.parallelized_with_batches(f, 
                                                last_kmers, 
                                                batch_size=512, 
                                                n_jobs=-1, 
                                                flatten_results=False,
                                                progress=True)

        last_kmers = merge_counters(res, step=5)
        logger.info(f"n={len(last_kmers)}, {last_kmers.most_common(10)}")
        kmers[str(k)] = last_kmers
        kmers["n_kmers"][k] = len(last_kmers)
        kmers["total_count"][k] = sum(last_kmers.values())

    for k in range(10, 0, -1):
        kmers[str(k)] = filter_by_count(kmers[str(k)], min_count=min_count)

    # save kmers
    basename = f"peptides_set_{'reviewed' if reviewed_only else 'unreviewed'}_min_occ_{min_count}.pkl"
    with fsspec.open(os.path.join(cache_folder, basename), "wb") as f:
        pkl.dump(kmers, f)

    with fsspec.open(os.path.join(remote_cache, basename), "wb") as f:
        pkl.dump(kmers, f)

    return kmers



def remote_and_local_dbptm_files(cache_folder):
    """Get local and remote file for dbPTM."""
    cache_folder = os.path.join(cache_folder, "dbPTM")
    os.makedirs(cache_folder, exist_ok=True)

    ptm_types = [ 
        "ADP-ribosylation", "AMPylation", "Acetylation", "Amidation", "Biotinylation", 
        "Blocked amino end", "Butyrylation", "C-linked Glycosylation", "Carbamidation", 
        "Carboxyethylation", "Carboxylation", "Cholesterol ester", "Citrullination", 
        "Crotonylation", "D-glucuronoylation", "Deamidation", "Decanoylation", 
        "Decarboxylation", "Dephosphorylation", "Disulfide bond", "Farnesylation", 
        "Formation of an isopeptide bond", "Formylation", "GPI-anchor", "Gamma-carboxyglutamic acid", 
        "Geranylgeranylation", "Glutarylation", "Glutathionylation", "Hydroxyceramide ester", 
        "Hydroxylation", "Iodination", "Lactoylation", "Lactylation", "Lipoylation", "Malonylation", 
        "Methylation", "Myristoylation", "N-carbamoylation", "N-linked Glycosylation", "N-palmitoylation", 
        "Neddylation", "Nitration", "O-linked Glycosylation", "O-palmitoleoylation", "O-palmitoylation", 
        "Octanoylation", "Oxidation", "Phosphatidylethanolamine amidation", "Phosphorylation", "Propionylation", 
        "Pyrrolidone carboxylic acid", "Pyrrolylation", "Pyruvate", "S-Cyanation", "S-archaeol", "S-carbamoylation", 
        "S-cysteinylation", "S-diacylglycerol", "S-linked Glycosylation", "S-nitrosylation", "S-palmitoylation", 
        "Serotonylation", "Stearoylation", "Succinylation", "Sulfation", "Sulfhydration", "Sulfoxidation", 
        "Sumoylation", "Thiocarboxylation", "UMPylation", "Ubiquitination"
    ]

    urls = ["https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/docs/ptmlist.txt"]
    local_files = [os.path.join(cache_folder, "ptmlist.csv")]
    for ptm in ptm_types:
        urls += [f"https://awi.cuhk.edu.cn/dbPTM/download/experiment/{ptm}.gz"]
        local_files += [os.path.join(cache_folder, ptm.replace(" ", "_").replace("-", "_").lower() + ".gz")]

    return urls, local_files



def read_ptm_dict(url, local_file):
    if not os.path.exists(local_file):
        with fsspec.open(url) as f:
            content = f.read().decode("utf-8")
        parts = content.split("\nID")[1:]
        df_content = []
        for part in parts:
            part = "ID" + part.split("//")[0]
            part = {line[:2]: line[5:] for line in part.split("\n")[:-1]}
            df_content.append(part)

        pd.DataFrame(df_content).to_csv(local_file, index=False)
            
    return pd.read_csv(local_file)


@cli.command()
def dbptm():
    cache_folder = get_local_cache()
    urls, local_files = remote_and_local_dbptm_files(cache_folder)
    col_headers = ["gene_name", "UniProt ID", "position", "PTM_type", "refs", "sequence"]

    # download and read files
    dfs = []
    for url, local_file in zip(urls, local_files):
        if not os.path.exists(local_file):
            wget.download(url, local_file)

        df = pd.read_table(local_file, names=col_headers).dropna(subset=["sequence"])
        df.to_csv("ptm.csv", index=False)
        dfs.append(df)

    # merge all files
    df = pd.concat(dfs, axis=0, ignore_index=True)
    print(df.head())
    print(df.shape)

    x = Counter([len(el) for el in df["sequence"]])
    print(x)
    exit()


    

if __name__ == "__main__":
    cli()