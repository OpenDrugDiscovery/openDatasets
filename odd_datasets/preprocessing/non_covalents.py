
import os
import wget
import gzip
import json
import itertools
import tarfile
import datamol as dm
from loguru import logger
from urllib import request
from Bio import SeqIO
from collections import Counter

file_urls = [
    # "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_plain_text_index.tar.gz",
    "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz",
    "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_NL.tar.gz",
    "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_PN.tar.gz",
    "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_PP.tar.gz",
    "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_other_PL.tar.gz",
    # "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_mol2.tar.gz",
    # "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_sdf.tar.gz"
]



def download_and_extract_archives(cache_folder):
    """Download and extract archives from PDBbind."""
    cache_folder = os.path.join(cache_folder, "pdbbind")
    os.makedirs(cache_folder, exist_ok=True)

    # download and read files
    for url in file_urls:
        local_file = os.path.join(cache_folder, url.split("/")[-1])

        # download and save file into the cache folder
        if os.path.exists(local_file):
            logger.info(f"File already downloaded at {local_file}")
        else:
            logger.info("Downloading", url)
            wget.download(url, local_file)

            with tarfile.open(local_file, "r") as f:
                f.extractall(cache_folder)
    

def create_pdbbind_set(cache_folder):
    """Create a set of PDBbind ligands."""
    pass

if __name__ == "__main__":
    download_and_extract_archives("cache")