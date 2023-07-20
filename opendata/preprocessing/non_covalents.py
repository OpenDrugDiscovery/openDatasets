
import os
import wget
import gzip
import json
import h5py
import copy
import fsspec
import itertools
import tarfile
import numpy as np
import pandas as pd
import datamol as dm
import pickle as pkl
from loguru import logger
from urllib import request
from Bio import SeqIO
from rdkit import Chem
from collections import Counter
from opendata import utils 
from openmm.app import PDBxFile, PDBFile
from opendata.preprocessing.pockets import PocketExtractor


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

cache_dir = utils.get_local_cache()
misato_dir = utils.get_misato_folder()

chem_table = Chem.GetPeriodicTable()

def get_misato_maps():
    d = dict(
            residue = 'atoms_residue_map.pickle',
            type = 'atoms_type_map.pickle',
            name = 'atoms_name_map_for_pdb.pickle'
            
    )

    maps = dict()
    for k, v in d.items():
        with fsspec.open(os.path.join(misato_dir, "maps", v), "rb") as fd:
            maps[k] = pkl.load(fd)
    maps['symbol'] = {i: chem_table.GetElementSymbol(i) for i in range(118)}

    elements = list(maps['name'].items())
    idxs = np.array([k[1] for k in maps['name']])
    names = np.array([k[0] for k in maps['name']])
    for aa in np.unique(names):
        ixs = np.argwhere(names == aa).flatten()
        sub = [elements[i] for i in ixs]
        ixs = list(np.argwhere(idxs[ixs] == 0).flatten()) + [len(sub)]
        subs = [sub[i:j] for i, j in zip(ixs[:-1], ixs[1:])]
        for sub in subs:
            vs = {x[-1]:[] for x in sub}
            for x in sub:
                vs[x[-1]].append(x[0])

            # for v, k in vs.items():
            #     if len(k) > 1:
            #         print(v, k)

    # make some adjustments:
    maps['name'][('PRO', 8, 'HC')] = "HB1"
    return maps


class MisatoMDFrame:

    misato_maps = get_misato_maps()
    def __init__(self, pdb_code, h5_file_descriptor, frame_idx) -> None:
        self.frame_idx = frame_idx
        self.pdb_code = pdb_code

        res =  dict()
        for k in ['atoms_type', 'atoms_number', 'atoms_residue']:
            res[k] = h5_file_descriptor.get(pdb_code+'/'+ k)[:]

        res["atom_symbol"] = np.vectorize(self.misato_maps["type"].__getitem__)(res['atoms_type'])
        res["res_name"] = np.vectorize(self.misato_maps["residue"].__getitem__)(res['atoms_residue'])
        res["res_number"] = np.zeros_like(res["atoms_number"])
        res["res_atom_index"] = np.zeros_like(res["atoms_number"])

        x, y = res["res_name"], res["atom_symbol"]
        predicat = (x[:-1] != x[1:]) | (((y[:-1] == 'O')|(y[:-1] == 'O2')) & ((y[1:] == 'N')|(y[1:] == 'N3')))
        
        for i, p in enumerate(predicat):
            if (p and (x[i] in ['ASN', 'GLN']) and (x[i] == x[i+1]) 
                and (''.join(y[i:i+6]) in ['ONHHCO', 'ONHHCO2'])):
                predicat[i] = False   
        end_idxs = h5_file_descriptor.get(pdb_code+'/'+ "molecules_begin_atom_index")[:]-1
        idxs = sorted(list(np.argwhere(predicat).flatten()) + list(end_idxs[end_idxs >= 0]))

        i = 0
        for k, j in enumerate(idxs):
            res["res_number"][i:j+1] = k+1
            res["res_atom_index"][i:j+1] = range(j-i+1)
            i = j+1
        if i != x.shape[0]:
            j = x.shape[0]-1
            res["res_number"][i:j+1] = k+1
            res["res_atom_index"][i:j+1] = range(j-i+1)


        tmp = pd.DataFrame(res).reset_index()
        res["atom_name"] = tmp.apply(self.get_atom_name, axis=1)
        if np.any(pd.isna(res["atom_name"])):
            logger.debug()
            print(idxs[:5], idxs[-5:], tmp.shape)
        
        res['x'] = h5_file_descriptor.get(pdb_code+'/'+'trajectory_coordinates')[frame_idx][:, 0]
        res['y'] = h5_file_descriptor.get(pdb_code+'/'+'trajectory_coordinates')[frame_idx][:, 1]
        res['z'] = h5_file_descriptor.get(pdb_code+'/'+'trajectory_coordinates')[frame_idx][:, 2]
        res['molecules_begin_atom_index'] = h5_file_descriptor.get(pdb_code+'/'+ "molecules_begin_atom_index")[:]
        self.data = res

    def get_atom_name(self, x):
        if x['res_name'] == 'MOL':
            r = self.misato_maps['symbol'][x['atoms_number']]+str(x['res_atom_index'])  
        else:
            try:
                r = self.misato_maps['name'][(x['res_name'], x['res_atom_index'], x['atom_symbol'])]
            except:
                print(x['index'], x['res_name'], x['res_atom_index'], x['atom_symbol'])
                # raise Exception("toto")
                return None
        return r

    def write_pdb(self, out_dir):
        """
        We go through each atom line and bring the inputs in the pdb format
        
        """
        data = copy.deepcopy(self.data)
        molecules_begin_atom_index = set(data.pop("molecules_begin_atom_index"))

        entries = pd.DataFrame(data).reset_index().to_dict('records')
        lines = []
        for i, entry in enumerate(entries):
            line = 'ATOM{index:7d} {atom_name:<4} {res_name:<4}{res_number:>5}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atoms_number:<5}'.format(**entry)
            lines.append(line)
            if i + 1 in molecules_begin_atom_index:
                lines.append("TER")
    
        fname = os.path.join(out_dir, f"{self.pdb_code}_{self.frame_idx}"+'.pdb')
        with fsspec.open(fname, 'w') as fd:
            fd.write('\n'.join(lines))
        
        return fname


class MisatoMD:

    def __init__(self, pdb_code, h5_file_descriptor) -> None:
        self.frames = [MisatoMDFrame(pdb_code, h5_file_descriptor, frame_idx=i) for i in range(100)]
        self.pdb_code = pdb_code

    

def get_misato():
    fname = os.path.join(misato_dir, "MD.hdf5")
    m_cache_dir = os.path.join(cache_dir,  "misato")

    fd = fsspec.open(fname, mode="rb")
    if hasattr(fd, "open"):
        fd = fd.open()
    md_data = h5py.File(fd)
    complex_codes = list(md_data.keys())
    # np.random.shuffle(complex_codes)
    for code in complex_codes[:10]:
        print(code)
        m = MisatoMDFrame(code, md_data, 0)
        fname = m.write_pdb(m_cache_dir)
        # Open the input PDB file
        p_extractor = PocketExtractor(distance_cutoff=4.5)
        p_extractor.from_pdb(pdb_path=fname, filepath=fname.replace(code, f'{code}_pocket'), save_pdb=True)


def create_pdbbind_set(cache_folder):
    """Create a set of PDBbind ligands."""
    pass


if __name__ == "__main__":
    # get_pdb_bind("cache")
    get_misato()
