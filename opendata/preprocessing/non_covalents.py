
import os
import wget
import gzip
import json
import h5py
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
    return maps


class MisatoMD:

    misato_maps = get_misato_maps()
    def __init__(self, pdb_code, f, frame_idx) -> None:
        self.frame_idx = frame_idx
        self.pdb_code = pdb_code

        res =  dict()
        for k in ['atoms_type', 'atoms_number', 'atoms_residue']:
            res[k] = f.get(pdb_code+'/'+ k)[:]

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
        end_idxs = f.get(pdb_code+'/'+ "molecules_begin_atom_index")[:]-1
        idxs = sorted(list(np.argwhere(predicat).flatten()) + list(end_idxs[end_idxs >= 0]))

        i = 0
        for k, j in enumerate(idxs + [-1]):
            res["res_number"][i:j+1] = k+1
            if j == -1: j = x.shape[0]-1
            res["res_atom_index"][i:j+1] = range(j-i+1)
            i = j+1

        tmp = pd.DataFrame(res).reset_index()
        res["atom_name"] = tmp.apply(self.get_atom_name, axis=1)
        if np.any(pd.isna(res["atom_name"])):
            logger.debug()
            print(idxs[:5], idxs[-5:], tmp.shape)
        
        res['x'] = f.get(pdb_code+'/'+'trajectory_coordinates')[frame_idx][:, 0]
        res['y'] = f.get(pdb_code+'/'+'trajectory_coordinates')[frame_idx][:, 1]
        res['z'] = f.get(pdb_code+'/'+'trajectory_coordinates')[frame_idx][:, 2]
        res['molecules_begin_atom_index'] = f.get(pdb_code+'/'+ "molecules_begin_atom_index")[:]
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
        data = copy(self.data)
        molecules_begin_atom_index = set(data.pop("molecules_begin_atom_index"))

        entries = pd.DataFrame(data).reset_index().to_dict('records')
        lines = []
        for i, entry in enumerate(entries):
            line = 'ATOM{index:7d}  {atom_name:<4}{res_name:<4}{res_number:>5}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_number:<5}'.format(**entry)
            lines.append(line)
            if i + 1 in molecules_begin_atom_index:
                lines.append("TER")
    
        fname = os.path.join(out_dir, f"{self.pdb_code}_{self.frame_idx}"+'.pdb')
        with fsspec.open(fname, 'w') as fd:
            fd.write('\n'.join(lines))


def isolate_pocket(pdb_code, f, frame_idx, out_dir, distance_cutoff=10):
    """distance_cutoff is in angstroms"""

    # Make a dir for that data point
    datum_data_dir = dm.fs.join(out_dir, pdb_code)
    dm.fs.mkdir(datum_data_dir, exist_ok=True)

    data = parse_misato_entries(pdb_code, f, frame_idx)

    # Save the original PDB file to the datum directory
    full_input_pdb_path = dm.fs.join(datum_data_dir, "original.pdb")
    write_pdb(data, pdb_code, frame_idx, )

    ## Step 1: extract and save the ligand XYZ file
    ligand = dm.read_sdf(
        dm.fs.join(base_data_dir, input_ligand_path),
        remove_hs=False,
    )[0]
    ligand = dm.atom_indices_to_mol(ligand)

    # Make a flat 2D ligand
    ligand_flat = dm.copy_mol(ligand)
    ligand_flat.RemoveAllConformers()
    rdDepictor.Compute2DCoords(ligand_flat)

    # Save the ligand
    ligand_flat_path = dm.fs.join(datum_data_dir, "ligand_flat.sdf")
    ligand_path = dm.fs.join(datum_data_dir, "ligand.sdf")
    dm.to_sdf(ligand, ligand_path)
    dm.to_sdf(ligand_flat, ligand_flat_path)

    ## Step 2: extract the pocket and save it as a PDB file

    # Split the ligand out
    pocket_pdb = copy.deepcopy(input_pdb)

    # Init a modeller object to edit the structure
    modeller = Modeller(pocket_pdb.topology, pocket_pdb.positions)

    # Residue name considered as part of a receptor
    receptor_residue_names = get_receptor_residue_names(
        allow_metals=False,
        allow_unknowns=False,
    )

    # Delete residues that are not AA
    is_not_receptor_residue = lambda res: res.name not in receptor_residue_names
    modeller = delete_residues(modeller, is_not_receptor_residue)

    # Get the positions of the ligand and the target
    ligand_pos = dm.get_atom_positions(ligand)

    # Get the positions of the target
    target_pos = modeller.getPositions()
    target_pos = target_pos.in_units_of(unit.angstroms)
    target_pos = np.array([[v.x, v.y, v.z] for v in target_pos])

    # Compute the pairwise distances between the ligand's atoms and the target's atoms
    d = spatial.distance.cdist(target_pos, ligand_pos).min(axis=1)

    # Flag as True any target's atoms inside the cutoff
    target_atoms_inside = d <= distance_cutoff

    # Collect the residues to delete from the target
    residues_to_delete = []

    # NOTE(hadim): The below could probably be better vectorized but it's fast enough for now.
    for residue in modeller.topology.residues():

        # Collect all the atoms indices
        atom_indices = [a.index for a in residue.atoms()]

        # If at least one atom is within the cuttoff, the whole residue is kept
        if not target_atoms_inside[atom_indices].any():
            residues_to_delete.append(residue)

    # Delete the residues
    modeller.delete(residues_to_delete)

    # Save the pocket PDB file
    pocket_path = dm.fs.join(datum_data_dir, "pocket.pdb")
    with fsspec.open(pocket_path, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

def get_pdb_bind():
    """Download and extract archives from PDBbind."""

    c_folder = os.path.join(cache_dir, "pdbbind")
    os.makedirs(c_folder, exist_ok=True)

    # download and read files
    for url in file_urls:
        local_file = os.path.join(c_folder, url.split("/")[-1])

        # download and save file into the cache folder
        if os.path.exists(local_file):
            logger.info(f"File already downloaded at {local_file}")
        else:
            logger.info("Downloading", url)
            wget.download(url, local_file)

            with tarfile.open(local_file, "r") as f:
                f.extractall(c_folder)
    
def get_misato():
    fname = os.path.join(misato_dir, "MD.hdf5")

    fd = fsspec.open(fname, mode="rb")
    if hasattr(fd, "open"):
        fd = fd.open()
    md_data = h5py.File(fd)
    complex_codes = list(md_data.keys())
    np.random.shuffle(complex_codes)
    print(len(complex_codes))
    for code in complex_codes[:300]:
        MisatoMD(code, md_data, 0)


def create_pdbbind_set(cache_folder):
    """Create a set of PDBbind ligands."""
    pass


if __name__ == "__main__":
    # get_pdb_bind("cache")
    get_misato()
