import pdbfixer

from openmm import app
import openmm.unit as unit
from openmm.app import PDBxFile
from openmm.app import Modeller
from typing import Callable
from loguru import logger
import copy
import fsspec

from scipy import spatial

import datamol as dm
import numpy as np

from rdkit.Chem import rdDepictor  # type: ignore
from Bio.PDB import PDBParser
import mdtraj as md
from openmm.app import PDBFile
from pydantic import BaseModel, Extra, Field

import mdtraj as md 


class PocketExtractor(BaseModel):
    """
    Args:
        distance_cutoff: the distance cutoff to use to extract the pocket in Angstrom.
        remove_water: whether to remove water molecules or not.
        add_hs: whether to add hydrogens or not.
    """

    class Config:
        extra = Extra.forbid

    distance_cutoff: float = Field(
        5.0, description="the distance cutoff to use to extract the pocket in Angstrom."
    )
    remove_water: bool = Field(True, description="whether to remove water molecules or not.")
    add_hs: bool = Field(True, description="whether to add hydrogens or not.")

    def __call__(self, modeller: Modeller, filepath="modfied_pdb.pdb", save_pdb: bool = False, as_df=True):
        modified_modeller = extract_pocket(modeller, self.distance_cutoff, self.remove_water, self.add_hs)
        if save_pdb:
            with fsspec.open(filepath, "w") as f:
                app.PDBFile.writeFile(
                    modified_modeller.topology, modified_modeller.positions, f, keepIds=True
                )
        pos =  to_array(modified_modeller.positions)
        if as_df:
            df,_ = md.Topology.from_openmm(modified_modeller.topology).to_dataframe()
            df["positions"]=pos.tolist()
            return df
        return {"positions": pos, "topology": md.Topology.from_openmm(modified_modeller.topology).to_dataframe()[0]}

    def from_pdb(self, pdb_path: str, filepath="modfied_pdb.pdb", save_pdb: bool = False, as_df=False):
        modeller = get_modeller(pdb_path)
        return self(modeller, filepath, save_pdb, as_df)


def get_receptor_residue_names(
    allow_metals: bool = False, allow_unknowns: bool = False, metals: list[str] = ["NA"]
):
    """Return a set of residue names that are considered as being part of the receptor.

    Args:
        allow_metals: consider mettals as part of the receptor.
        allow_unknowns: consider UNK residue as part of the receptor.
    """

    receptor_residue_names = set(pdbfixer.pdbfixer.proteinResidues)
    receptor_residue_names |= set(pdbfixer.pdbfixer.dnaResidues)
    receptor_residue_names |= set(pdbfixer.pdbfixer.rnaResidues)
    receptor_residue_names.add("N")

    if allow_unknowns:
        receptor_residue_names.add("UNK")

    if allow_metals:
        receptor_residue_names |= {*metals}

    return receptor_residue_names


def delete_residues(
    modeller: app.Modeller,
    delete_fn: Callable,
    copy_modeller: bool = True,
):
    """Delete residues provided by a function `delete_fn` that return
    True for each residues that needs to be deleted. A new modeller
    with deleted residues is returned.
    Args:
        modeller: the modeller object to deleted the residues from.
        delete_fn: a function that returns a boolean whether a given residue
            should be deleted.
        copy_modeller: whether to copy the modeller or not. It has implication
            when testing the equality between two residue objects.
    """
    if copy_modeller:
        modeller = copy.deepcopy(modeller)
    to_delete = filter(lambda res: delete_fn(res), modeller.topology.residues())
    modeller.delete(to_delete)
    return modeller


def load_pdb(pdb_path: str):
    with fsspec.open(pdb_path, "r") as f:
        try:
            input_pdb = PDBFile(f)
        except:
            input_pdb = PDBxFile(f)
    return input_pdb


def get_modeller(pdb_path: str):
    pdb = load_pdb(pdb_path)
    return Modeller(pdb.topology, pdb.positions)


def extract_pocket(modeller: Modeller, distance_cutoff=5.0, remove_water=True, add_hs=True):
    if remove_water:
        modeller.deleteWater()

    # Get the positions of the ligand
    traj = trajectory(modeller)
    ligand_idx = get_indexs_mol(traj)
    ligand_pos = traj.xyz[0][ligand_idx] * 10  # type: ignore

    # Get the positions of the target
    target_pos = modeller.getPositions()
    target_pos = target_pos.in_units_of(unit.angstroms)
    target_pos = np.array([[v.x, v.y, v.z] for v in target_pos])

    # NOTE(Cristian): this is fine for a single frame but for a trajectory will probably be better to use something with  md.compute_neighbors
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
    modeller.addHydrogens()
    return modeller


def to_array(quant: unit.Quantity) -> np.ndarray:
    return np.array(quant / quant.unit)


def trajectory(m: Modeller) -> md.Trajectory:
    return md.Trajectory(to_array(m.positions), md.Topology.from_openmm(m.topology))


def get_indexs_mol(trajectory: md.Trajectory, resname: str = "MOL") -> list[int]:
    idx = trajectory.top.select(f"resname {resname}")  # type: ignore
    if idx.size == 0:
        idx = trajectory.top.select(f"resname '{resname}'")  # type: ignore
        if idx.size == 0:
            logger.error("Selection not found for resname: {resname}")
            raise ValueError
    return idx.tolist()