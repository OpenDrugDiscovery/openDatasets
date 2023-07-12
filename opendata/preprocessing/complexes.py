from typing import Callable

import copy
import fsspec

from scipy import spatial

import datamol as dm
import numpy as np

from rdkit.Chem import rdDepictor

import pdbfixer

from openmm import app
import openmm.unit as unit
from openmm.app import PDBxFile
from openmm.app import Modeller


def get_receptor_residue_names(allow_metals: bool = False, allow_unknowns: bool = False):
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
        # EN: metallic residues will fail with vina
        # hadim: add more metalic atoms here.
        receptor_residue_names |= {"NA"}

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


def prepare_inputs(
    datum_id: str,
    input_ligand_path: str,
    input_pdb_path: str,
    base_data_dir: str,
    distance_cutoff: float,
):
    """distance_cutoff is in angstroms"""

    # Make a dir for that data point
    datum_data_dir = dm.fs.join(base_data_dir, datum_id)
    dm.fs.mkdir(datum_data_dir, exist_ok=True)

    # Open the input PDB file
    with fsspec.open(dm.fs.join(base_data_dir, input_pdb_path), "r") as f:
        input_pdb = PDBxFile(f)

    # Save the original PDB file to the datum directory
    full_input_pdb_path = dm.fs.join(datum_data_dir, "original.pdb")
    with fsspec.open(full_input_pdb_path, "w") as f:
        app.PDBFile.writeFile(input_pdb.topology, input_pdb.positions, f, keepIds=True)

    ## Step 1: use the already extracted ligand SDF file

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

    # Remove water molecule
    modeller.deleteWater()

    # Residue name considered as part of a receptor
    receptor_residue_names = get_receptor_residue_names(
        allow_metals=False,
        allow_unknowns=False,
    )

    # Delete residues that are not AA
    is_not_receptor_residue = lambda res: res.name not in receptor_residue_names
    modeller = delete_residues(modeller, is_not_receptor_residue)

    # Get the positions of the ligand
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