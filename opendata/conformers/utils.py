import mdtraj
import numpy as np
from loguru import logger
from typing import Union

import openmm.unit as unit
from opendata.utils import flatten
from openff.toolkit.utils.exceptions import (
    UnassignedValenceParameterException,
    UnassignedProperTorsionParameterException,
)
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule, Topology


def filter_by_rmsd(conformer_positions, topology, n_conformers: int = 25, center_positions: bool = True):
    """
    Reduce the total set of conformers.

    Parameters
    ----------
    conformer_positions: np.ndarray
        Conformers to filter
    topology: simtk.openmm.app.Topology
        The topology of the molecule
    n_conformers: int
        The number of conformers to keep
    center_positions: bool
        Whether to center the conformers after calculating the RMSD and filtering
    Source: https://github.com/openmm/spice-dataset/blob/main/pubchem/createPubchem.py#L11-L23
    NOTE (Cas): Could be replaced by similar functionality in Datamol to remove the mdtraj dependency
    """

    # working_with_states = hasattr(conformer_states[0], "getPositions")
    # if working_with_states:
    #     xyz = np.array([s.getPositions().value_in_unit(unit.nanometer) for s in conformer_states])
    # else:
    #     xyz = np.array(conformer_states)

    traj = mdtraj.Trajectory(conformer_positions, mdtraj.Topology.from_openmm(topology))
    traj.center_coordinates()

    final_states = {0}
    min_rmsd = mdtraj.rmsd(traj, traj, 0, precentered=True)

    for _ in range(n_conformers - 1):
        best = np.argmax(min_rmsd)
        min_rmsd = np.minimum(min_rmsd, mdtraj.rmsd(traj, traj, best, precentered=True))
        final_states.add(best)

    res = [conformer_positions[i] for i in final_states]
    res = [c - c.mean(axis=0) for c in res]

    return res


def get_topology_system(mol: Union[str, Molecule], ff_engine: ForceField):
    """
    Initiates a MD simulation system given a molecule and a ForceField.

    Parameters
    ----------
    mol: Union[str, Molecule]
        The molecule to simulate. If a string is provided, it is assumed to be a SMILES string and the user is for defining the stereochemistry.
    ff_engine: openff.toolkit.typing.engines.smirnoff.ForceField
        The force field to use for the simulation.

    Returns
    -------
    openmm_topology: openmm.app.Topology
        The OpenMM topology for the molecule.
    system: openmm.System
        The OpenMM system for the molecule.

    Source: https://github.com/openmm/spice-dataset/blob/main/pubchem/createPubchem.py#L28-L36
    """
    if isinstance(mol, str):
        mol = Molecule.from_smiles(mol, allow_undefined_stereo=False)

    # Parameterize the simulation
    openff_topology = Topology()
    openff_topology.add_molecule(mol)
    openmm_topology = openff_topology.to_openmm()

    try:
        system = ff_engine.create_openmm_system(openff_topology)
    except (
        UnassignedValenceParameterException,
        UnassignedProperTorsionParameterException,
        ValueError,
    ) as error:
        logger.warning(f"Failed to setup the simulation for {mol.to_smiles()} due to a {type(error)}.")
        return
    return openmm_topology, system

