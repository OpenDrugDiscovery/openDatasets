import mdtraj
import openmm
import numpy as np
import datamol as dm

from loguru import logger
from typing import Union, Optional

import openmm.unit as unit
from openmm import app, State
from openmm.unit import Quantity
from opendata.collections.utils import flatten
from openff.toolkit.utils.exceptions import (
    UnassignedValenceParameterException,
    UnassignedProperTorsionParameterException,
)
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule, Topology


def generate_conformers_from_smiles(
    smi: str,
    ff_spec: str = "openff_unconstrained-2.0.0.offxml",
    temperatures: list=[300, 500, 1000, 1500],
    n_conformers: int = None
) -> Union[Molecule, None]:
    """
    Generate the conformations for a molecule given a smiles string.

    Parameters
    ----------
    smi: str
        The smiles string of the molecule
    ff_engine: str
        The force field engine to use for the simulation
    n_conformers: int
        The total number of conformers to generate

    Returns
    -------
    mol: openff.toolkit.topology.Molecule
        The molecule with the generated conformers

    Source: https://github.com/openmm/spice-dataset/blob/main/pubchem/createPubchem.py#L25-L83
    """
    ff_engine = ForceField(ff_spec)

    # Parameterize the system
    mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    ret = mol_to_openmm_topology_and_system(mol, ff_engine)
    
    if n_conformers is None:
        n_conformers = int((len(mol.find_rotatable_bonds()) + 3) **2.5)
        n_conformers = min(5000, n_conformers)
    

    if ret is None:
        mol, all_conformers = None, []
    else:
        topology, system = ret
        # Generating {n_starting_points} starting points
        mol.generate_conformers(n_conformers=10, rms_cutoff=None)

        fn = lambda xt: md_simulate_conformers(xt[0].to_openmm(), temperature=xt[1] * unit.kelvin, 
                                               topology=topology, system=system)
        all_conformers = dm.utils.parallelized(
            fn,
            [(x, t) for x in mol.conformers for t in temperatures ],
            progress=True,
        )
        all_conformers = flatten(all_conformers)

        # Select a subset that is maximally different from one another
        all_conformers = filter_by_rmsd(all_conformers, topology, n_conformers=n_conformers)

        # Set all the generated conformers for the mol
        mol._conformers = None
        for state in all_conformers:
            c = state.getPositions(asNumpy=True)
            mol.add_conformer(c - c.mean(axis=0))
        logger.info(f"{len(mol.conformers)}")

    return mol


def filter_by_rmsd(conformer_states, topology, n_conformers: int = 25):
    """
    Reduce the total set of conformers.

    Parameters
    ----------
    conformer_states: list of simtk.openmm.app.Simulation
        The set of conformers to filter
    topology: simtk.openmm.app.Topology
        The topology of the molecule
    n_conformers: int
        The number of conformers to keep
    Source: https://github.com/openmm/spice-dataset/blob/main/pubchem/createPubchem.py#L11-L23
    NOTE (Cas): Could be replaced by similar functionality in Datamol to remove the mdtraj dependency
    """

    xyz = np.array([s.getPositions().value_in_unit(unit.nanometer) for s in conformer_states])

    traj = mdtraj.Trajectory(xyz, mdtraj.Topology.from_openmm(topology))
    traj.center_coordinates()

    final_states = {0}
    min_rmsd = mdtraj.rmsd(traj, traj, 0, precentered=True)

    for i in range(n_conformers - 1):
        best = np.argmax(min_rmsd)
        min_rmsd = np.minimum(min_rmsd, mdtraj.rmsd(traj, traj, best, precentered=True))
        final_states.add(best)

    return [conformer_states[i] for i in final_states]


def mol_to_openmm_topology_and_system(mol: Union[str, Molecule], ff_engine: ForceField):
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


def md_simulate_conformers(
    starting_conformation: Union[Quantity, State],
    topology,
    system,
    n_replicates: int = 10,
    n_simulation_steps: int = 1e4,
    energy_threshold: Optional[Quantity] = 1e4 * unit.kilojoules_per_mole,
    temperature: Quantity = 500 * unit.kelvin,
    friction_coeff: Quantity = 1 / unit.picosecond,
    step_size: Quantity = 0.001 * unit.picosecond,
    max_energy_minimization_iters: int = 0,
):
    """
    Find conformers through a MD simulation from a given start confirmation

    Parameters
    ----------
    starting_conformation: Union[Quantity, State]
        The initial conformer used as starting point for the simulation
    topology: openmm.app.Topology
        The topology for the molecule
    system: openmm.System
        The system that is being simulated
    n_replicates: int
        The max number of conformers to return
    n_simulation_steps: int
        The number of simulation steps
    energy_threshold: Optional[Quantity]
        The energy threshold for a conformer to be considered low energy.
    temperature: Quantity
        The temperature for the LangevinMiddleIntegrator
    friction_coeff: Quantity
        The friction_coeff for the LangevinMiddleIntegrator
    step_size: Quantity
        The step_size for the LangevinMiddleIntegrator
    max_energy_minimization_iters: int
        The max number of energy minimization iterations

    Source: https://github.com/openmm/spice-dataset/blob/main/pubchem/createPubchem.py#L44-L56
    """

    # logger.info(f"{temperature}")
    # logger.info(f"{friction_coeff}")
    # logger.info(f"{step_size}")
    # # print(temperature, friction_coeff, step_size)
    # exit()
    integrator = openmm.LangevinMiddleIntegrator(temperature, friction_coeff, step_size)

    # Setup the simulation
    simulation = app.Simulation(topology, system, integrator, 
                                platform=openmm.Platform.getPlatformByName("Reference"))

    if isinstance(starting_conformation, State):
        simulation.context.setState(starting_conformation)
    else:
        simulation.context.setPositions(starting_conformation)

    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.minimizeEnergy(max_energy_minimization_iters)

    # Run the simulation
    states = []
    for i in range(n_replicates):
        simulation.step(n_simulation_steps)
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        if energy_threshold is None or state.getPotentialEnergy() < energy_threshold:
            states.append(state)
    return states

