import os
import shutil
import openmm
import fsspec
import numpy as np
import datamol as dm
import pickle as pkl
import openmm.unit as unit
from openmm import app, State
from openmm.unit import Quantity
from opendata.utils import get_local_cache


def parallel_tempering_md(
        starting_positions,
        topology: app.Topology,
        system: openmm.System,
        outpath_prefix: str,
        temperatures: list=[300, 500, 1000, 1500],
        n_steps: int = 1e4,
        step_size: Quantity = 2 * unit.femtosecond,
        save_every: int = 10,
        friction_coeff: Quantity = 1 / unit.picosecond,
        max_minimization_steps: int = 0,
        n_jobs=1,
):
    """
    Run parallel tempering MD simulations to generate conformers.

    Parameters
    ----------
    starting_positions: Union[Union[Quantity, State], list[Union[Quantity, State]]]
        The initial conformer(s) used as starting point for the simulation
    topology: openmm.app.Topology
        The topology for the molecule
    system: openmm.System
        The system that is being simulated
    n_conformers: int
        The max number of conformers to return
    temperatures: list
        The temperatures for the LangevinMiddleIntegrator
    n_steps: int
        The number of simulation steps
    step_size: Quantity
        The step_size for the LangevinMiddleIntegrator
    save_every: int
        The frequency at which to save the conformers
    friction_coeff: Quantity
        The friction_coeff for the LangevinMiddleIntegrator
    max_minimization_steps: int
        The max number of energy minimization iterations
    n_jobs: int
        The number of jobs to run in parallel. If -1, use all available cores.

    Returns
    -------
    conformers: List
        List of generated conformers.
    energies: List
        List of energies for each conformer.
    """

    fn = lambda xt: run_single_md(starting_positions, # xt[0].to_openmm(), 
                                  temperature=xt * unit.kelvin, 
                                  topology=topology, system=system,
                                  n_steps=n_steps, step_size=step_size,
                                  friction_coeff=friction_coeff, 
                                  max_minimization_steps=max_minimization_steps,
                                  save_every=save_every)
    res = dm.utils.parallelized(
        fn,
        [t for t in temperatures ],
        n_jobs=n_jobs,
    )

    positions = [x[0] for x in res]
    energies = [x[1] for x in res]
    res = dict(temperatures=temperatures, positions=positions, energies=energies)

    out_dir = os.path.join(get_local_cache(), "md/parallel_tempering", outpath_prefix)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    with fsspec.open(os.path.join(out_dir, "output.pkl"), "wb") as fd:
        pkl.dump(res, fd)

    return np.concatenate(positions), np.concatenate(energies)


def run_single_md(
    starting_positions,
    topology,
    system,
    n_steps: int = 1e4,
    step_size: Quantity = 0.001 * unit.picosecond,
    save_every: int = 10,
    temperature: Quantity = 500 * unit.kelvin,
    friction_coeff: Quantity = 1 / unit.picosecond,
    max_minimization_steps: int = 0,
):
    """
    Find conformers through a MD simulation from a given start confirmation

    Parameters
    ----------
    starting_conformation: Union[Quantity, State, NDarray]
        The initial conformer used as starting point for the simulation
    topology: openmm.app.Topology
        The topology for the molecule
    system: openmm.System
        The system that is being simulated
    n_replicates: int
        The max number of conformers to return
    n_steps: int
        The number of simulation steps
    step_size: Quantity
        The step_size for the LangevinMiddleIntegrator
    save_every: int
        The frequency at which to save the conformers
    temperature: Quantity
        The temperature for the LangevinMiddleIntegrator
    friction_coeff: Quantity
        The friction_coeff for the LangevinMiddleIntegrator
    max_minimization_steps: int
        The max number of energy minimization iterations

    Source: https://github.com/openmm/spice-dataset/blob/main/pubchem/createPubchem.py#L44-L56
    """
    
    integrator = openmm.LangevinMiddleIntegrator(temperature, friction_coeff, step_size)

    # Setup the simulation
    simulation = app.Simulation(topology, system, integrator, 
                                platform=openmm.Platform.getPlatformByName("Reference"))

    if isinstance(starting_positions, State):
        simulation.context.setState(starting_positions)
    else:
        simulation.context.setPositions(starting_positions)

    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.minimizeEnergy(max_minimization_steps)

    # Run the simulation
    positions, energies = [], []
    n = int(n_steps // save_every)
    for _ in range(n+1):
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        positions.append(state.getPositions(asNumpy=True))
        energies.append(state.getPotentialEnergy())
        simulation.step(save_every)
        # if energy_threshold is None or state.getPotentialEnergy() < energy_threshold:
        #     states.append(state)

    return np.array(positions), np.array(energies)

