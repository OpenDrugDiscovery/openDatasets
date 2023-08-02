# Adapted from https://github.com/shirtsgroup/cg_openmm/blob/master/cg_openmm/simulation/rep_exch.py
import os
import shutil
from typing import Any
import numpy as np
import numpy.ma as ma
import openmmtools
from openmm import unit
from loguru import logger
from openmm.unit import is_quantity, Quantity
from openmmtools.multistate import (MultiStateReporter, 
                                    MultiStateSampler,
                                    ReplicaExchangeSampler)
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.mcmc import LangevinDynamicsMove
from opendata.conformers.utils import filter_by_rmsd


# silent citation spam
MultiStateSampler._global_citation_silence = True

kB = (unit.MOLAR_GAS_CONSTANT_R).in_units_of(unit.kilojoule / (unit.kelvin * unit.mole))


def run_replica_exchange(
    topology,
    system,
    starting_positions,
    out_dir,
    n_steps: int = 1e4,
    step_size: Quantity = 0.001 * unit.picosecond,
    save_every: int = 10,
    temperatures = None,
    friction_coeff: Quantity = 1 / unit.picosecond,
    max_minimization_steps: int = 0,
    exchange_frequency=1000,
        ):
    """
    Find conformers through a MD simulation from a given start confirmation
    
    Parameters
    ----------
    topology: openmm.app.Topology
        The topology for the molecule
    system: openmm.System
        The system that is being simulated
    starting_positions: np.ndarray
        The initial conformer used as starting point for the simulation
    n_steps: int
        The number of simulation steps
    step_size: Quantity
        The step_size for the LangevinMiddleIntegrator
    save_every: int
        The frequency at which to save the conformers
    temperatures: list
        The temperatures for the ReplicaExchangeSampler
    friction_coeff: Quantity
        The friction_coeff for the LangevinMiddleIntegrator
    max_minimization_steps: int
        The max number of energy minimization iterations
    exchange_frequency: int
        The frequency at which to exchange thermodynamic states

    Returns
    -------
    positions: np.ndarray
        The positions of the conformers
    energies: np.ndarray
        The energies of the conformers
    """     
    exchange_attempts = int(np.floor(n_steps / exchange_frequency))
    if temperatures is None:
        temperatures = [((300.0 + i) * unit.kelvin) for i in range(0, 1200, 100)]

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    reporter_filepath = os.path.join(out_dir, "reporter.nc")
    reporter = MultiStateReporter(reporter_filepath, checkpoint_interval=1)

    # Create sampler and thermodynamic states and configure simulation object.
    sampler_states = [SamplerState(starting_positions) for _ in temperatures]
    thermodynamic_states = [ThermodynamicState(system=system, temperature=t) 
        for t in temperatures]
        
    move = LangevinDynamicsMove(timestep=step_size, 
                                collision_rate=friction_coeff,
                                n_steps=save_every, reassign_velocities=False)

    simulation = ReplicaExchangeSampler(mcmc_moves=move, 
                                        number_of_iterations=exchange_attempts,
                                        replica_mixing_scheme='swap-neighbors')
    simulation.create(thermodynamic_states, sampler_states, reporter)
    simulation.minimize(max_iterations=max_minimization_steps)

    try:
        simulation.run()
    except BaseException:
        logger.info("Replica exchange simulation failed! Verify your simulation settings.")
        simulation = None

    # n_iterations x n_replicas x n_atoms x 3
    positions = reporter._storage_checkpoint.variables['positions'][:]
    positions = positions.astype(np.float32).reshape(-1, starting_positions.shape[-2], 3)
    positions = positions.filled(np.nan)

    # n_iterations x n_replicas x n_replicas
    tmp = reporter.read_energies()[0] * (kB *np.array(temperatures))[None, :, None] 
    energies = np.zeros(tmp.shape, dtype=np.float32) + np.nan
    for (i, j) in np.ndindex(tmp.shape[:2]):
        for k, x in enumerate(tmp[i][j]):
            v = float(str(x).split(' ')[0])
            energies[i, j, k] = v
    energies = energies.reshape(-1, energies.shape[-1])

    return positions, energies


if __name__ == "__main__":
    import openmm 
    from opendata import utils
    from openff.toolkit.typing.engines.smirnoff import ForceField
    from openff.toolkit.topology import Molecule
    from opendata.conformers.utils import get_topology_system
    from opendata.conformers.simple_md import parallel_tempering_md


    # setup the platform 
    from openmmtools.cache import global_context_cache
    global_context_cache.platform = openmm.Platform.getPlatformByName("CPU")

    smi = "CC(C(=O)NC(C)C(=O)O)N"
    ff_engine = ForceField("openff_unconstrained-2.0.0.offxml")

    # Parameterize the system
    mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    ret = get_topology_system(mol, ff_engine)

    topology, system = ret
    # Generating {n_starting_points} starting points
    mol.generate_conformers(n_conformers=1, rms_cutoff=None)

    positions = mol.conformers[0].to_openmm()

    cache = utils.get_local_cache()
    opath = os.path.join(cache, "re_test")

    re = run_replica_exchange(topology=topology, 
                         system=system, 
                         starting_positions=positions, 
                         out_dir=opath, 
                         n_steps=6000,
                         save_every=100,
                         step_size=2*unit.femtosecond,
                         exchange_frequency=179)
    
    print(re[0].shape, re[1].shape)
    re = parallel_tempering_md(starting_positions=positions,
                                topology=topology, 
                                system=system, 
                                n_steps=6000,
                                save_every=100,
                                step_size=2*unit.femtosecond,
                                )
    print(re[0].shape, re[1].shape)





