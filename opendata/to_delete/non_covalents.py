import os
import sys
import json
import time
import tqdm
import mdtraj
import fsspec
import openmm
import parmed
import pandas as pd
import numpy as np
import datamol as dm
from typing import Union
from loguru import logger
from rdkit import Chem
from simtk import unit, Quantity
from typing import Union, Optional
from openmm.app import Simulation, PDBFile, DCDReporter, StateDataReporter, LangevinIntegrator, NoCutoff, HBonds, Platform, Modeller
from openff.toolkit.typing.engines.smirnoff import ForceField
from openforcefields.generators import SystemGenerator
from opendata.to_delete.base import QCFractalOptDataset
from openff.toolkit.topology import Molecule, Topology


def get_platform():
    # check whether we have a GPU platform and if so set the precision to mixed
    speed = 0
    for i in range(Platform.getNumPlatforms()):
        p = Platform.getPlatform(i)
        # print(p.getName(), p.getSpeed())
        if p.getSpeed() > speed:
            platform = p
            speed = p.getSpeed()

    if platform.getName() == 'CUDA' or platform.getName() == 'OpenCL':
        platform.setPropertyDefaultValue('Precision', 'mixed')
        print('Set precision for platform', platform.getName(), 'to mixed')

    return platform


def load_pocket_and_ligand(pdb_file, mol_file):
    # Read the molfile into RDKit, add Hs and create an openforcefield Molecule object 
    # And ensure the chiral centers are all defined
    mol =  Chem.MolFromMolFile(mol_file)
    molh = Chem.AddHs(mol, addCoords=True)
    Chem.AssignAtomChiralTagsFromStructure(molh)
    ligand_mol = Molecule(molh)

    protein_pdb = PDBFile(pdb_file)
    # Use Modeller to combine the protein and ligand into a complex
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    complex = dict(modeller=modeller, ligand=ligand_mol, protein=protein_pdb)
    return complex
    

def run_complex_md_simulation(
    complex: dict, 
    h5_trajectory_file: str,
    protein_ff_spec: str = 'amber/ff14SB.xml',
    ligand_ff_spec: str = 'gaff-2.11',
    n_simulation_steps: int = 1e4,
    equilibration_steps = 500,
    reporting_interval = 100,
    temperature: Quantity = 500 * unit.kelvin,
    friction_coeff: Quantity = 1 / unit.picosecond,
    step_size: Quantity = 0.001 * unit.picosecond,
    ):
    platform = get_platform()

    # Initialize a SystemGenerator using the GAFF for the ligand
    logger.info('Preparing system')
    forcefield_kwargs = {'constraints': HBonds, 
                         'rigidWater': False, 
                         'removeCMMotion': False, 
                         'hydrogenMass': 4*unit.amu }
    system_generator = SystemGenerator(
        forcefields=[protein_ff_spec],
        small_molecule_forcefield=ligand_ff_spec,
        forcefield_kwargs=forcefield_kwargs)
    

    modeller, ligand = complex['modeller'], complex['ligand']
    system = system_generator.create_system(modeller.topology, molecules=ligand)

    integrator = LangevinIntegrator(temperature, friction_coeff, step_size)
    # system.addForce(openmm.MonteCarloBarostat(1*unit.atmospheres, temperature, 25))
    logger.info(f'Uses Periodic box: {system.usesPeriodicBoundaryConditions()}')
    logger.info(f'Default Periodic box:, {system.getDefaultPeriodicBoxVectors()}')

    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    simulation.context.setPositions(modeller.positions)
    logger.info('Minimising ...')
    simulation.minimizeEnergy()

    # equilibrate
    simulation.context.setVelocitiesToTemperature(temperature)
    logger.info('Equilibrating ...')
    simulation.step(equilibration_steps)

    # Run the simulation.
    # The enforcePeriodicBox arg to the reporters is important.
    # It's a bit counter-intuitive that the value needs to be False, but this is needed to ensure that
    # all parts of the simulation end up in the same periodic box when being output.
    h5_reporter = mdtraj.reporters.HDF5Reporter(h5_trajectory_file, reporting_interval, velocities=True, enforcePeriodicBox=False)
    state_reporter = StateDataReporter(sys.stdout, reporting_interval, step=True, potentialEnergy=True, temperature=True)
    simulation.reporters.extend([h5_reporter, state_reporter])
    logger.info(f'Starting simulation with {n_simulation_steps} steps ...')
    t0 = time.time()
    simulation.step(n_simulation_steps)
    logger.info(f'Simulation complete in {time.time() - t0} seconds at {temperature} K')
    return simulation


def extract_conformers(h5_trajectory_file, n_conformers=100):

    traj = mdtraj.load_hdf5(h5_trajectory_file)
    traj.center_coordinates()

    final_states = {0}
    min_rmsd = mdtraj.rmsd(traj, traj, 0, precentered=True)

    for _ in range(n_conformers - 1):
        best = np.argmax(min_rmsd)
        min_rmsd = np.minimum(min_rmsd, mdtraj.rmsd(traj, traj, best, precentered=True))
        final_states.add(best)

    return [traj[i] for i in final_states]


class RefinedSetProteinLigand(QCFractalOptDataset):
    
    def __init__(self, server_info_file: Union[str, None] = None, ff_spec: str = "openff_unconstrained-2.0.0.offxml"):
        super().__init__(server_info_file)
        self.ff_spec = ff_spec
        self.ff_engine = ForceField(ff_spec)



        rs_folder = os.path.join(cache_folder, "pdbbind", "refined-set")

        pdbs = os.listdir(rs_folder)
        pocket_ligand_files = [(pdb_code,
                                os.path.join(rs_folder, pdb_code, f"{pdb_code}_pocket.pdb"),
                                os.path.join(rs_folder, pdb_code, f"{pdb_code}_ligand.mol2"),
                                os.path.join(rs_folder, pdb_code, f"{pdb_code}_pocket.pdb"))
                               for pdb_code in pdbs]
        
        pocket_ligand_files = [f for f in pocket_ligand_files if os.path.exists(f[1]) and os.path.exists(f[2])]

        

    def load_refined_set(self):

        for pdb_code, pocket_file, ligand_file, traj_file in tqdm.tqdm(pocket_ligand_files):

            complex = load_pocket_and_ligand(pocket_file, ligand_file)
            simulation = run_complex_md_simulation(complex, 
                h5_trajectory_file=traj_file,)
            conformers = extract_conformers(traj_file, n_conformers=100)



if __name__ == "__main__":
    

