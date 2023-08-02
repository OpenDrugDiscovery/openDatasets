import mdtraj
import openmm
import numpy as np
import datamol as dm

from loguru import logger
from typing import Union, Optional, List

import openmm.unit as unit
from openmm import app, State
from openmm.unit import Quantity
from opendata.utils import flatten
from openff.toolkit.utils.exceptions import (
    UnassignedValenceParameterException,
    UnassignedProperTorsionParameterException,
)
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule, Topology
from opendata.conformers.simple_md import parallel_tempering_md
from opendata.conformers.replica_exchange import run_replica_exchange
from opendata.conformers.utils import filter_by_rmsd, get_topology_system


class ConformerGenerator():
    
    def __init__(self, 
                 use_replica_exchange=True,
                 ff_spec: str = "openff_unconstrained-2.0.0.offxml",
                 temperatures: list=[300, 500, 1000, 1500],
                 n_conformers: int = None
                 ) -> None:
        """
        Parameters
        ----------
        use_replica_exchange : bool, optional
            Whether to use replica exchange, by default True
        ff_spec : str, optional
            The force field engine to use for the simulation, by default "openff_unconstrained-2.0.0.offxml"
        temperatures : list, optional
            The temperatures to use for the simulation, by default [300, 500, 1000, 1500]
        n_conformers : int, optional
            The total number of conformers to generate, by default None
        """
        self.use_replica_exchange = use_replica_exchange
        self.ff_spec = ff_spec
        self.temperatures = temperatures
        self.n_conformers = n_conformers
        self.ff_engine = ForceField(ff_spec)

    def from_smiles(self, smi: str) -> List:
        """
        Generate conformers from a SMILES string.

        Parameters
        ----------
        smi : str
            SMILES string.

        Returns
        -------
        conformers : List[Conformer]
            List of generated conformers.
        """
        
        # Parameterize the system
        mol = Molecule.from_smiles(smi, allow_undefined_stereo=False)

        if mol is not None:
            topology, system = get_topology_system(mol, self.ff_engine)
            res = self.from_openmm(mol, topology, system)
        else:
            res = None
        return res
    
    def from_topology(self, topology: Topology) -> List:
        """
        Generate conformers from a Topology.
        
        Parameters
        ----------
        topology : Topology
            OpenFF Topology.
        
        Returns
        -------
        conformers : List[Conformer]
            List of generated conformers.
        """
        mol = Molecule.from_topology(topology)
        if mol is not None:
            topology, system = get_topology_system(mol, self.ff_engine)
            res = self.from_openmm(mol, topology, system)
        else:
            res = None
        return res

    def from_openmm(self, mol, topology, system):
        """
        Generate conformers from an OpenMM topology and system.

        Parameters
        ----------
        mol : Molecule
            OpenFF Molecule.
        topology : openmm.Topology
            OpenMM topology.
        system : openmm.System
            OpenMM system.

        Returns
        -------
        conformers : List[Conformer]
            List of generated conformers.
        """

        if self.n_conformers is None:
            n_conformers = int((len(mol.find_rotatable_bonds()) + 3) **2.5)
            n_conformers = min(5000, n_conformers)
        else:
            n_conformers = self.n_conformers

        # Generating {n_starting_points} starting points
        mol.generate_conformers(n_conformers=2, rms_cutoff=None)
        starting_points = mol.conformers[0].to_openmm()

        # Select a subset that is maximally different from one another
        if self.use_replica_exchange:
            all_conformers, _ = run_replica_exchange(starting_positions=starting_points,
                                        topology=topology,
                                        system=system,
                                        temperatures=self.temperatures,)
        else:
            all_conformers, _ = parallel_tempering_md(starting_positions=starting_points,
                                        topology=topology,
                                        system=system,
                                        temperatures=self.temperatures,)

        # Select a subset that is maximally different from one another
        all_conformers = filter_by_rmsd(all_conformers, topology, 
                                        n_conformers=self.n_conformers, center_positions=True)

        # Set all the generated conformers for the mol
        mol._conformers = None
        for c in all_conformers:
            mol.add_conformer(c)

        return mol