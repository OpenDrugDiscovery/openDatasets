# Adapted from https://github.com/shirtsgroup/cg_openmm/blob/master/cg_openmm/simulation/rep_exch.py
import os
import time
from typing import Any
import numpy as np
import openmmtools
from openmm import unit
from loguru import logger
import matplotlib.cm as cm
import matplotlib.pyplot as pyplot
from matplotlib.colors import Normalize
from scipy.special import erf
from scipy.optimize import minimize_scalar
from pymbar import timeseries
from openmm.app.pdbfile import PDBFile
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mdtraj import Topology, Trajectory
from openmmtools.multistate import (MultiStateReporter, 
                                    MultiStateSampler,
                                    ReplicaExchangeAnalyzer,
                                    ReplicaExchangeSampler)

print("test")

# quiet down some citation spam
MultiStateSampler._global_citation_silence = True

kB = (unit.MOLAR_GAS_CONSTANT_R).in_units_of(unit.kilojoule / (unit.kelvin * unit.mole))

def extract_trajectory(
    topology, reporter,
    state_index=None, replica_index=None,
    frame_begin=0, frame_stride=1, frame_end=-1):
    """
    Internal function for extract trajectory (replica or state) from .nc file,
    Based on YANK extract_trajectory code.
    """

    # Get dimensions
    trajectory_storage = reporter._storage_checkpoint  
    n_iterations = reporter.read_last_iteration()
    n_frames = trajectory_storage.variables['positions'].shape[0]
    n_atoms = trajectory_storage.variables['positions'].shape[2]
    
    # Determine frames to extract.
    # Convert negative indices to last indices.
    if frame_begin < 0:
        frame_begin = n_frames + frame_begin
    if frame_end < 0:
        frame_end = n_frames + frame_end + 1
    frame_indices = range(frame_begin, frame_end, frame_stride)
    if len(frame_indices) == 0:
        raise ValueError('No frames selected')
        
    # Determine the number of frames that the trajectory will have.
    if state_index is None:
        n_trajectory_frames = len(frame_indices)        
    else:
        # With SAMS, an iteration can have 0 or more replicas in a given state.
        # Deconvolute state indices.
        state_indices = [None for _ in frame_indices]
        for i, iteration in enumerate(frame_indices):
            replica_indices = reporter._storage_analysis.variables['states'][iteration, :]
            state_indices[i] = np.where(replica_indices == state_index)[0]
        n_trajectory_frames = sum(len(x) for x in state_indices)        
        
    # Initialize positions and box vectors arrays.
    # MDTraj Cython code expects float32 positions.
    positions = np.zeros((n_trajectory_frames, n_atoms, 3), dtype=np.float32)

    # Extract state positions and box vectors.
    if state_index is not None:
        # Extract state positions
        frame_idx = 0
        for i, iteration in enumerate(frame_indices):
            for replica_index in state_indices[i]:
                positions[frame_idx, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                frame_idx += 1

    else:  # Extract replica positions
        for i, iteration in enumerate(frame_indices):
            positions[i, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)

    return positions

class ReplicaExchange:
    def __init__(self,
        topology,
        system,
        positions,
        out_filepath,
        total_time=10.0 * unit.picosecond,
        step_time=1.0 * unit.femtosecond,
        temperatures=None,
        friction=1.0 / unit.picosecond,
        minimize=True,
        exchange_frequency=1000,
        overwrite = False,
        ):

        """
        Arguments
        ---------
    
        topology: OpenMM Topology

        system: OpenMM System()

        positions: Positions array for the model we would like to test

        total_time: Total run time for individual simulations

        step_time: Simulation integration time step

        temperatures: List of temperatures for which to perform replica exchange simulations, default = None

        friction: Langevin thermostat friction coefficient, default = 1 / ps

        minimize: Whether minimization is done before running the simulation

        exchange_frequency: Number of time steps between replica exchange attempts, Default = None

        """
        self.topology = topology
        self.system = system
        self.positions = positions
        self.out_filepath = out_filepath
        self.total_time = total_time
        self.step_time = step_time
        self.temperatures = temperatures
        self.friction = friction
        self.minimize = minimize
        self.exchange_frequency = exchange_frequency
        self.overwrite = overwrite

        
        if temperatures is None:
            self.temperatures = [((300.0 + i) * unit.kelvin) for i in range(0, 1200, 100)]

        self._init_reporter_()

    def reporter_to_dict(self):
        reporter = MultiStateReporter(self.out_filepath, open_mode="r")
       
        # figure out what the temperature list is
        states = reporter.read_thermodynamic_states()[0]

        analyzer = ReplicaExchangeAnalyzer(reporter)
        x = analyzer.read_energies()
        replica_energies, replica_state_indices = x[0], x[3]

        temperatures = [s.temperature for s in states]
        beta_k = np.array([ 1 / (kB * temp._value) for temp in temperatures])
        beta_k = np.array([i / i.unit for i in beta_k])
        replica_energies *= (beta_k ** (-1))[None, :, None]

        print(replica_energies.shape)
        print(replica_state_indices.shape)

        traj = self.get_trajectory()
        print(traj.shape)
        exit()

        return dict(reporter=reporter,
                    states=states,
                    energies=replica_energies, 
                    state_indices=replica_state_indices,
                    trajectory=traj)

    @property
    def n_replica(self):
        return len(self.temperatures)

    @property
    def n_steps(self):
        return int(np.floor(self.total_time / self.step_time))

    @property
    def exchange_attempts(self):
        return int(np.floor(self.n_steps / self.exchange_frequency))
    
    def _init_reporter_(self):
        if self.overwrite:
            if os.path.exists(self.out_filepath):
                os.remove(self.out_filepath)
        os.makedirs(os.path.dirname(self.out_filepath), exist_ok=True)
        self.reporter = MultiStateReporter(self.out_filepath, checkpoint_interval=1)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self):
        sampler_states = list()
        thermodynamic_states = list()

        # Define thermodynamic states.
        # box_vectors = system.getDefaultPeriodicBoxVectors()
        # no box vectors, non-periodic system.
        thermodynamic_states = [
            openmmtools.states.ThermodynamicState(
            system=self.system, temperature=temperature) 
            for temperature in self.temperatures]

        sampler_states = [
            openmmtools.states.SamplerState(self.positions)
            for temperature in self.temperatures]
         
        # Create and configure simulation object.
        move = openmmtools.mcmc.LangevinDynamicsMove(
            timestep=self.step_time,
            collision_rate=self.friction,
            n_steps=self.exchange_frequency,
            reassign_velocities=False,
        )

        simulation = ReplicaExchangeSampler(
            mcmc_moves=move,
            number_of_iterations=self.exchange_attempts,
            replica_mixing_scheme='swap-neighbors',
        )
        simulation.create(thermodynamic_states, sampler_states, self.reporter)

        if self.minimize:
            simulation.minimize()

        try:
            simulation.run()
        except BaseException:
            logger.info("Replica exchange simulation failed! Verify your simulation settings.")
            simulation = None
            
        self.simulation = simulation
 
    def get_trajectory(self):
        """
        Extract trajectory from reporter file as a (n_frames, n_replicate, n_atoms, 3) array
        """
        all_pos = self.reporter._storage_checkpoint.variables['positions']
        return all_pos[:, :, :, :].astype(np.float32)

    def process_replica_exchange_data(self,
        output_data="output/output.nc", output_directory="output", series_per_page=4,
        write_data_file=True, plot_production_only=False, print_timing=False,
        equil_nskip=1, frame_begin=0, frame_end=-1,
    ):
        """
        Read replica exchange simulation data, detect equilibrium and decorrelation time, and plot replica exchange results.
        
        output_data: path to output .nc file from replica exchange simulation, (default='output/output.nc')
        
        output_directory: path to which output files will be written (default='output')

        series_per_page: number of replica data series to plot per pdf page (default=4)
        
        write_data_file: Option to write a text data file containing the state_energies array (default=True)
        
        plot_production_only: Option to plot only the production region, as determined from pymbar detectEquilibration (default=False)

        equil_nskip: skip this number of frames to sparsify the energy timeseries for pymbar detectEquilibration (default=1) - this is used only when frame_begin=0 and the trajectory has less than 40000 frames.
        
        frame_begin: analyze starting from this frame, discarding all prior as equilibration period (default=0)
        
        frame_end: analyze up to this frame only, discarding the rest (default=-1).

        :returns:
            - replica_energies ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.float( [number_replicas,number_simulation_steps] ), simtk.unit ) ) - The potential energies for all replicas at all (printed) time steps
            - replica_state_indices ( np.int64( [number_replicas,number_simulation_steps] ), simtk.unit ) - The thermodynamic state assignments for all replicas at all (printed) time steps
            - production_start ( int - The frame at which the production region begins for all replicas, as determined from pymbar detectEquilibration
            - sample_spacing ( int - The number of frames between uncorrelated state energies, estimated using heuristic algorithm )
            - n_transit ( np.float( [number_replicas] ) ) - Number of half-transitions between state 0 and n for each replica
            - mixing_stats ( tuple ( np.float( [number_replicas x number_replicas] ) , np.float( [ number_replicas ] ) , float( statistical inefficiency ) ) ) - transition matrix, corresponding eigenvalues, and statistical inefficiency
        """
               
        reporter = MultiStateReporter(self.out_filepath, open_mode="r")
        
        # figure out what the time between output is.
        # We assume all use the same time step (which i think is required)
        mcmove = reporter.read_mcmc_moves()[0]
        time_interval = mcmove.n_steps*mcmove.timestep
       
        # figure out what the temperature list is
        states = reporter.read_thermodynamic_states()[0]
        temperatures = [s.temperature for s in states]

        analyzer = ReplicaExchangeAnalyzer(reporter)
        
        x = analyzer.read_energies()
        replica_energies, replica_state_indices = x[0], x[3]
                
        beta_k = np.array([ 1 / (kB * temp._value) for temp in temperatures])
        n_replicas = len(temperatures)
        beta_k = np.array([i / i.unit for i in beta_k])
        replica_energies *= (beta_k ** (-1))[None, :, None]
        print(beta_k)

        total_steps = len(replica_energies[0][0])
        state_energies = np.zeros([n_replicas, total_steps])

        # there must be some better way to do this as list comprehension.
        for step in range(total_steps):
            for state in range(n_replicas):
                state_energies[state, step] = replica_energies[
                    np.where(replica_state_indices[:, step] == state)[0], 0, step
                ]
                
        # can run physical-validation on these state_energies
        # Start of equilibrated data:
        t0 = np.zeros((n_replicas))
        # Statistical inefficiency:
        g = np.zeros((n_replicas))
        
        subsample_indices = {}
        
        # If sufficiently large, discard the first 20000 frames as equilibration period and use 
        # subsampleCorrelatedData to get the energy decorrelation time.
        if total_steps >= 40000:
            production_start=20000
            for state in range(n_replicas):
                subsample_indices[state] = timeseries.subsampleCorrelatedData(
                    state_energies[state][production_start:],
                    conservative=True)
                g[state] = subsample_indices[state][1]-subsample_indices[state][0]
        
        else:
            # For small trajectories, use detectEquilibration
            for state in range(n_replicas):
                t0[state], g[state], _ = timeseries.detect_equilibration(state_energies[state], nskip=equil_nskip)  
                # Choose the latest equil timestep to apply to all states    
                production_start = int(np.max(t0))
        
        # Assume a normal distribution (very rough approximation), and use mean plus
        # the number of standard deviations which leads to (n_replica-1)/n_replica coverage
        # For 12 replicas this should be the mean + 1.7317 standard deviations
        
        # x standard deviations is the solution to (n_replica-1)/n_replica = erf(x/sqrt(2))
        # This is equivalent to a target of 23/24 CDF value 
        
        print(f"Correlation times (frames): {g.astype(int)}")
        
        def erf_fun(x):
            return np.power((erf(x/np.sqrt(2))-(n_replicas-1)/n_replicas),2)
            
        # x must be larger than zero    
        opt_g_results = minimize_scalar(
            erf_fun,
            bounds=(0,10),
            method='bounded',
            )
        
        if not opt_g_results.success:
            print("Error solving for correlation time, exiting...")
            print(f"erf opt results: {opt_g_results}")
            exit()
        
        sample_spacing = int(np.ceil(np.mean(g)+opt_g_results.x*np.std(g)))
        
        t11 = time.perf_counter()
        if print_timing:
            print(f"detect equil and subsampling time: {t11-t10}")
                    
        print("state    mean energies  variance")
        for state in range(n_replicas):
            state_mean = np.mean(state_energies[state,production_start::sample_spacing])
            state_std = np.std(state_energies[state,production_start::sample_spacing])
            print(
                f"  {state:4d}    {state_mean:10.6f} {state_std:10.6f}"
            )

        t12 = time.perf_counter()
        
        if write_data_file == True:
            f = open(os.path.join(output_directory, "replica_energies.dat"), "w")
            for step in range(total_steps):
                f.write(f"{step:10d}")
                for replica_index in range(n_replicas):
                    f.write(f"{replica_energies[replica_index,replica_index,step]:12.6f}")
                f.write("\n")
            f.close()

        t13 = time.perf_counter()
        if print_timing:
            print(f"Optionally write .dat file: {t13-t12}")
                
        t14 = time.perf_counter()
        
        if plot_production_only==True:
            self.plot_energies(
                state_energies[:,production_start:],
                temperatures,
                series_per_page,
                time_interval=time_interval,
                time_shift=production_start*time_interval,
                file_name=f"{output_directory}/rep_ex_ener.pdf",
            )
            
            self.plot_energy_histograms(
                state_energies[:,production_start:],
                temperatures,
                file_name=f"{output_directory}/rep_ex_ener_hist.pdf",
            )

            self.plot_summary(
                replica_state_indices[:,production_start:],
                temperatures,
                series_per_page,
                time_interval=time_interval,
                time_shift=production_start*time_interval,
                file_name=f"{output_directory}/rep_ex_states.pdf",
            )
            
            self.plot_matrix(
                replica_state_indices[:,production_start:],
                file_name=f"{output_directory}/state_probability_matrix.pdf",
            )
            
        else:
            self.plot_energies(
                state_energies,
                temperatures,
                series_per_page,
                time_interval=time_interval,
                file_name=f"{output_directory}/rep_ex_ener.pdf",
            )
            self.plot_energy_histograms(
                state_energies,
                temperatures,
                file_name=f"{output_directory}/rep_ex_ener_hist.pdf",
            )
            self.plot_summary(
                replica_state_indices,
                temperatures,
                series_per_page,
                time_interval=time_interval,
                file_name=f"{output_directory}/rep_ex_states.pdf",
            )
            self.plot_matrix(
                replica_state_indices,
                file_name=f"{output_directory}/state_probability_matrix.pdf",
            )
        
        t15 = time.perf_counter()
        
        if print_timing:
            print(f"plotting time: {t15-t14}")
        
        # Analyze replica exchange state transitions
        # For each replica, how many times does the thermodynamic state go between state 0 and state n
        # For consistency with the other mixing statistics, use only the production region here
        
        replica_state_indices_prod = replica_state_indices[:,production_start:]
        
        # Number of one-way transitions from states 0 to n or states n to 0 
        n_transit = np.zeros((n_replicas,1))
        
        # Replica_state_indices is [n_replicas x n_iterations]
        for rep in range(n_replicas):
            last_bound = None
            for i in range(replica_state_indices_prod.shape[1]):
                if replica_state_indices_prod[rep,i] == 0 or replica_state_indices_prod[rep,i] == (n_replicas-1):
                    if last_bound is None:
                        # This is the first time state 0 or n is visited
                        pass
                    else:
                        if last_bound != replica_state_indices_prod[rep,i]:
                            # This is a completed transition from 0 to n or n to 0
                            n_transit[rep] += 1
                    last_bound = replica_state_indices_prod[rep,i]                
                            
        t16 = time.perf_counter()
        
        if print_timing:
            print(f"replica transition analysis: {t16-t15}")
            
        # Compute transition matrix from the analyzer
        mixing_stats = analyzer.generate_mixing_statistics(number_equilibrated=production_start)
        
        t17 = time.perf_counter()
        
        if print_timing:
            print(f"compute transition matrix: {t17-t16}")
            print(f"total time elapsed: {t17-t1}")

        # Close reporter/.nc file:
        reporter.close()

        return (replica_energies, replica_state_indices, production_start, sample_spacing, n_transit, mixing_stats)

    def make_replica_dcd_files(self,
        topology, timestep=5*unit.femtosecond, time_interval=200,
        output_dir="output", output_data="output.nc", checkpoint_data="output_checkpoint.nc",
        frame_begin=0, frame_stride=1, center=False):
        """
        Make dcd files from replica exchange simulation trajectory data.

        :param topology: OpenMM Topology
        :type topology: `Topology() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1app_1_1topology_1_1Topology.html>`_

        :param timestep: Time step used in the simulation (default=5*unit.femtosecond)
        :type timestep: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>` float * simtk.unit

        :param time_interval: frequency, in number of time steps, at which positions were recorded (default=200)
        :type time_interval: int

        :param output_dir: path to which we will write the output (default='output')
        :type output_dir: str

        :param output_data: name of output .nc data file (default='output.nc')
        :type output_data: str    

        :param checkpoint_data: name of checkpoint .nc data file (default='output_checkpoint.nc')
        :type checkpoint_data: str   

        :param frame_begin: Frame at which to start writing the dcd trajectory (default=0)
        :type frame_begin: int

        :param frame_stride: advance by this many time intervals when writing dcd trajectories (default=1)
        :type frame_stride: int

        :param center: align all frames in the replica trajectories (default=False)
        :type center: Boolean
        """

        file_list = []

        output_data_path = os.path.join(output_dir, output_data)

        # Get number of replicas:
        reporter = MultiStateReporter(output_data_path, open_mode='r', checkpoint_storage=checkpoint_data)
        states = reporter.read_thermodynamic_states()[0]
        n_replicas=len(states)

        sampler_states = reporter.read_sampler_states(iteration=0)
        xunit = sampler_states[0].positions[0].unit

        for replica_index in range(n_replicas):
            replica_positions = extract_trajectory(topology, reporter, replica_index=replica_index,
                frame_begin=frame_begin, frame_stride=frame_stride)

            n_frames_tot = replica_positions.shape[0]

            # Determine simulation time (in ps) for each frame:
            time_delta_ps = (timestep*time_interval).value_in_unit(unit.picosecond)
            traj_times = np.linspace(
                frame_begin*time_delta_ps,
                (frame_begin+frame_stride*(n_frames_tot-1))*time_delta_ps,
                num=n_frames_tot,
            )

            file_name = f"{output_dir}/replica_{replica_index+1}.dcd"

            # Trajectories are written in nanometers:
            replica_traj = Trajectory(
                replica_positions,
                Topology.from_openmm(self.topology),
                time=traj_times,
            )

            if center:
                ref_traj = replica_traj[0]
                replica_traj.superpose(ref_traj)
                # This rewrites to replica_traj        

            Trajectory.save_dcd(replica_traj,file_name)

        reporter.close()

        return file_list
        
    def restart(self):

        """
        Restart an OpenMMTools replica exchange simulation using an OpenMM model and
        output .nc files from the previous segment of the simulation. 

        output_data: Path to the NETCDF file for previous segment of simulation - this will be appended to (default="output/output.nc")
        :type output_data: str
        """

        reporter = MultiStateReporter(self.out_filepath, open_mode="r+")
        simulation = ReplicaExchangeSampler.from_storage(reporter)

        n_iter_remain = self.exchange_attempts - simulation.iteration

        simulation.extend(n_iterations=n_iter_remain)
        self.simulation = simulation
                
    def get_minimum_energy_ensemble(self):

        """
        Get an ensemble of low (potential) energy poses, and write the lowest energy structure to a PDB file if a file_name is provided.
        
        replica_energies: List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps
        :type replica_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )
        
        replica_positions: List of positions for all output frames for all replicas
        :type replica_positions: np.array( ( float * simtk.unit.positions for num_beads ) for simulation_steps )
        
        file_name: Output destination for PDB coordinates of minimum energy pose, Default = None
        
        :returns:
        - ensemble ( List() ) - A list of poses that are in the minimum energy ensemble.

        :Example:
        
        >>> from foldamers.cg_model.cgmodel import CGModel
        >>> from cg_openmm.simulation.rep_exch import *
        >>> cgmodel = CGModel()
        >>> replica_energies,replica_positions,replica_state_indices = run_replica_exchange(cgmodel.topology,cgmodel.system,cgmodel.positions)
        >>> ensemble_size = 5
        >>> file_name = "minimum.pdb"
        >>> minimum_energy_ensemble = get_minimum_energy_ensemble(cgmodel.topology,replica_energies,replica_positions,ensemble_size=ensemble_size,file_name=file_name)
        
        """
        # Get the minimum energy structure sampled during the simulation
        rd = self.reporter_to_dict()
        r_energies, r_positions = rd["energies"], rd["trajectory"]
        energies = np.zeros()


        ensemble = []
        ensemble_energies = []
        for replica in range(len(r_energies)):
            energies = np.array([energy for energy in r_energies[replica][replica]])
            for energy in range(len(energies)):
                if len(ensemble) < ensemble_size:
                    ensemble.append(r_positions[replica][energy])
                    ensemble_energies.append(energies[energy])
                else:
                    for comparison in range(len(ensemble_energies)):
                        if energies[energy] < ensemble_energies[comparison]:
                            ensemble_energies[comparison] = energies[energy]
                            ensemble[comparison] = r_positions[replica][energy]

        return ensemble


    def plot_energies(self,
        state_energies,
        temperatures,
        series_per_page,
        time_interval=1.0 * unit.picosecond,
        time_shift=0.0 * unit.picosecond,    
        file_name="rep_ex_ener.pdf",
        legend=True,
    ):
        """
        Plot the potential energies for a batch of replica exchange trajectories

        state_energies: List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps
        :type state_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

        temperatures: List of temperatures for which to perform replica exchange simulations, default = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)]
        :type temperature: List( float * simtk.unit.temperature )

        time_interval: interval between energy exchanges.
        :type time_interval: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

        time_shift: amount of time before production period to shift the time axis(default = 0)
        :type time_shift: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_
        
        file_name: The pathname of the output file for plotting results, default = "replica_exchange_energies.png"
        :type file_name: str

        legend: Controls whether a legend is added to the plot
        :type legend: Logical

        """

        simulation_times = np.array(
            [
                step * time_interval.value_in_unit(unit.picosecond)
                for step in range(len(state_energies[0]))
            ]
        )
        
        simulation_times += time_shift.value_in_unit(unit.picosecond)
        
        # To improve pdf render speed, sparsify data to display less than 2000 data points
        n_xdata = len(simulation_times)
        
        if n_xdata <= 1000:
            plot_stride = 1
        else:
            plot_stride = int(np.floor(n_xdata/1000))
        
        # If more than series_per_page replicas, split into separate pages for better visibility
        nmax = series_per_page
        npage = int(np.ceil(len(temperatures)/nmax))
        
        with PdfPages(file_name) as pdf:
            page_num=1
            plotted_per_page=0
            pyplot.figure()
            for state in range(len(temperatures)):
                if plotted_per_page <= (nmax):
                    pyplot.plot(
                        simulation_times[::plot_stride],
                        state_energies[state,::plot_stride],
                        alpha=0.5,
                        linewidth=1,
                    )
                    plotted_per_page += 1
                    
                if (plotted_per_page >= nmax) or (state==(len(temperatures)-1)):
                    # Save and close previous page
                    pyplot.xlabel("Simulation Time ( Picoseconds )")
                    pyplot.ylabel("Potential Energy ( kJ / mol )")
                    pyplot.title("Replica Exchange Simulation")
                    
                    if legend:
                        pyplot.legend(
                            [round(temperature.value_in_unit(unit.kelvin), 1) for temperature in temperatures[(0+(page_num-1)*nmax):(page_num*nmax)]],
                            loc="center left",
                            bbox_to_anchor=(1, 0.5),
                            title="T (K)",
                        )  
                    
                    pdf.savefig(bbox_inches="tight") # Save current fig to pdf page
                    pyplot.close()
                    plotted_per_page = 0
                    page_num += 1
                    
        return
        

    def plot_energy_histograms(
        self, 
        state_energies,
        temperatures,
        file_name="rep_ex_ener_hist.pdf",
        legend=True,
    ):
        """
        Plot the potential energies for a batch of replica exchange trajectories

        state_energies: List of dimension num_replicas X simulation_steps, which gives the energies for all replicas at all simulation steps
        :type state_energies: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

        temperatures: List of temperatures for which to perform replica exchange simulations, default = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)]
        :type temperature: List( float * simtk.unit.temperature )

        file_name: The pathname of the output file for plotting results, default = "replica_exchange_energies.png"
        :type file_name: str

        legend: Controls whether a legend is added to the plot
        :type legend: Logical

        """

        figure = pyplot.figure(figsize=(8.5,11))

        for state in range(len(temperatures)):
            n_out, bin_edges_out = np.histogram(
                state_energies[state,:],bins=20,density=True,
            )
            
            bin_centers = np.zeros((len(bin_edges_out)-1,1))
            for i in range(len(bin_edges_out)-1):
                bin_centers[i] = (bin_edges_out[i]+bin_edges_out[i+1])/2
            
            pyplot.plot(bin_centers,n_out,'o-',alpha=0.5,linewidth=1,markersize=6)
                

        pyplot.xlabel("Potential Energy ( kJ / mol )")
        pyplot.ylabel("Probability")
        pyplot.title("Replica Exchange Energy Histogram")
        
        if legend:
            pyplot.legend(
                [round(temperature._value, 1) for temperature in temperatures],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title="T (K)",
            )

        pyplot.savefig(file_name, bbox_inches="tight")
        pyplot.close()

        return
        
        
    def plot_matrix(
        self,
        replica_state_indices,
        file_name='state_probability_matrix.pdf'
        ):
        
        # Plot a matrix of replica vs. state, coloring each box in the grid by normalized frequency 
        # For each replica, histogram the state indices data 
        # Then normalize the data and create [n_replica x n_state] patch graph
        
        n_replicas = replica_state_indices.shape[0]
        
        hist_all = np.zeros((n_replicas, n_replicas))
        
        state_bin_edges = np.linspace(-0.5,n_replicas-0.5,n_replicas+1)
        state_bin_centers = 0.5+state_bin_edges[0:n_replicas]
        
        for rep in range(n_replicas):
            hist_all[rep,:], bin_edges = np.histogram(
                replica_state_indices[rep,:],bins=state_bin_edges,density=True,
            )
            
        # No need for global normalization, since each replica's state probabilities must sum to 1
        
        hist_norm = np.zeros_like(hist_all)
        for rep in range(n_replicas):
            for state in range(n_replicas):
                hist_norm[rep,state] = hist_all[rep,state]/np.max(hist_all[rep,:])    
        
        mean_score = np.mean(hist_norm)
        min_score = np.amin(hist_norm)
        
        ax = pyplot.subplot(111)
        
        cmap=pyplot.get_cmap('nipy_spectral') 
        norm=Normalize(vmin=0,vmax=1) 
        
        ax.imshow(hist_norm,cmap=cmap,norm=norm)
        ax.set_aspect('equal', 'box')
        
        # Append colorbar axis to right side
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%",pad=0.20)  
        
        pyplot.colorbar(
            cm.ScalarMappable(cmap=cmap,norm=norm),
            cax=cax,
            label='normalized frequency',
            )
        
        ax.set_xlabel("State")
        ax.set_ylabel("Replica")
        pyplot.suptitle(f"Replica exchange state probabilities\n(Mean: {mean_score:.4f} Min: {min_score:.4f})")  
        
        pyplot.savefig(file_name)
        pyplot.close()    
        
        return hist_all
        
        
    def plot_summary(self,
        replica_states,
        temperatures,
        series_per_page,
        time_interval=1.0 * unit.picosecond,
        time_shift=0.0 * unit.picosecond,
        file_name="rep_ex_states.pdf",
        legend=True,
    ):
        """
        Plot the thermodynamic state assignments for individual temperature replicas as a function of the simulation time, in order to obtain a visual summary of the replica exchanges from a OpenMM simulation.

        replica_states: List of dimension num_replicas X simulation_steps, which gives the thermodynamic state indices for all replicas at all simulation steps
        :type replica_states: List( List( float * simtk.unit.energy for simulation_steps ) for num_replicas )

        temperatures: List of temperatures for which to perform replica exchange simulations, default = [(300.0 * unit.kelvin).__add__(i * unit.kelvin) for i in range(-20,100,10)]
        :type temperature: List( float * simtk.unit.temperature )

        time_interval: interval between energy exchanges.
        :type time_interval: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

        time_shift: amount of time before production period to shift the time axis(default = 0)
        :type time_shift: `SIMTK <https://simtk.org/>`_ `Unit() <http://docs.openmm.org/7.1.0/api-python/generated/simtk.unit.unit.Unit.html>`_

        file_name: The pathname of the output file for plotting results, default = "replica_exchange_state_transitions.png"
        :type file_name: str

        legend: Controls whether a legend is added to the plot
        :type legend: Logical

        """
        
        simulation_times = np.array(
            [
                step * time_interval.value_in_unit(unit.picosecond)
                for step in range(len(replica_states[0]))
            ]
        )
        
        simulation_times += time_shift.value_in_unit(unit.picosecond)
        
        # To improve pdf render speed, sparsify data to display less than 2000 data points
        n_xdata = len(simulation_times)
        
        if n_xdata <= 1000:
            plot_stride = 1
        else:
            plot_stride = int(np.floor(n_xdata/1000))
        
        # If more than series_per_page replicas, split into separate pages for better visibility
        nmax = series_per_page
        npage = int(np.ceil(len(temperatures)/nmax))
            
        with PdfPages(file_name) as pdf:
            page_num=1
            plotted_per_page=0
            pyplot.figure()
            for replica in range(len(replica_states)):
                state_indices = np.array([int(round(state)) for state in replica_states[replica]])
                
                if plotted_per_page <= (nmax):
                    
                    pyplot.plot(
                        simulation_times[::plot_stride],
                        state_indices[::plot_stride],
                        alpha=0.5,
                        linewidth=1
                    )
                    plotted_per_page += 1
                    
                if (plotted_per_page >= nmax) or (replica==(len(replica_states)-1)):
                    # Save and close previous page
                    pyplot.xlabel("Simulation Time ( Picoseconds )")
                    pyplot.ylabel("Thermodynamic State Index")
                    pyplot.title("State Exchange Summary")
                    
                    if legend:
                        pyplot.legend(
                            [i for i in range((page_num-1)*nmax,page_num*nmax)],
                            loc="center left",
                            bbox_to_anchor=(1, 0.5),
                            title="Replica Index",
                        )
                    
                    pdf.savefig(bbox_inches="tight") # Save current fig to pdf page
                    pyplot.close()
                    plotted_per_page = 0
                    page_num += 1

        return

if __name__ == "__main__":
    import datamol as dm
    from openff.toolkit.typing.engines.smirnoff import ForceField
    from openff.toolkit.topology import Molecule, Topology
    from opendata.conformers.conformers import mol_to_openmm_topology_and_system
    import openmm 
    #from opendata import utils

    # setup the platform 
    from openmmtools.cache import global_context_cache
    global_context_cache.platform = openmm.Platform.getPlatformByName("CPU")

    smi = "CC(C(=O)NC(C)C(=O)O)N"
    ff_engine = ForceField("openff_unconstrained-2.0.0.offxml")

    # Parameterize the system
    mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    ret = mol_to_openmm_topology_and_system(mol, ff_engine)

    topology, system = ret
    # Generating {n_starting_points} starting points
    mol.generate_conformers(n_conformers=1, rms_cutoff=None)

    positions = mol.conformers[0].to_openmm()

    #cache = utils.get_local_cache()
    opath = "output"
    if not os.path.exists(opath):
        os.mkdir(opath)

    output_data = os.path.join(opath, "output.nc")
    print(positions)

    re = ReplicaExchange(topology=topology, system=system, positions=positions, overwrite=False,
                        out_filepath=output_data, temperatures = [300.0  * unit.kelvin, 500.0  * unit.kelvin])
    re()
    re.process_replica_exchange_data()




