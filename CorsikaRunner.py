import os
import random
from astropy import units as u
import uuid

# ------------------------
# Shared particle map
# ------------------------

particle_map = {
    # ------------------------
    # Basics
    # ------------------------
    # Table 4, Page 123 CORSIKA Guide
    "gamma": 1,
    "e+": 2, "positron": 2,
    "e-": 3, "electron": 3,
    "mu+": 5, "antimuon": 5,
    "mu-": 6, "muon": 6,
    "pi0": 7,
    "pi+": 8,
    "pi-": 9,
    "K0_L": 10,
    "K+": 11,
    "K-": 12,
    "n": 13, "neutron": 13,
    "p": 14, "proton": 14,
    "p_bar": 15, "antiproton": 15,
    "K0_S": 16,
    "eta": 17,
    "Lambda": 18,
    "Sigma+": 19,
    "Sigma0": 20,
    "Sigma-": 21,
    "Xi0": 22,
    "Xi-": 23,
    "Omega-": 24,
    "n_bar": 25,
    "Lambda_bar": 26,
    "Sigma_bar-": 27,
    "Sigma_bar0": 28,
    "Sigma_bar+": 29,
    "Xi_bar0": 30,
    "Xi_bar+": 31,
    "Omega_bar+": 32,
    "eta_prime": 48,
    "phi": 49,
    "omega": 50,
    "rho0": 51,
    "rho+": 52,
    "rho-": 53,
    "Delta++": 54,
    "Delta+": 55,
    "Delta0": 56,
    "Delta-": 57,
    "Delta_bar--": 58,
    "Delta_bar-": 59,
    "Delta_bar0": 60,
    "Delta_bar+": 61,
    "K*0": 62,
    "K*+": 63,
    "K*-": 64,
    "K*0_bar": 65,
    "nu_e": 66,
    "nu_e_bar": 67,
    "nu_mu": 68,
    "nu_mu_bar": 69,
    
    # ------------------------
    # Charmed and Heavy Mesons/Baryons 
    # ------------------------
    # Table 4, Page 124 CORSIKA Guide
    "D0": 116, "D+": 117, "D-": 118, "D0_bar": 119,
    "J/psi": 130,
    "tau+": 131, "tau-": 132,
    "nu_tau": 133, "nu_tau_bar": 134,
    "Lambda_c+": 137,
      
    # ------------------------
    #        Nuclei
    # ------------------------
    # Note: ID = A * 100 + Z
    "helium": 402,   
    "lithium": 703,  
    "beryllium": 904,  
    "boron": 1105, 
    "carbon": 1206,  
    "nitrogen": 1407,
    "oxygen": 1608, 
    "flourine": 1909, 
    "neon": 2010, 
    "sodium": 2311, 
    "magnesium": 2412, 
    "aluminium": 2713, 
    "silicon": 2814, 
    "phosphorus": 3115, 
    "sulfur": 3216, 
    "chlorine": 3517, 
    "argon": 4018, 
    "potassium": 3919,
    "calcium": 4020,
    "scandium": 4521,
    "titanium": 4822,
    "vanadium": 5123,
    "chromium": 5224, 
    "manganese": 5525,
    "iron": 5626
}

class CorsikaRunner:
    """
    A class to configure and run CORSIKA simulations.

    Attributes:
        path_corsika_executable (str): Path to the CORSIKA executable.
        path_input_card_template (str): Path to the input card template.
        path_inputcard (str): Path where the filled input card will be stored.
        current_config (dict): Stores current simulation parameters.
        particle_map (dict): Maps particle names to CORSIKA particle IDs.
    """

    def __init__(
        self,
        path_corsika_executable,
        path_input_card_template="input_template.inp",
    ):
        """
        Initializes the CorsikaRunner with paths to executables and input files.

        Args:
            path_corsika_executable (str): Path to the CORSIKA executable.
            path_input_card_template (str, optional): Path to the input card template.
                Defaults to "input_template.inp".
            path_data_output (str, optional): Directory for simulation output data.
                Defaults to the current working directory.
        """
        self.path_corsika_executable = path_corsika_executable
        self.path_input_card_template = path_input_card_template
        self.temp_output_dir = None

        # The filled-out input card is stored in the local directory
        self.path_inputcard = os.path.join(os.getcwd(), "input_particletracks.inp")

        # Default configuration
        self.current_config = {
            "run_number": 1,
            "primary_particle": None,
            "primary_energy": None,
            "observation_level": None,
            "zenith_angle": None,
            "azimuth_angle": None,
            "seeds": None,
            "path_output": None,
            "ext_atmosphere": None
        }

        # Mapping between CORSIKA particle ID and particle name
        self.particle_map = particle_map

    def configure_run(
        self,
        primary_particle,
        primary_energy,
        observation_level=0 * u.m,
        run_number=1,
        zenith_angle=0 * u.deg,
        azimuth_angle = 0 * u.deg,
        random_seeds=True,
        path_output=None,
        ext_atmosphere=None
    ):
        """
        Configures the simulation parameters for a CORSIKA run.

        Args:
            primary_particle (str): Name of the primary particle (e.g., "gamma", "proton").
                Refer to `CorsikaRunner.particle_map` for valid options.
            primary_energy (astropy.units.Quantity): Energy of the primary particle.
            observation_level (astropy.units.Quantity, optional): Altitude of the observation level
                above sea level. Defaults to `0 * u.m`.
            run_number (int, optional): Identifier for the simulation run. Defaults to `1`.
            zenith_angle (astropy.units.Quantity, optional): Zenith angle of the incident primary particle.
                Defaults to `0 * u.deg`.
            random_seeds (bool, optional): If `True`, the showers are generated with random seeds.
                Defaults to `True`.
            path_output (str, optional): Path under which the produced simulation files are stored. Defaults to the
                current script directory.
            ext_atmosphere (str, optional): External atmospheric profile identifier (i.e atmprof10.dat -> ext_atmosphere='10').
                Defaults to internal atmosphere (not location specific).

        Returns:
            int: Always returns `0` to indicate successful configuration.
        """
        self.current_config["run_number"] = int(run_number)
        self.current_config["primary_particle"] = self._parse_particletype(primary_particle)
        self.current_config["primary_energy"] = primary_energy.to(u.GeV)
        self.current_config["zenith_angle"] = zenith_angle.to(u.deg)
        self.current_config["azimuth_angle"] = azimuth_angle.to(u.deg)
        self.current_config["observation_level"] = observation_level.to(u.cm)
        self.current_config["path_output"] = path_output or os.getcwd()
        
        if ext_atmosphere:
            self.current_config["ext_atmosphere"] = f'ATMOSPHERE  {ext_atmosphere} T'
        else: 
            self.current_config["ext_atmosphere"] = '*'

        # Generate random seed if enabled
        self.current_config["seeds"] = self._generate_seeds() if random_seeds else "*"

        self._generate_inputcard()

        return 0

    def _generate_inputcard(self):
        """
        Populates the input card template with the user-provided configuration,
        saves it to disk, and returns the output as a string.

        Returns:
            str: The formatted input card content.
        """
        with open(self.path_input_card_template, "r") as f:
            template_content = f.read()

        # Generates a unique random folder name to store files in the 
        # CORSIKA directory
        # Note: we cannot directly save to the target directory as 
        # CORSIKA has a limit to the maximum number of characters in a path
        # We first save things in a random directory and copy it back to the 
        # user defined path after the simulation as finished
        self.temp_output_dir = str(uuid.uuid4()) 
        
        
        # Replace placeholders
        template_content = template_content.format(
            run_number=self.current_config["run_number"],
            seeds=self.current_config["seeds"],
            primary_particle=self.current_config["primary_particle"],
            primary_energy=self.current_config["primary_energy"].value,
            zenith_angle=self.current_config["zenith_angle"].value,
            azimuth_angle=self.current_config["azimuth_angle"].value,
            observation_level=self.current_config["observation_level"].value,
            output_directory=os.path.join(f'./{self.temp_output_dir}/', "sim_"),
            ext_atmosphere=self.current_config["ext_atmosphere"],
        )

        # Write to the output file
        with open(self.path_inputcard, "w") as f:
            f.write(template_content)

        return template_content

    def _parse_particletype(self, particle_name):
        """
        Converts a particle name to the corresponding CORSIKA particle ID.

        Args:
            particle_name (str): Particle name (e.g., "gamma", "proton").

        Returns:
            int: Corresponding CORSIKA particle ID.

        Raises:
            ValueError: If the particle name is not recognized.
        """
        particle_name = particle_name.lower().strip()
        particle_id = self.particle_map.get(particle_name)

        if particle_id is None:
            raise ValueError(f"Unknown primary particle name: '{particle_name}'")

        return particle_id

    def _generate_seeds(self):
        """
        Generates a set of random seeds for the different RNGs in CORSIKA.

        Returns:
            str: SEED definitions for the CORSIKA input card.
        """
        seed_min, seed_max = 10**7, 10**9
        second_number_min, second_number_max = 100, 1000
        seeds = "\n".join(
            f"SEED    {random.randint(seed_min, seed_max)}    "
            f"{random.randint(second_number_min, second_number_max)}    0     "
            f"seed for random number sequence {i + 1}"
            for i in range(4)
        )
        return seeds

    def run_simulation(self):
        """
        Calls the CORSIKA binary to run the simulation with user-defined parameters.
        """

        # Store the current working directory
        _current_path = os.getcwd()

        # Change to the CORSIKA binary directory
        os.chdir(os.path.dirname(self.path_corsika_executable))

        # Make sure the temporary directory exists
        os.makedirs(self.temp_output_dir, exist_ok=True)  # Create the folder
        
        # Run the simulation
        print("Starting CORSIKA simulation (this may take a few minutes)...")
        os.system(
            f"{self.path_corsika_executable} < {self.path_inputcard} > "
            f"{os.path.join(self.temp_output_dir, "corsika_output.log")}"
        )
        
        print('Simulation has completed')
        print('\t-> Copying files to user directory')
        

        # Restore the original working directory
        os.chdir(_current_path)
        
        # Delete the output dir if it exists
        os.system(
            f"rm -rf {self.current_config['path_output']}"
        )
        
        # Make sure the user output directory exists
        os.makedirs(self.current_config["path_output"], exist_ok=True)  # Create the folder
        
        # Copy over files from temp directory to user-specified one 
        _path_tmp_dir = os.path.join(os.path.dirname(self.path_corsika_executable), self.temp_output_dir)
        os.system(
            f"cp {_path_tmp_dir}/* {self.current_config['path_output']}"
        )
        
        print('\t-> Cleanup temporary working directory')
        os.system(
            f"rm -rf {_path_tmp_dir}"
        )
        