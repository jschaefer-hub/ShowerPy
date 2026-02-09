import os
import struct
import random
import numpy as np
import pandas as pd
import eventio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from CorsikaRunner import particle_map

class CorsikaPlotter:
    """
    A class to load, parse, and visualize CORSIKA simulation data.

    Attributes:
        path_data (str): Path to the directory containing CORSIKA output files.
        cherenkov_photons (pd.DataFrame or None): Parsed Cherenkov photon data.
        particle_tracks (pd.DataFrame or None): Parsed particle track data.
        file_paths (dict): Dictionary storing available file paths for different data types.
    """

    def __init__(self, path_data):
        """
        Initializes the CorsikaPlotter and loads available data.

        Args:
            path_data (str): Path to the directory containing CORSIKA simulation output.
        """

        # ------------------------
        # Paths and Dataframes
        # ------------------------
        self.path_data = path_data
        self.cherenkov_photons = None   # stores the dataframe
        self.particle_tracks = None     # stores the dataframe
        
        # ------------------------
        # Simulation parameters
        # ------------------------
        # Note: Lots of them will be populated from the eventio file header
        self.zenith_deg = None
        self.azimuth_deg = None
        # TODO: Corsika stores everything in cm ... maybe I should change the internal logic to m at some point?
        self.first_interaction_height_cm = None
        self.impact_point_x = 0.0  # Shower core X offset from telescope [cm]
        self.impact_point_y = 0.0  # Shower core Y offset from telescope [cm]
        self.primary_energy = None # Simulated primary energy of the primary particle [GeV]
        self.primary_particle_id = None # Primary particle ID used in the simulation
        self.observation_level = None   # Simulated observation level [cm] a.s.l.
        
        # Mapping between CORSIKA particle ID and particle name
        self.particle_map = particle_map

        # Dictionary to store full paths of available files
        self.file_paths = {
            "em_data": None,
            "muon_data": None,
            "hadron_data": None,
            "cherenkov_data": None,
        }

        # Check if we have all files and load their paths
        self._check_available_files()

        # Load data into pandas DataFrames
        self.cherenkov_photons = self._parse_cherenkov_data()
        self.particle_tracks = self._parse_particle_data()
        
        # Transform Cherenkov photon coordinates to particle track coordinate system
        # Note: Cherenkov photons are stored in a seperate coordinate system around he impact point
        # This adds the core offset so photons appear at the shower core location
        self._transform_cherenkov_to_particle_coords()

    def get_particle_name_by_id(self, particle_id):
        """Returns the name of the particle given its CORSIKA ID."""
        for name, p_id in self.particle_map.items():
            if p_id == particle_id:
                return name
        return f"unknown({particle_id})"

    def _get_particle_ids_from_selection(self, selection_key):
        """
        Parses a selection key (e.g., 'muon + antimuon', 'lepton', 'hadron', 'nuclei')
        and returns a list of corresponding CORSIKA particle IDs.
        """
        # Split by '+' to handle combinations
        subnames = [name.strip() for name in selection_key.split('+')]
        particle_ids = set()
        
        # ------------------------
        # Lepton mapping
        # ------------------------
        # Define lepton IDs explicitly
        lepton_ids = {
            2, 3,       # e+, e-
            5, 6,       # mu+, mu-
            66, 67,     # nu_e, anti_nu_e
            68, 69,     # nu_mu, anti_nu_mu
            131, 132,   # tau+, tau-
            133, 134    # nu_tau, anti_nu_tau
        }
        
        
        # ------------------------
        # Hadron mapping
        # ------------------------
        # Easier to define what we dont want here
        # This is just the lepton_ids with gammas (1) excluded
        non_hadron_ids = lepton_ids | {1}

        for name in subnames:
            if name == "lepton":
                particle_ids.update(lepton_ids)
                
            elif name == "nuclei":
                # CORSIKA nuclei follow A*100 + Z (e.g., He4 is 402)
                # Elementary particles stop around ID 195 (Omega_b)
                for p_name, p_id in self.particle_map.items():
                    if p_id >= 200:
                        particle_ids.add(p_id)

            elif name == "hadron":
                # All particles in the map EXCEPT Gamma and Leptons
                for p_name, p_id in self.particle_map.items():
                    if p_id not in non_hadron_ids:
                        particle_ids.add(p_id)
            
            elif name in self.particle_map:
                particle_ids.add(self.particle_map[name])
            
            else:
                print(f"Warning: Unknown particle type '{name}', skipping.")
        
        # Return intersection with map to ensure IDs exist in current configuration
        valid_ids = set(self.particle_map.values())
        return list(particle_ids.intersection(valid_ids))

    def _check_available_files(self):
        """
        Checks which types of simulation output files are available in the given directory.

        Raises:
            ValueError: If no CORSIKA files are found.
        """
        try:
            files = os.listdir(self.path_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Directory '{self.path_data}' not found.")

        file_patterns = {
            "track_em": "em_data",
            "track_mu": "muon_data",
            "track_hd": "hadron_data",
            "cherenkov_iact": "cherenkov_data",
        }

        # ------------------------
        #  Check for each file
        # ------------------------
        for file in files:
            full_path = os.path.join(self.path_data, file)
            if os.path.isfile(full_path):
                for key, attr in file_patterns.items():
                    if file.endswith(key):
                        self.file_paths[attr] = full_path

        # ------------------------
        #  User output
        # ------------------------
        print("Looking for available files:")
        # Get longest filetype name so everything is printed nicely!
        max_key_length = max(map(len, self.file_paths.keys()))
        
        # Print some output for the user about found files
        for key, value in self.file_paths.items():
            status = f"Found {os.path.basename(value)}" if value else "Not found"
            print(f"\t -> {key.ljust(max_key_length)} : {status}")

        # Raise error if no files are found in provided directory
        if all(value is None for value in self.file_paths.values()):
            raise ValueError("No CORSIKA files found!")

    def _parse_cherenkov_data(self):
        """
        Parses Cherenkov photon data and extras shower geometry from header.

        Returns:
            pd.DataFrame: DataFrame containing Cherenkov photon information.
        """

        print("\nParsing Cherenkov photon data")

        # Open up the first event
        # Note: we do not expect more than one event here
        f = eventio.IACTFile(self.file_paths["cherenkov_data"])
        event = next(iter(f))

        # ------------------------
        #  Parse event header
        # ------------------------

        # Extract shower geometry from event header
        # Note:  event header is a structured numpy array with named fields
        # can be listed with print(event.header.dtype.names)
        # for i in event.header.dtype.names:
        #     print(i)
        # return 

        event_header = event.header

        # Get input parameters
        self.zenith_deg = np.degrees(float(event_header['zenith']))
        self.azimuth_deg = np.degrees(float(event_header['azimuth']))
        self.primary_particle_id = int(event_header['particle_id'])
        self.primary_energy = float(event.header['total_energy'])
        self.observation_level = float(event.header['observation_height'][0])

        # Take absolute value to get the height above ground
        # TODO: For some reason this is negative -> find out why?
        self.first_interaction_height_cm = abs(float(event_header['first_interaction_height']))
        
        # ------------------------
        #  User output
        # ------------------------

        # Primary particle and energy
        print(
            f"\t-> {self.get_particle_name_by_id(self.primary_particle_id).title()} "
            f"with {self.primary_energy} GeV energy"
        )
        # Arival direction
        print(
            f"\t-> Arriving from from zenith={self.zenith_deg:.1f}°,"
            f"azimuth={self.azimuth_deg:.1f}°"
        )
        # First interaction height 
        print(
            f"\t-> First interaction height: {self.first_interaction_height_cm * 1e-5:.1f} km"
        )
        # Observation level
        print(
            f"\t-> Observation level: {self.observation_level * 1e-5:.1f} km"
        )

        # ------------------------
        #  Create Dataframe
        # ------------------------
        # Extract telescope position and photon bunches
        # Note: telescope position not interesting if we only have a single one
        # telescope_position = pd.DataFrame(f.telescope_positions)
        cherenkov_photons = pd.DataFrame(event.photon_bunches[0])
        cherenkov_photons.columns = [
            "x_impact_cm",
            "y_impact_cm",
            "cos_incident_x",
            "cos_incident_y",
            "time_since_first_interaction_ns",
            "emission_height_asl_cm",
            "photons", # Note: This is the expected number of photons in this specific bunch.
            "wavelength_nm",
        ]

        # Note: 'photons' is often a weight/count in the bunch, don't drop if you need density accuracy!
        return cherenkov_photons

    def _parse_particle_data(self):
        """
        Parses particle track data from simulation output files.

        Returns:
            pd.DataFrame: A DataFrame containing particle track information.
        """
        print("\nParsing particle track data")

        # This is the mapping of the data in the fortran files for each data record
        columns = [
            "particle_id",
            "energy_gev",
            "x_start",
            "y_start",
            "z_start",
            "t_start",
            "x_end",
            "y_end",
            "z_end",
            "t_end",
        ]

        # ------------------------
        #  Parse binary files
        # ------------------------

        # Open up the dataframe which later will contain all data
        particle_tracks_df = pd.DataFrame(columns=columns)

        for particle_file in list(self.file_paths.values())[:-1]:
            if particle_file is None:
                continue

            print(f"\t-> Reading {os.path.basename(particle_file)}")

            # Iterate over the Fortran file and parse data in accordance
            # This is based on the official CORSIKA/EVENTIO Documentation
            tracks = []
            with open(particle_file, "rb") as f:
                while True:

                    # Read first four bites and parse as integer to know how much
                    # Data we are gonna expect
                    marker1_bytes = f.read(4)
                    if len(marker1_bytes) < 4:
                        break
                    marker1 = struct.unpack("i", marker1_bytes)[0]
                    
                    # Now we read the datachunk of the determined size
                    data_bytes = f.read(marker1)
                    if len(data_bytes) < marker1:
                        break

                    # Check if we have reached the Endmarker for the data record
                    # Should the the same as marker 1
                    marker2_bytes = f.read(4)
                    if len(marker2_bytes) < 4:
                        break
                    marker2 = struct.unpack("i", marker2_bytes)[0]

                    if marker1 != marker2:
                        raise ValueError(f"Fortran record markers do not match: {marker1} vs {marker2}")

                    # Parse out the data as 10 32-bit integers (order see above)
                    tracks.append(struct.unpack("10f", data_bytes))

            # Skip all other data
            if not tracks:
                continue

            # ------------------------
            #  Create Dataframe
            # ------------------------

            # Form a pandas dataframe and discard nan entries
            temp_df = pd.DataFrame(tracks, columns=columns).dropna(axis=1, how="all")

            if particle_tracks_df.empty:
                particle_tracks_df = temp_df
            else:
                particle_tracks_df = pd.concat([particle_tracks_df, temp_df], ignore_index=True)

        return particle_tracks_df

    def _transform_cherenkov_to_particle_coords(self):
        """
        Transforms Cherenkov photon coordinates to the particle track coordinate system.
        
        Cherenkov photon impact positions are originally centered around the impact point.
        This method adds the extrapolated impact point coordinates to the cherenkov data
        to place them at the shower core location in the particle track coordinate system.
        """
        print('\nCorrecting Cherenkov coordinate system')

        # ------------------------
        # Check for requirements
        # ------------------------
        
        # We must have the cherenkov and particle track data
        if self.cherenkov_photons is None or self.particle_tracks.empty:
            print("\t-> Error: Cherenkov and particle track data not yet parsed")
            return
            
        # We also need to know the shower zenith and azimuth
        if self.zenith_deg is None or self.azimuth_deg is None:
             print("\t-> Error: Shower geometry not available, skipping coordinate transform.")
             return

        # ------------------------
        #  Determine impact point
        # ------------------------

        # Note:
        # Use the primary particle's trajectory to find the exact impact point on ground
        # We use the first entry of the primary particle type to determine its travel direction
        # We then use two points to define the shower axis line and intersect with observation level
        
        # Find primary particle in dataframe
        # Note: primary will always start at time 0
        primary = self.particle_tracks[
            (self.particle_tracks["t_start"] == 0)
            ]

        if not (len(primary) == 1):
            print(f'ERROR: Could not uniquely identify primary particle, found {len(primary)} matching criteria')
            return

        x1, y1, z1 = primary[["x_start", "y_start", "z_start"]].iloc[0]
        # print(x1, y1, z1)
        x2, y2, z2 = primary[["x_end", "y_end", "z_end"]].iloc[0]
        # Using :N_TOTAL_DIGITS.DECIMALf
        print(f"\t-> Primary trajectory: Start=({x1*1e-5:6.1f}, {y1*1e-5:6.1f}, {z1*1e-5:6.1f}) km")
        print(f"\t                       End  =({x2*1e-5:6.1f}, {y2*1e-5:6.1f}, {z2*1e-5:6.1f}) km")
        
        # Direction vector
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        
        # Intersect with observation level 
        z_obs = self.observation_level  # Observation level at sea level
        
        # Parametric line: (x, y, z) = (x1, y1, z1) + t * (dx, dy, dz)
        # At observation level: z_obs = z1 + t * dz
        # t = (z_obs - z1) / dz
        t = (z_obs - z1) / dz
        
        # Calculate impact point
        self.impact_point_x = x1 + t * dx
        self.impact_point_y = y1 + t * dy
        
        print(f"\t-> Shower axis impact point: x={self.impact_point_x*1e-5:.2f} km, y={self.impact_point_y*1e-5:.2f} km")

        print(f"\t-> Shifting Cherenkov photon coordinates")
        # Apply the shift to the cherenkov data
        self.cherenkov_photons["x_impact_cm"] += self.impact_point_x
        self.cherenkov_photons["y_impact_cm"] += self.impact_point_y

    def _cartesian_to_polar(self, x, y):
        """Convert Cartesian coordinates (x, y) to polar coordinates (r, theta)."""
        r = np.sqrt(x**2 + y**2) 
        theta = np.arctan2(y, x) 
        return r, theta
    
    def _ring_area(self, r_inner, r_outer):
        """Calculates the area of a ring using inner and outer diameter

        Args:
            r_inner (float): Inner diameter of ring
            r_outer (float): Outer diameter of ring

        Returns:
            float: Area
        """        
        return np.pi*(r_outer**2 - r_inner**2)
    
    def _get_showerstart_height(self):
        # Identify meaningful shower start for plot via z-height distribution
        nparticles, hasl = np.histogram(
            self.particle_tracks["z_start"] * 1e-5, bins=np.arange(0, 100, 1)
        )

        # Flip arrays to start from higher altitudes going down
        nparticles = np.flip(nparticles)
        hasl = np.flip(hasl)
        
        # Begin Plot one step prior to when more than 10 particles are involved
        shower_start = hasl[np.argmax(nparticles > 10) - 1]

        return shower_start 
    
    def plot_side_profile(self, ax=None, alpha=0.4, color_dict=None):
        """
        Plots a side profile of the particle tracks with optional color coding.

        Args:
            ax (matplotlib.axes.Axes, optional): Axis object to plot on. Defaults to None.
            alpha (float, optional): Transparency level for plotted tracks. Defaults to 0.1.
            color_dict (dict, optional): Dictionary mapping particle names to colors.
                Example: {"proton": "red", "electron": "blue"}.

        Returns:
            matplotlib.axes.Axes: The axis containing the plot.
        """
        shower_start = self._get_showerstart_height()

        if ax is None:
            _, ax = plt.subplots(figsize=(3, 8))

        # If no color dictionary is provided, plot all particles in black
        if color_dict is None:
            color_dict = {}

        legend_handles = []
        colored_particle_ids = set() # Use a set for faster lookup

        # Iterate over the provided colors and plot those separately
        for particle_selection, color in color_dict.items():
            
            # Get list of particle IDs for this selection
            current_ids = self._get_particle_ids_from_selection(particle_selection)
            if not current_ids:
                continue

            # Add these IDs to the set of colored particles so they aren't plotted in black later
            colored_particle_ids.update(current_ids)
            
            # Filter tracks for these IDs
            subset = self.particle_tracks[self.particle_tracks["particle_id"].isin(current_ids)]
            
            if subset.empty:
                continue
            
            segments = np.array([
                [[row["x_start"] * 1e-5, row["z_start"] * 1e-5],
                [row["x_end"] * 1e-5, row["z_end"] * 1e-5]]
                for _, row in subset.iterrows()
            ])
            
            ax.add_collection(LineCollection(
                segments, color=color, alpha=alpha, linewidth=0.5, label=particle_selection, zorder=2
            ))
            
            # Add solid color line for legend
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=particle_selection))
        
        # All other particle types segments will be shown in black
        # Filter where particle_id is NOT in colored_particle_ids
        filtered_df = self.particle_tracks[~self.particle_tracks["particle_id"].isin(colored_particle_ids)].copy()
        
        if not filtered_df.empty:
            all_segments = np.array([
                [[row["x_start"] * 1e-5, row["z_start"] * 1e-5],
                [row["x_end"] * 1e-5, row["z_end"] * 1e-5]]
                for _, row in filtered_df.iterrows()
            ])
            ax.add_collection(LineCollection(
                all_segments, color=color_dict.get('default', 'black'), alpha=alpha, linewidth=0.08, zorder=1
            ))

        # Autoscale to ensure all data is visible
        ax.autoscale()

        
        # Set plot limits and add legend
        ax.set_ylim(0, shower_start)
        if legend_handles:
            ax.legend(handles=legend_handles)
        
        # Set plot limits and add legend
        ax.set_ylim(0, shower_start)

        return ax

    def plot_cher_distribution(self, ax=None, nbins=300, vmin=None, vmax=None, 
                                    show_colorbar=True, use_log=False, auto_center=True,
                                    cmap="binary"):
            """
            Plots the Cherenkov photon distribution on ground
            
            Args:
                use_log (bool): Use logarithmic color scaling. Defaults to True.
                auto_center (bool): Automatically center the plot on the photon pool.
                cmap (str): Matplotlib colormap name. Defaults to "viridis".
            """
            if ax is None:
                _, ax = plt.subplots(figsize=(7, 6))

            # ------------------------
            #  Unit conversion
            # ------------------------
            # Convert coordinates to km and get weights
            x_km = self.cherenkov_photons["x_impact_cm"] * 1e-5
            y_km = self.cherenkov_photons["y_impact_cm"] * 1e-5
            weights = self.cherenkov_photons["photons"]


            # ------------------------
            #  Autocentering
            # ------------------------
            if auto_center:
                # Calculate the centre of gravity for the photon distribution
                total_w = np.sum(weights)
                cx = np.sum(x_km * weights)/ total_w
                cy = np.sum(y_km * weights)/ total_w

                # Set a 2km x 2km window around the core
                # TODO: this should be adjustable
                ax.set_xlim(cx - 1.0, cx + 1.0)
                ax.set_ylim(cy - 1.0, cy + 1.0)

            # ------------------------
            #  Scaling and Norm
            # ------------------------
            # Bin the data and calculate the 99.9 percentile as vmax
            if vmax is None:
                hist_temp, _, _ = np.histogram2d(x_km, y_km, bins=nbins, weights=weights)
                vmax = np.percentile(hist_temp[hist_temp > 0], 99.9)
            

            if use_log:
                # TODO: maybe this can be done prettier?
                if not vmin:
                    vmin = 1.0
                else:
                    pass
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                if not vmin:
                    vmin = 0
                else:
                    pass
                norm = colors.Normalize(vmin=vmin, vmax=vmax)


            # ------------------------
            #  Create Plot
            # ------------------------
            
            # Create our histogram
            mesh = ax.hist2d(
                x_km,
                y_km,
                bins=nbins,
                weights=weights,
                norm=norm,
                cmap=cmap,
            )
            
            # 
            ax.set_aspect("equal", adjustable='box') # important or else the cherenkov pool will not look right
            ax.set_xlabel("X Position [km]")
            ax.set_ylabel("Y Position [km]")


            # Note: This took longer than expected ^^ 
            if show_colorbar:
                from mpl_toolkits.axes_grid1 import make_axes_locatable

                # Get the exakt size of our plot 
                divider = make_axes_locatable(ax)

                # Take small part of the plot and add an axis to it
                cax = divider.append_axes("right", size="5%", pad=0.1)

                # Now we can add our colorbar
                cbar = plt.colorbar(mesh[3], cax=cax)

                # Give it a label and set it :) 
                label = "Number of photons"
                cbar.set_label(label, rotation=270, labelpad=15)

            return ax
    
    def plot_ground_photon_density(self, ax=None, nbins = 200, color = 'black'):
        """Determines and plots the Cherenkov photon density on ground with 
        respect to the radial distance. 

        Args:
            ax (matplotlib.axes.Axes, optional): Axis object to plot on. Defaults to None.
            nbins (int, optional): Number of radial bins. Defaults to 200.
            color (str, optional): Color of plot. Defaults to 'black'.

        Returns:
            matplotlib.axes.Axes: The axis containing the plot.
        """        
        
        if self.zenith_deg > 0:
            print("WARNING: This implementation only works for showers from 0 deg Zenith")
            return
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 6))
            
            
        # Convert the impact Cartesian coordinates into polar coordinates 
        impact_r, _ = self._cartesian_to_polar(
                        self.cherenkov_photons.x_impact_cm * 1e-2,
                        self.cherenkov_photons.y_impact_cm * 1e-2
        )
        
        # Setup logarithmic bins to calculate the photon density for
        density_bins = np.logspace(
                                    np.log10(1),
                                    np.log10(800), 
                                    nbins
        ).reshape((nbins//2,2))
    
        # Open up lists to store the output 
        photon_density = []
        radial_bin_centre = []
        
        # Loop over all ring bins
        for inner_radius, outer_radius in density_bins:
            
            # Calculate ring area and determine number of photons within it
            area = self._ring_area(inner_radius, outer_radius) 
            n_photons = ((impact_r < outer_radius) & (impact_r >= inner_radius)).sum()
            
            # Calculate photon density
            photon_density.append(n_photons/area)
            
            # Calculate radial centre of the ring bin
            bin_centre = (inner_radius +  outer_radius)/2.
            radial_bin_centre.append(bin_centre)
            
            
        plt.plot(radial_bin_centre, photon_density, c = color)
        plt.xlabel('Radial distance [m]')
        plt.ylabel(r'Photon density [m$^{-2}$]')
        plt.xscale('log')
        plt.xlim(10, 1e3)
        
        return ax
            
    def plot_particle_height_distribution(self, ax=None, color_dict=None, height_steps=0.1):
        """
        Plots the distribution of particle starting heights.

        Args:
            ax (matplotlib.axes.Axes, optional): Axis object to plot on. Defaults to None.
            color_dict (dict, optional): Dictionary mapping particle names to colors.
                Example: {"proton": "red", "electron": "blue"}.
            height_steps (float, optional): Step size for height bins in km. Defaults to 0.1.

        Returns:
            matplotlib.axes.Axes: The axis containing the plot.
        """
        
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        
        # Get the height at which the shower started 
        shower_start = self._get_showerstart_height()
            
        legend_handles = []
        
        #plot distribution of all particles first
        n_particles, bins = np.histogram(self.particle_tracks.z_start,
                                        bins = np.arange(0,shower_start, height_steps)*1e5)
        bin_centres = (bins[:-1]+bins[1:])/2.

        ax.plot(bin_centres *1e-5, n_particles, c = 'black')
        legend_handles.append(
            plt.Line2D([0], [0],
            color='black', 
            lw=2,
            label='All particles')
        )
        
        # Now we loop over all of the particle and color combinations that have
        # been provided and plot them separately
        

        if color_dict:
            for particle_selection, color in color_dict.items():
                
                # Get list of particle IDs for this selection
                current_ids = self._get_particle_ids_from_selection(particle_selection)
                
                if not current_ids:
                    continue

                # Select all entries with these particle IDs
                subset = self.particle_tracks[self.particle_tracks["particle_id"].isin(current_ids)]
                
                if subset.empty:
                   continue

                # Create the histogram 
                n_particles, bins = np.histogram(subset.z_start,
                                                bins = np.arange(0,shower_start,height_steps)*1e5
                )
                
                # Now we are done with all subnames and plot things 
                bin_centres = (bins[:-1]+bins[1:])/2.

                ax.plot(bin_centres *1e-5, n_particles, c=color)
                    
                # Add solid color line for legend
                legend_handles.append(plt.Line2D([0], [0],
                                    color=color, 
                                    lw=2, 
                                    label=particle_selection)
                )
            
        plt.legend(handles=legend_handles)
        plt.ylabel('Number of particles')
        plt.xlabel('Height a.s.l [km]')
        
        return ax