
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from src.utils.logger_functions import console
import json
from abc import ABC, abstractmethod
import pandas as pd
import hdbscan
from scipy import stats
from sklearn.preprocessing import StandardScaler




class ABCPlotter(ABC):
    def __init__(self, main_path: str, pair_path: str, input_path: str):
        self.label_size = 21
        self.legend_size = 18
        self.font_size = 24
        self.grid_style = ":"
        self.grid_alpha = 0.75
        self.main_path = main_path
        self.dpi = 300

class LatencyPlotter(ABCPlotter):
    def __init__(self, filename: str, main_path: str, input_path: str, pair_path: str):
        super().__init__(main_path = main_path, pair_path = pair_path, input_path=input_path)

        self.filename = filename
        self.chip_id = self.filename.split('_')[0].replace('ID','')
        self.div = self.filename.split('_')[2].replace('DIV','')
        self.area = self.filename.split('_')[1]
        self.input_path = input_path
        self.pair_path = pair_path
        self.chip_path = f'{self.chip_id}_HeartShape/{self.chip_id}_DIV{self.div}_HeartShape/'
        self.filepath = os.path.join(self.input_path, self.chip_path)
        self.pairpath = os.path.join(self.pair_path, self.chip_path)
        self.sorterpath =  os.path.join(self.input_path, self.chip_path)
        self.full_sorterpath = os.path.join(self.sorterpath, f'Sorter_{self.filename}')
        self.savepath = os.path.join(self.main_path+'/Latency_plot_by_amplitude/',self.chip_path)
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.sample_frequency = 20000
        self.chip_height = 120
        self.chip_width = 220
        self.max_pre = 3

        self.pairings = None
        self.spike_dict = None
        self.spikes = None
        self.spikes_extremum = None
        self.unit_ids = None
        self.unit_to_el_mapping = None
        self.latency = None
        self.elec_to_color = None

        
        self.pair_data = {
            'electrodes_pre': [],
            'electrodes_post': [],
            'unit_pre': [],
            'unit_post': [],
            'pre_extremum': [],
            'post_extremum': []

        }

        self.latency_data = {}

        with open(os.path.join(self.filepath,  self.filename[:-3] + '_processed.pkl'), 'rb') as f:
            try:
                self.spike_dict = pickle.load(f)
                self.spikes = np.array([[int(row[0]), float(row[1]), float(row[2])] for row in self.spike_dict['SPIKEMAT']])
                self.spikes_extremum = self.spike_dict["SPIKEMAT_EXTREMUM"]
                self.unit_to_el_mapping = self.spike_dict['UNIT_TO_EL']
            except Exception as e:
                console.error(f"An error occurred while loading the spike dictionary: {e}")

        with open(os.path.join(self.pairpath, 'pairing_of_units.pkl'), 'rb') as f:
            pairings_all = pickle.load(f)
            self.pairings = pairings_all[self.area]

        with open(os.path.join(self.full_sorterpath, 'wf_folder_curated/sparsity.json'), 'rb') as f:
            sorting_info = json.load(f)
            self.unit_ids = sorting_info['unit_ids']
        

    def get_electrode_unit_info(self):
        for row_idx, pair in enumerate(self.pairings):
            unit_post = [list(self.pairings.keys())[row_idx]]
            unit_pre = []
            for i in range(self.max_pre):
                if self.pairings[unit_post[0]][i] is not None:
                    unit_pre.append(self.pairings[unit_post[0]][i])
            unit_post = unit_post * len(unit_pre)
            if unit_pre:
                for i in range(len(unit_pre)):
                    post_extremum = [int(np.unique(self.spikes_extremum['Electrode'][self.spikes_extremum['UnitIdx'] == self.unit_ids.index(unit_post[i])]))]
                    pre_extremum = [int(np.unique(self.spikes_extremum['Electrode'][self.spikes_extremum['UnitIdx'] == self.unit_ids.index(unit_pre[i])]))]
                    self.pair_data['post_extremum'].append(post_extremum)
                    self.pair_data['pre_extremum'].append(pre_extremum)

                    electrodes_pre = sorted(self.unit_to_el_mapping[unit_pre[i]])
                    electrodes_post = sorted(self.unit_to_el_mapping[unit_post[i]])

                    self.pair_data['electrodes_pre'].append(electrodes_pre)
                    self.pair_data['electrodes_post'].append(electrodes_post)

                    self.pair_data['unit_pre'].append(unit_pre[i])
                    self.pair_data['unit_post'].append(unit_post[i])

        return self.pair_data

    def get_latency(self, input_electrode_number, output_electrode_number):
        # Filter spikes for input and output electrodes
        spikes = self.spikes
        input_spikes = spikes[spikes[:, 0] == input_electrode_number]
        output_spikes = spikes[spikes[:, 0] == output_electrode_number]
        
        # Sort spikes by time
        input_spikes = input_spikes[np.argsort(input_spikes[:, 1])]
        output_spikes = output_spikes[np.argsort(output_spikes[:, 1])]

        # Preallocate latency array with an upper bound on size
        max_possible_size = len(input_spikes) + len(output_spikes)
        latency = np.zeros(max_possible_size, dtype=[('input spike', 'i4'), ('spike time', 'i4'), 
                                                    ('latency', 'f4'), ('category', 'U6')])

        input_spike_count = 0
        input_index = 0
        output_index = 0
        index = 0

        # Iterate through both input and output spikes by merge sort technique
        while input_index < len(input_spikes) and output_index < len(output_spikes):
            input_spike_time = input_spikes[input_index][1]
            output_spike_time = output_spikes[output_index][1]
            
            if input_spike_time <= output_spike_time:
                input_time = input_spike_time
                latency[index] = (input_spike_count, input_time, 0, "input")
                input_spike_count += 1
                input_index += 1
            else:
                if input_index > 0:  # There has been at least one input spike
                    latency[index] = (input_spike_count, output_spike_time, output_spike_time - input_time, "output")
                output_index += 1
            index += 1

        # Process remaining input spikes
        while input_index < len(input_spikes):
            input_time = input_spikes[input_index][1]
            latency[index] = (input_spike_count, input_time, 0, "input")
            input_spike_count += 1
            input_index += 1
            index += 1

        # Process remaining output spikes
        while output_index < len(output_spikes):
            output_time = output_spikes[output_index][1]
            if input_index > 0:  # There has been at least one input spike
                latency[index] = (input_spike_count, output_time, output_time - input_time, "output")
            output_index += 1
            index += 1

        latency = latency[:index]
        self.latency = latency
        # Return only the populated part of the latency array
        return self.latency

    def convert_elno_to_xy(self, elno):
        x = int(elno/self.chip_width)
        y = elno % self.chip_width
        return x,y

    def get_elec_color(self, electrode, elec_to_color):
        elec_to_color_electrodes = np.array(elec_to_color['electrode'])
        elec_to_color_colors = np.array(elec_to_color['color'])
        location = np.where(elec_to_color_electrodes == electrode)[0]
        if len(location) > 0:
            self.elec_to_color = elec_to_color_colors[location[0]]
        return self.elec_to_color
    
    def plot_color_coded_electrodes(self, savepath, unit_pre, unit_post, electrodes_pre, electrodes_post):
        electrode_values_pre = []
        color_values_pre = []

        electrode_values_post = []
        color_values_post = []

        colormap_pre = plt.get_cmap('Blues')
        colormap_post = plt.get_cmap('RdPu')
        num_elecs_pre = len(electrodes_pre)
        num_elecs_post = len(electrodes_post)
        colors_pre = [colormap_pre(i / num_elecs_pre) for i in range(num_elecs_pre)]
        colors_post = [colormap_post(i / num_elecs_post) for i in range(num_elecs_post)]
        
        electrode_values_pre.extend(electrodes_pre)
        color_values_pre.extend(colors_pre)
        electrode_values_post.extend(electrodes_post)
        color_values_post.extend(colors_post)

        color_values = []
        color_values.extend(colors_pre)
        color_values.extend(colors_post)
        elec_to_color = {'electrode': electrode_values_pre + electrode_values_post, 'color': color_values}

        plt.figure(figsize=(8, 5))
        plt.gca().invert_yaxis()
        for electrode in electrodes_pre:
            x, y = self.convert_elno_to_xy(electrode)
            plt.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)])
        for electrode in electrodes_post:
            x, y = self.convert_elno_to_xy(electrode)
            plt.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)])
        plt.xlim(0, 220)
        plt.ylim(0, 120)
        plt.title(f"Pair presynaptic {unit_pre} and postsynaptic {unit_post}")
        plt.savefig(os.path.join(savepath, f"area_{self.area}_color_code_pre_{unit_pre}_post_{unit_post}.pdf"), format='pdf', dpi=300)
        plt.savefig(os.path.join(savepath, f"area_{self.area}_color_code_pre_{unit_pre}_post_{unit_post}.png"), format='png')
        plt.close()

        return elec_to_color

    def plot_latency_and_location(self, savepath, input_ids, output_ids, unit_pre, unit_post, electrodes_pre, electrodes_post):
        include_input_in_latency = True
        spikes = self.spikes
        elec_to_color_dict = self.plot_color_coded_electrodes(savepath, unit_pre, unit_post, electrodes_pre, electrodes_post)
        print(elec_to_color_dict)
        gain = 6.3
        for idx, input_id in enumerate(input_ids):
            if include_input_in_latency:
                output_ids = list(set(output_ids + [id for id in input_ids if id != input_id]))

            input_electrode = np.array([input_id], dtype=float)
            input_color = elec_to_color_dict['color'][elec_to_color_dict['electrode'].index(input_id)]
            input_spikes = spikes[spikes[:, 0] == input_electrode]
            output_electrodes = spikes[np.isin(spikes[:, 0], output_ids)]
            output_electrodes_filtered = np.unique(output_electrodes[:, 0])

            print(f'Plotting latencies for input {input_electrode}')

            fig1, ax1 = plt.subplots()
            ax1.set_ylabel("Experiment Time (s)")
            ax1.set_xlim((0., 10.))
            ax1.set_xlabel("Latency (ms)")
            #ax1.scatter(input_spikes[:, 1], input_spikes[:, 2] / 1000, s=7, label='input', color=input_color)
            #modify figure size
            #fig1.set_size_inches(10, 30)

            for output_electrode in output_electrodes_filtered:
                key = f'input_{input_id}_output_{output_electrode}'
                output_color = elec_to_color_dict['color'][elec_to_color_dict['electrode'].index(output_electrode)]
                output_spikes = spikes[spikes[:, 0] == output_electrode]
                output_spikes[:, 1] = output_spikes[:, 1].astype(int)
                
                if output_color is not None:
                    latency = self.get_latency(input_electrode, output_electrode)
                    
                    input_before = latency[latency['category'] == 'input']
                    output_before = latency[latency['category'] == 'output']
                    #with open(os.path.join(self.savepath, f"input_el_{input_id}_output_el_{int(output_electrode)}.pkl"), 'wb') as f:
                    #    pickle.dump(latency, f)

                    ax1.scatter(input_before['latency'], input_before['spike time'] / 1000, s=7, label='input', color=input_color)
                    ax1.scatter(output_before['latency'], output_before['spike time'] / 1000, s=7, label='output', color=output_color)

                    #For color coding the spikes based on unit index
                    #target_spike_times = np.array(spikes_extremum['Spike_Time'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_post[0])], dtype=int)
                    #latency_extremum = output_before[np.isin(output_before['spike time'], target_spike_times)]

                    output_before_amplitude_color = np.zeros(output_before.shape[0], dtype=[('latency', 'f8'), ('spike time', 'f8'), ('amplitude', 'f8')])
                    output_before_amplitude_color['latency'] = output_before['latency']
                    output_before_amplitude_color['spike time'] = output_before['spike time'] /1000
                    output_before_amplitude_color['amplitude'] = (np.array([output_spikes[output_spikes[:, 1] == spike_time, 2][0] for spike_time in output_before['spike time']])*gain).tolist()
                    
                    #plot the output_before_amplitude_color with amplitude as color
                    ax1.scatter(output_before_amplitude_color['latency'], output_before_amplitude_color['spike time'], s=20, c=output_before_amplitude_color['amplitude'], cmap='viridis', label='output spikes')
                    #add colorbar
                    cbar = plt.colorbar(ax1.scatter(output_before_amplitude_color['latency'], output_before_amplitude_color['spike time'], s=20, c=output_before_amplitude_color['amplitude'], cmap='viridis', label='output spikes'))
                    cbar.set_label('Amplitude uV')
                    ax1.legend()

                    self.latency_data[key] = output_before_amplitude_color
                    #ax1.scatter(output_spikes[:, 1], output_spikes[:, 2] / 1000, s=20, c=output_spikes[:, 2], cmap='viridis', label='output spikes')
                    #print(len(latency_extremum)/len(output_before))
                    #ax1.scatter(latency_extremum['latency'], latency_extremum['spike time'] / 1000, s=20, label='output extremum', color='green')
                    

                plt.savefig(os.path.join(savepath, f"area_{self.area}_input_el_{input_id}_unit_pre_{unit_pre}_output_el_{int(output_electrode)}_unit_post_{unit_post}_STTRP.pdf"), format='pdf')
                plt.savefig(os.path.join(savepath, f"area_{self.area}_input_el_{input_id}_unit_pre_{unit_pre}_output_el_{int(output_electrode)}_unit_post_{unit_post}_STTRP.png"), format='png')
            #plt.show()
            plt.close(fig1)

            fig2, ax2 = plt.subplots()
            ax2.set_title(f"Color code for input electrode {input_id}")
            ax2.invert_yaxis()

            for electrode in output_ids:
                x, y = self.convert_elno_to_xy(electrode)
                ax2.scatter(y, x, color=elec_to_color_dict['color'][elec_to_color_dict['electrode'].index(electrode)])

            x, y = self.convert_elno_to_xy(input_id)
            ax2.scatter(y, x, color=elec_to_color_dict['color'][elec_to_color_dict['electrode'].index(input_id)])
            ax2.scatter(y, x, color='goldenrod', marker='x', s=30, label=f'Input {input_id}')

            ax2.set_xlim(0, 220)
            ax2.set_ylim(0, 120)
            ax2.legend()
            plt.savefig(os.path.join(savepath, f"area_{self.area}_input_el_{input_id}_color_code.pdf"), format='pdf', dpi=300)
            plt.savefig(os.path.join(savepath, f"area_{self.area}_input_el_{input_id}_color_code.pdf"), format='png')
            plt.close(fig2)
        return
    

    def run_for_all_pairs(self):
        pair_data = self.get_electrode_unit_info()
        for pre_extremum, post_extremum, electrodes_pre, electrodes_post, unit_pre, unit_post in zip(pair_data['pre_extremum'], pair_data['post_extremum'], pair_data['electrodes_pre'], pair_data['electrodes_post'], pair_data['unit_pre'], pair_data['unit_post']):
            console.info("Plotting latencies for {unit_pre} and {unit_post}...")
            original_directionality = os.path.join(self.savepath, 'original_directionality/')
            if not os.path.exists(original_directionality):
                os.makedirs(original_directionality)
            inverted_directionality = os.path.join(self.savepath, 'inverted_directionality/')
            if not os.path.exists(inverted_directionality):
                os.makedirs(inverted_directionality)
            #elec_to_color = self.plot_color_coded_electrodes(original_directionality, unit_pre, unit_post, electrodes_pre, electrodes_post)
            self.plot_latency_and_location(original_directionality, pre_extremum, post_extremum, unit_pre, unit_post, electrodes_pre, electrodes_post)

            #elec_to_color = self.plot_color_coded_electrodes(inverted_directionality, unit_post, unit_pre, electrodes_post, electrodes_pre)
            self.plot_latency_and_location(inverted_directionality, post_extremum, pre_extremum, unit_post, unit_pre, electrodes_post, electrodes_pre)
        
        with open(os.path.join(self.savepath, f"{self.filename[:-3]}_latencies_for_clustering.pkl"), 'wb') as f:
                        pickle.dump(self.latency_data, f)




class ClusteringPlotter(ABCPlotter):
    def __init__(self, filename: str, main_path: str, input_path: str, pair_path: str):
        super().__init__(main_path = main_path, input_path = input_path, pair_path = pair_path)

        self.filename = filename
        self.chip_id = self.filename.split('_')[0].replace('ID','')
        self.div = self.filename.split('_')[2].replace('DIV','')
        self.area = self.filename.split('_')[1]
        self.input_path = input_path
        self.chip_path = f'{self.chip_id}_HeartShape/{self.chip_id}_DIV{self.div}_HeartShape/'
        self.filepath = os.path.join(self.main_path+self.input_path, self.chip_path)
        self.savepath = os.path.join(self.main_path+'/Latency_plot_clustered/',self.chip_path)
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.latency_data = None
        
        with open(os.path.join(self.filepath, self.filename[:-3] + '_latencies_for_clustering.pkl'), 'rb') as f:
            try:
                self.latency_data = pickle.load(f)
            except Exception as e:
                console.error(f"An error occurred while loading the latency data: {e}")

    def perform_hdbscan_clustering(self, data, min_cluster_size=40, min_samples=15, epsilon=0.15):
        """
        Perform HDBSCAN clustering with stricter requirements for cluster formation.
        """
        X = data['latency'].values.reshape(-1, 1)
        
        # Add small jitter to prevent numerical issues
        X = X + np.random.normal(0, np.ptp(X) * 1e-10, X.shape)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom',
            metric='euclidean',
            cluster_selection_epsilon=float(epsilon),
            allow_single_cluster=True,
            prediction_data=True
        )
        
        cluster_labels = clusterer.fit_predict(X)
        
        # More stringent post-processing for clusters
        unique_labels = set(cluster_labels) - {-1}
        for label in unique_labels:
            cluster_points = X[cluster_labels == label]
            
            # Calculate temporal density and other metrics
            temporal_range = np.ptp(cluster_points)
            if temporal_range > 0:
                temporal_density = len(cluster_points) / temporal_range
                
                # Multiple criteria for a valid cluster
                is_valid_cluster = (
                    temporal_density >= 30 and  # Increased density requirement
                    len(cluster_points) >= min_cluster_size and  # Size requirement
                    temporal_range <= 0.5  # Maximum spread requirement (in ms)
                )
                
                if not is_valid_cluster:
                    cluster_labels[cluster_labels == label] = -1
        
        result_df = data.copy()
        result_df['cluster'] = cluster_labels
        
        return result_df, clusterer

    def find_density_peaks(self, latencies, n_bins=50):
        """
        Find peaks in the density distribution of latencies using histogram binning
        instead of KDE to avoid singularity issues.
        """
        # Add small random noise to break ties
        latencies_jittered = latencies + np.random.normal(0, 1e-6, size=len(latencies))
        
        # Create histogram
        hist, bin_edges = np.histogram(latencies_jittered, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Smooth the histogram slightly
        smooth_hist = np.convolve(hist, [0.25, 0.5, 0.25], mode='same')
        
        # Find peaks
        peak_indices = []
        min_peak_height = np.max(smooth_hist) * 0.1  # Only consider peaks above 10% of max density
        
        for i in range(1, len(smooth_hist)-1):
            if (smooth_hist[i] > smooth_hist[i-1] and 
                smooth_hist[i] > smooth_hist[i+1] and 
                smooth_hist[i] > min_peak_height):
                peak_indices.append(i)
        
        peaks = bin_centers[peak_indices]
        peak_heights = smooth_hist[peak_indices]
        
        # Sort peaks by height
        sorted_idx = np.argsort(peak_heights)[::-1]
        return peaks[sorted_idx]

    def optimize_clustering(self, data, min_clusters=2, max_clusters=6):
        """
        Optimize HDBSCAN parameters with stricter cluster requirements.
        """
        X = data['latency'].values.reshape(-1, 1)
        
        # Parameter ranges for stricter clustering
        min_cluster_sizes = range(40, 80, 5)  # Increased minimum size
        epsilon_range = [float(x) for x in np.arange(0.1, 0.3, 0.02)]
        min_samples_range = [15, 20, 25]  # Increased minimum samples
        
        best_score = -np.inf
        best_params = None
        best_n_clusters = 0
        results = []
        
        for min_cluster_size in min_cluster_sizes:
            for epsilon in epsilon_range:
                for min_samples in min_samples_range:
                    if min_samples > min_cluster_size:
                        continue
                    
                    try:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_method='eom',
                            metric='euclidean',
                            cluster_selection_epsilon=float(epsilon),
                            allow_single_cluster=True
                        )
                        
                        clusterer.fit(X)
                        labels = clusterer.labels_
                        
                        # Apply strict criteria to each cluster
                        valid_clusters = 0
                        cluster_stats = []
                        
                        for label in set(labels) - {-1}:
                            cluster_points = X[labels == label]
                            cluster_size = len(cluster_points)
                            temporal_range = np.ptp(cluster_points)
                            
                            if temporal_range > 0:
                                temporal_density = cluster_size / temporal_range
                                
                                # Check if cluster meets strict criteria
                                if (temporal_density >= 30 and 
                                    cluster_size >= min_cluster_size and 
                                    temporal_range <= 0.5):
                                    valid_clusters += 1
                                    cluster_stats.append({
                                        'density': temporal_density,
                                        'size': cluster_size,
                                        'range': temporal_range
                                    })
                        
                        n_clusters = valid_clusters
                        
                        if min_clusters <= n_clusters <= max_clusters and cluster_stats:
                            # Calculate scores based on strict criteria
                            densities = [stat['density'] for stat in cluster_stats]
                            sizes = [stat['size'] for stat in cluster_stats]
                            ranges = [stat['range'] for stat in cluster_stats]
                            
                            # Scoring that heavily favors dense, compact clusters
                            density_score = np.mean(densities) / 50  # Higher density requirement
                            size_score = min(sizes) / min_cluster_size
                            compactness_score = 1 - np.mean(ranges)
                            
                            # Calculate noise ratio (prefer more noise now)
                            noise_ratio = np.sum(labels == -1) / len(X)
                            noise_score = noise_ratio if noise_ratio > 0.4 else 0  # Prefer higher noise ratio
                            
                            combined_score = (
                                density_score * 0.4 +
                                size_score * 0.2 +
                                compactness_score * 0.2 +
                                noise_score * 0.2
                            )
                            
                            results.append({
                                'min_cluster_size': min_cluster_size,
                                'min_samples': min_samples,
                                'epsilon': epsilon,
                                'n_clusters': n_clusters,
                                'score': combined_score,
                                'noise_ratio': noise_ratio
                            })
                            
                            if combined_score > best_score:
                                best_score = combined_score
                                best_params = {
                                    'min_cluster_size': min_cluster_size,
                                    'min_samples': min_samples,
                                    'epsilon': epsilon
                                }
                                best_n_clusters = n_clusters
                                    
                    except Exception as e:
                        continue
        
        return {
            'best_params': best_params,
            'n_clusters_found': best_n_clusters,
            'best_score': best_score,
            'all_results': pd.DataFrame(results) if results else None
        }


    def plot_clusters(self, data, clusterer):
        """
        Plot the clustering results.
        """
        plt.figure(figsize=(12, 8))

        unique_clusters = sorted(set(data['cluster'].unique()) - {-1})

        # Plot noise points first
        noise_mask = data['cluster'] == -1
        if noise_mask.any():
            plt.scatter(data.loc[noise_mask, 'latency'], 
                        data.loc[noise_mask, 'spike time'],
                        c='lightgray', alpha=0.5, s=100, label='Noise')

        # Plot clustered points
        scatter = plt.scatter(data.loc[~noise_mask, 'latency'],
                                data.loc[~noise_mask, 'spike time'],
                                c=data.loc[~noise_mask, 'cluster'],
                                cmap='viridis', alpha=0.6, s=100)

        # Plot cluster centers as vertical lines
        for cluster in unique_clusters:
            center = data[data['cluster'] == cluster]['latency'].mean()
            plt.axvline(x=center, color='red', linestyle='--', alpha=0.5,
                        label=f'Cluster {cluster} center')

        plt.xlabel('Latency (ms)')
        plt.ylabel('Experiment Time (s)')
        plt.title(f'HDBSCAN Clustering (Found {len(unique_clusters)} clusters)')
        if len(unique_clusters) > 0:
            plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 10)
        return plt.gcf()
    
    def run_clustering_for_all_pairs(self):
        for key, value in self.latency_data.items():
            data = pd.DataFrame(value)
            data = data[data['latency'] <= 10]
            # Optimize clustering parameters for 2-6 clusters
            optimization_results = self.optimize_clustering(data, min_clusters=2, max_clusters=6)

            if optimization_results['best_params'] is not None:
                # Perform clustering with optimal parameters
                result_df, clusterer = self.perform_hdbscan_clustering(
                    data,
                    **optimization_results['best_params']
                )
                
                # Plot results
                plt.figure(figsize=(12, 8))
                self.plot_clusters(result_df, clusterer)
                plt.savefig(os.path.join(self.savepath, f"{self.area}_{key}_clustered.pdf"), format='pdf')
                plt.savefig(os.path.join(self.savepath, f"{self.area}_{key}_clustered.png"), format='png')
                plt.close()
            
                result_df.to_csv(os.path.join(self.savepath, f"{self.area}_{key}_clustered.csv"), index=False)
        return
    


    


        

