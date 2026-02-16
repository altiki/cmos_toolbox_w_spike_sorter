import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd


def convert_elno_to_xy(elno):
    chipWidth = 220
    x = int(elno/chipWidth)
    y = elno % chipWidth
    return x,y

def get_electrode_unit_info(data, pairings, unit_ids):
    spikes = np.array([[int(row[0]), float(row[1]), float(row[2])] for row in data['SPIKEMAT']])
    spikes_extremum = data['SPIKEMAT_EXTREMUM']

    unit_pre_index = pairings[0]
    unit_post_index = pairings[1]    
    unit_pre = unit_ids[unit_pre_index]
    unit_post = unit_ids[unit_post_index]
    if unit_pre != unit_post:
        pre_extremum = [int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_pre_index]))]
        post_extremum = [int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_post_index]))]
        


        electrodes_pre = sorted(data['UNIT_TO_EL'][unit_pre])

        
        electrodes_post = sorted(data['UNIT_TO_EL'][unit_post])

    return spikes, electrodes_pre, electrodes_post, pre_extremum, post_extremum, unit_pre, unit_post


def get_latency_all(spikes, spikes_extremum, unit_ids, unit_pre, unit_post, input_electrode_number, output_electrode_number):
    # Filter spikes for input and output electrodes
    input_spikes = spikes_extremum[spikes_extremum['UnitIdx'] == unit_ids.index(unit_pre)].reset_index(drop=True)
    #input_spikes = pd.DataFrame(spikes[spikes[:, 0] == input_electrode_number], columns=['Electrode', 'Spike_Time', 'Amplitude']).reset_index(drop=True)
    #output_spikes = spikes[spikes['UnitIdx'] == unit_ids.index(unit_post)].reset_index(drop=True)
    output_spikes = pd.DataFrame(spikes[spikes[:, 0] == output_electrode_number], columns=['Electrode', 'Spike_Time', 'Amplitude']).reset_index(drop=True)

    # Create a list to store latencies
    latencies = []
    input_spike_count = 0
    input_time = None
    
    # Combine all spikes and sort by time
    all_spikes = pd.concat([
        input_spikes.assign(is_input=True),
        output_spikes.assign(is_input=False)
    ]).sort_values('Spike_Time').reset_index(drop=True)
    
    for _, spike in all_spikes.iterrows():
        current_time = spike['Spike_Time']
        
        if spike['is_input']:
            # Record as input spike
            latencies.append((input_spike_count, current_time, 0, "input"))
            input_time = current_time
            input_spike_count += 1
        elif input_time is not None:  # This is an output spike and we have a previous input
            # Only record output if it's within 10ms window
            if current_time - input_time < 10:
                latencies.append((input_spike_count-1, current_time, current_time - input_time, "output"))
    
    # Convert list to structured numpy array
    if not latencies:  # If no latencies were found
        return np.array([], dtype=[('input spike', 'i4'), ('spike time', 'f8'),
                                 ('latency', 'f4'), ('category', 'U6')])
    
    return np.array(latencies, dtype=[('input spike', 'i4'), ('spike time', 'f8'),
                                    ('latency', 'f4'), ('category', 'U6')])


def get_latency_after(spikes, input_electrode_number, output_electrode_number):
    # Filter spikes for input and output electrodes
    #input_spikes = spikes_extremum[spikes_extremum['UnitIdx'] == unit_ids.index(unit_pre)].reset_index(drop=True)
    input_spikes = pd.DataFrame(spikes[spikes[:, 0] == input_electrode_number], columns=['Electrode', 'Spike_Time', 'Amplitude']).reset_index(drop=True)
    #output_spikes = spikes[spikes['UnitIdx'] == unit_ids.index(unit_post)].reset_index(drop=True)
    output_spikes = pd.DataFrame(spikes[spikes[:, 0] == output_electrode_number], columns=['Electrode', 'Spike_Time', 'Amplitude']).reset_index(drop=True)

    # Create a list to store latencies
    latencies = []
    input_spike_count = 0
    input_time = None
    
    # Combine all spikes and sort by time
    all_spikes = pd.concat([
        input_spikes.assign(is_input=True),
        output_spikes.assign(is_input=False)
    ]).sort_values('Spike_Time').reset_index(drop=True)
    
    for _, spike in all_spikes.iterrows():
        current_time = spike['Spike_Time']
        
        if spike['is_input']:
            # Record as input spike
            latencies.append((input_spike_count, current_time, 0, "input"))
            input_time = current_time
            input_spike_count += 1
        elif input_time is not None:  # This is an output spike and we have a previous input
            # Only record output if it's within 10ms window
            if current_time - input_time < 10:
                latencies.append((input_spike_count-1, current_time, current_time - input_time, "output"))
    
    # Convert list to structured numpy array
    if not latencies:  # If no latencies were found
        return np.array([], dtype=[('input spike', 'i4'), ('spike time', 'f8'),
                                 ('latency', 'f4'), ('category', 'U6')])
    
    return np.array(latencies, dtype=[('input spike', 'i4'), ('spike time', 'f8'),
                                    ('latency', 'f4'), ('category', 'U6')])

def get_latency_with_extremum(spikes_extremum, unit_ids, unit_pre, unit_post):
    # Filter spikes for input and output electrodes
    input_spikes = spikes_extremum[spikes_extremum['UnitIdx'] == unit_ids.index(unit_pre)].reset_index(drop=True)
    output_spikes = spikes_extremum[spikes_extremum['UnitIdx'] == unit_ids.index(unit_post)].reset_index(drop=True)
    
    # Create a list to store latencies
    latencies = []
    input_spike_count = 0
    input_time = None
    
    # Combine all spikes and sort by time
    all_spikes = pd.concat([
        input_spikes.assign(is_input=True),
        output_spikes.assign(is_input=False)
    ]).sort_values('Spike_Time').reset_index(drop=True)
    
    for _, spike in all_spikes.iterrows():
        current_time = spike['Spike_Time']
        
        if spike['is_input']:
            # Record as input spike
            latencies.append((input_spike_count, current_time, 0, "input"))
            input_time = current_time
            input_spike_count += 1
        elif input_time is not None:  # This is an output spike and we have a previous input
            # Only record output if it's within 10ms window
            if current_time - input_time < 10:
                latencies.append((input_spike_count-1, current_time, current_time - input_time, "output"))
    
    # Convert list to structured numpy array
    if not latencies:  # If no latencies were found
        return np.array([], dtype=[('input spike', 'i4'), ('spike time', 'f8'),
                                 ('latency', 'f4'), ('category', 'U6')])
    
    return np.array(latencies, dtype=[('input spike', 'i4'), ('spike time', 'f8'),
                                    ('latency', 'f4'), ('category', 'U6')])

def plot_latency_and_location_with_extremum_both_plots(save_path, filename, exp_duration, unit_to_el, input_ids, output_ids, unit_pre, unit_post, unit_ids, spikes, spikes_extremum, extremum_output=True):
    # Create color mapping for electrodes
    input_electrodes = sorted(unit_to_el[unit_pre])
    output_electrodes = sorted(unit_to_el[unit_post])   
    electrodes_pre = input_electrodes
    electrodes_post = output_electrodes
    
    colormap_pre = plt.get_cmap('Blues')
    colormap_post = plt.get_cmap('RdPu')
    num_elecs_pre = len(electrodes_pre)
    num_elecs_post = len(electrodes_post)
    colors_pre = [colormap_pre(i / num_elecs_pre) for i in range(num_elecs_pre)]
    colors_post = [colormap_post(i / num_elecs_post) for i in range(num_elecs_post)]
    
    electrode_values_pre = electrodes_pre
    electrode_values_post = electrodes_post
    color_values = []
    color_values.extend(colors_pre)
    color_values.extend(colors_post)
    elec_to_color = {'electrode': electrode_values_pre + electrode_values_post, 'color': color_values}
    
    for idx, input_id in enumerate(input_ids):
        latency_extremum = None
        input_electrode = np.array([input_id], dtype=float)
        input_color = elec_to_color['color'][elec_to_color['electrode'].index(input_id)]
        #input_spikes = spikes_extremum['Spike_Time'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_pre)]
        output_electrodes_filtered = np.array(output_ids)

        # Create figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # First subplot - Latency plot
        ax1.set_ylabel("Experiment Time (s)")
        ax1.set_xlim((0., 10.))
        ax1.set_xlabel("Latency (ms)")
        validation = None
        
        for output_electrode in output_electrodes_filtered:
            output_color = elec_to_color['color'][elec_to_color['electrode'].index(output_electrode)]
            if output_color is not None:
                latency_all = get_latency_all(spikes, spikes_extremum, unit_ids, unit_pre, unit_post, input_id, output_electrode)
                
                latency_all = latency_all[latency_all['latency'] < 10]

                input_before = latency_all[latency_all['category'] == 'input']
                output_before = latency_all[latency_all['category'] == 'output']
                
                ax1.scatter(output_before['latency'], output_before['spike time'] / 1000, s=7, label='output', color=output_color)
                ax1.scatter(input_before['latency'], input_before['spike time'] / 1000, s=7, label='input', color=input_color)

                if extremum_output:
                    latency_extremum = get_latency_with_extremum(spikes_extremum, unit_ids, unit_pre, unit_post)
                    latency_extremum = latency_extremum[latency_extremum['latency'] < 10]
                    output_extremum = latency_extremum[latency_extremum['category'] == 'output']
                    ax1.scatter(output_extremum['latency'], output_extremum['spike time'] / 1000, s=10, label='output extremum', color='green')
                
                ax1.set_title(f"Source {unit_pre} to Target {unit_post}")
                handles, labels = ax1.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())
        
        # Second subplot - Electrode location plot
        ax2.invert_yaxis()
        for electrode in electrodes_pre:
            x, y = convert_elno_to_xy(electrode)
            ax2.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)], marker ='s')
        for electrode in electrodes_post:
            x, y = convert_elno_to_xy(electrode)
            ax2.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)], marker ='s')
        
        # Mark extremum points
        pre_extremum_electrode = int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_pre)]))
        post_extremum_electrode = int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_post)]))
        
        x, y = convert_elno_to_xy(pre_extremum_electrode)
        ax2.scatter(y, x, color='yellow', marker='x', s=100)
        
        x, y = convert_elno_to_xy(post_extremum_electrode)
        ax2.scatter(y, x, color='green', marker='x', s=100)
        
        ax2.set_xlim(0, 220)
        ax2.set_ylim(0, 120)
        ax2.set_title(f"Pair presynaptic {unit_pre} and postsynaptic {unit_post}")
        
        plt.tight_layout()
        
      
        plt.savefig(os.path.join(save_path, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}.pdf"), format='pdf', dpi=300)
        plt.savefig(os.path.join(save_path, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}.png"), format='png')
        plt.close()

    return latency_extremum


def plot_latency_and_location_after(save_path, filename, exp_duration, unit_to_el, input_ids, output_ids, unit_pre, unit_post, unit_ids, spikes, spikes_extremum, extremum_output=False):
    # Create color mapping for electrodes
    input_electrodes = sorted(unit_to_el[unit_pre])
    output_electrodes = sorted(unit_to_el[unit_post])   
    electrodes_pre = input_electrodes
    electrodes_post = output_electrodes
    
    colormap_pre = plt.get_cmap('Blues')
    colormap_post = plt.get_cmap('RdPu')
    num_elecs_pre = len(electrodes_pre)
    num_elecs_post = len(electrodes_post)
    colors_pre = [colormap_pre(i / num_elecs_pre) for i in range(num_elecs_pre)]
    colors_post = [colormap_post(i / num_elecs_post) for i in range(num_elecs_post)]
    
    electrode_values_pre = electrodes_pre
    electrode_values_post = electrodes_post
    color_values = []
    color_values.extend(colors_pre)
    color_values.extend(colors_post)
    elec_to_color = {'electrode': electrode_values_pre + electrode_values_post, 'color': color_values}
    
    for idx, input_id in enumerate(input_ids):
        latency_extremum = None
        input_electrode = np.array([input_id], dtype=float)
        input_color = elec_to_color['color'][elec_to_color['electrode'].index(input_id)]
        #input_spikes = spikes_extremum['Spike_Time'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_pre)]
        output_electrodes_filtered = np.array(output_ids)

        # Create figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # First subplot - Latency plot
        ax1.set_ylabel("Experiment Time (s)")
        ax1.set_xlim((0., 10.))
        ax1.set_xlabel("Latency (ms)")
        validation = None
        
        for output_electrode in output_electrodes_filtered:
            output_color = elec_to_color['color'][elec_to_color['electrode'].index(output_electrode)]
            if output_color is not None:
                latency_all = get_latency_after(spikes, input_electrode, output_ids)
                
                latency_all = latency_all[latency_all['latency'] < 10]

                input_before = latency_all[latency_all['category'] == 'input']
                output_before = latency_all[latency_all['category'] == 'output']
                
                ax1.scatter(output_before['latency'], output_before['spike time'] / 1000, s=7, label='output', color=output_color)
                ax1.scatter(input_before['latency'], input_before['spike time'] / 1000, s=7, label='input', color=input_color)

                if extremum_output:
                    latency_extremum = get_latency_with_extremum(spikes_extremum, unit_ids, unit_pre, unit_post)
                    latency_extremum = latency_extremum[latency_extremum['latency'] < 10]
                    output_extremum = latency_extremum[latency_extremum['category'] == 'output']
                    ax1.scatter(output_extremum['latency'], output_extremum['spike time'] / 1000, s=10, label='output extremum', color='green')
                
                ax1.set_title(f"Source {unit_pre} to Target {unit_post}")
                handles, labels = ax1.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())
        
        # Second subplot - Electrode location plot
        ax2.invert_yaxis()
        for electrode in electrodes_pre:
            x, y = convert_elno_to_xy(electrode)
            ax2.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)], marker ='s')
        for electrode in electrodes_post:
            x, y = convert_elno_to_xy(electrode)
            ax2.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)], marker ='s')
        
        # Mark extremum points
        pre_extremum_electrode = int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_pre)]))
        post_extremum_electrode = int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_post)]))
        
        x, y = convert_elno_to_xy(pre_extremum_electrode)
        ax2.scatter(y, x, color='yellow', marker='x', s=100)
        
        x, y = convert_elno_to_xy(post_extremum_electrode)
        ax2.scatter(y, x, color='green', marker='x', s=100)
        
        ax2.set_xlim(0, 220)
        ax2.set_ylim(0, 120)
        ax2.set_title(f"Pair presynaptic {unit_pre} and postsynaptic {unit_post}")
        
        plt.tight_layout()
        
      
        plt.savefig(os.path.join(save_path, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}.pdf"), format='pdf', dpi=300)
        plt.savefig(os.path.join(save_path, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}.png"), format='png')
        plt.close()

    return latency_extremum