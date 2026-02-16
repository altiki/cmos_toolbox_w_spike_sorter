
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

def get_elec_color(electrode, elec_to_color):
    elec_to_color_electrodes = np.array(elec_to_color['electrode'])
    elec_to_color_colors = np.array(elec_to_color['color'])
    location = np.where(elec_to_color_electrodes == electrode)[0]
    return elec_to_color_colors[location[0]] if len(location) > 0 else None

def get_electrode_unit_info(data, pairings, area, unit_ids):
    spikes = np.array([[int(row[0]), float(row[1]), float(row[2])] for row in data['SPIKEMAT']])
    spikes_extremum = data['SPIKEMAT_EXTREMUM']
    max_pre = 3
    electrodes_pre_all = []
    electrodes_post_all = []
    pre_extremum_all = []
    post_extremum_all = []
    unit_pre_all = []
    unit_post_all = []
    for pair_idx, pair in enumerate(pairings[area]):
        unit_post = [list(pairings[area].keys())[pair_idx]]
        
        unit_pre = []
        for i in range(0, max_pre):
            if pairings[area][unit_post[0]][i] is not None:
                unit_pre.append(pairings[area][unit_post[0]][i])
        unit_post = unit_post * len(unit_pre)
        if unit_pre:
            for i in range(len(unit_pre)):
                post_extremum = [int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_post[i])]))]
                pre_extremum = [int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_pre[i])]))]
                post_extremum_all.append(post_extremum)
                pre_extremum_all.append(pre_extremum)

                electrodes_pre = sorted(data['UNIT_TO_EL'][unit_pre[i]])
                #electrodes_pre = [data['UNIT_TO_EL'][unit] for unit in unit_pre]
                #electrodes_pre = sorted([item for sublist in electrodes_pre for item in sublist])
                electrodes_pre_all.append(electrodes_pre)
                
                electrodes_post = sorted(data['UNIT_TO_EL'][unit_post[i]])
                #electrodes_post = [data['UNIT_TO_EL'][unit] for unit in unit_post]
                #electrodes_post = sorted([item for sublist in electrodes_post for item in sublist])
                electrodes_post_all.append(electrodes_post)
                unit_post_all.append(unit_post[i])
                unit_pre_all.append(unit_pre[i])

    return spikes, electrodes_pre_all, electrodes_post_all, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all  

def get_electrode_unit_info_te(data, pairings, area, unit_ids):
    spikes = np.array([[int(row[0]), float(row[1]), float(row[2])] for row in data['SPIKEMAT']])
    spikes_extremum = data['SPIKEMAT_EXTREMUM']
    electrodes_pre_all = []
    electrodes_post_all = []
    pre_extremum_all = []
    post_extremum_all = []
    unit_pre_all = []
    unit_post_all = []
    unit_indices_pre = pairings['source'].astype(int)
    unit_indices_post = pairings['target'].astype(int)
    lag_all = pairings['lag']
    for unit_pre_idx, unit_post_idx in zip(unit_indices_pre, unit_indices_post):

        unit_pre = unit_ids[unit_pre_idx]
        unit_post = unit_ids[unit_post_idx]
        if unit_pre != unit_post:
            print(unit_pre,unit_post)
            print(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_post_idx]))
            post_extremum = [int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_post_idx]))]
            pre_extremum = [int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_pre_idx]))]
            post_extremum_all.append(post_extremum)
            pre_extremum_all.append(pre_extremum)

            electrodes_pre = sorted(data['UNIT_TO_EL'][unit_pre])
            #electrodes_pre = [data['UNIT_TO_EL'][unit] for unit in unit_pre]
            #electrodes_pre = sorted([item for sublist in electrodes_pre for item in sublist])
            electrodes_pre_all.append(electrodes_pre)
            
            electrodes_post = sorted(data['UNIT_TO_EL'][unit_post])
            #electrodes_post = [data['UNIT_TO_EL'][unit] for unit in unit_post]
            #electrodes_post = sorted([item for sublist in electrodes_post for item in sublist])
            electrodes_post_all.append(electrodes_post)
            unit_pre_all.append(unit_pre)
            unit_post_all.append(unit_post)


    return spikes, electrodes_pre_all, electrodes_post_all, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all



def calculate_latency(lag, exp_duration, unit_to_el, input_ids, output_ids, unit_pre, unit_post, unit_ids, spikes, spikes_extremum, extremum_output=None):
    all_results = []
    
    for idx, input_id in enumerate(input_ids):
        latency_extremum = None
        validation = None
        
        for output_electrode in np.array(output_ids):
            latency_all = get_latency_all(spikes, spikes_extremum, unit_ids, unit_pre, unit_post, input_id, output_electrode)
            
            latency_all = latency_all[latency_all['latency'] < 10]
            input_before = latency_all[latency_all['category'] == 'input']
            output_before = latency_all[latency_all['category'] == 'output']
            
            if extremum_output is not None:
                latency_extremum = get_latency_with_extremum_V2(spikes_extremum, unit_ids, unit_pre, unit_post)
                latency_extremum = latency_extremum[latency_extremum['latency'] < 10]
                output_extremum = latency_extremum[latency_extremum['category'] == 'output']
                validation = check_pair(latency_extremum, lag, exp_duration)
                
                # Store results for this specific input_id and output_electrode
                all_results.append({
                    'input_id': input_id,
                    'output_electrode': output_electrode,
                    'latency_extremum': latency_extremum,
                    'validation': validation
                })
    
    return all_results


def run_for_all_files_latency_calculation(save_path, filename, data_te,exp_duration,unit_ids,spikes,spikes_extremum, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all):
    data_te['validated_results'] = []
    for idx in range(len(pre_extremum_all)):
        pre_extremum = pre_extremum_all[idx]
        post_extremum = post_extremum_all[idx]
        unit_pre = unit_pre_all[idx]
        unit_post = unit_post_all[idx]
        lag = lag_all[idx]
        unit_to_el = data_te['UNIT_TO_EL']
        results = calculate_latency(lag, exp_duration, unit_to_el, pre_extremum, post_extremum, unit_pre, unit_post, unit_ids, spikes, spikes_extremum, extremum_output=True)

        #for result in results:
        mask = (data_te['mTE']['target'].astype(int) == data_te['UNIT_IDS'].index(unit_post)) & \
    (data_te['mTE']['source'].astype(int) == data_te['UNIT_IDS'].index(unit_pre))
        validated_result = {
            'source_electrode': pre_extremum[0],
            'target_electrode': post_extremum[0],
            'source_unit_id': unit_pre,
            'target_unit_id': unit_post,
            'lag': lag,
            'latency_extremum': results[0]['latency_extremum'],
            'validation': results[0]['validation'],
            'mTE':  data_te['mTE']['te'][mask][0],
            'local_mTE': data_te['LOCAL_mTE'][data_te['UNIT_IDS'].index(unit_post)][data_te['UNIT_IDS'].index(unit_pre)]
        }
        if 'validated_results' not in data_te:
            data_te['validated_results'] = []
        data_te['validated_results'].append(validated_result)
            


        
    print('this happens')    
    # Save the data_te into a pickle file
    with open(os.path.join(save_path, f"{filename[:-3]}_processed_info_metrics.pkl"), 'wb') as f:
        pickle.dump(data_te, f)
        print(f"Saved extended processed info metrics for {filename}")

def run_for_all_files_latency_calculation_split(pickle_file, data_te,exp_duration,unit_ids,spikes,spikes_extremum, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all):
    data_te['validated_results'] = []
    for idx in range(len(pre_extremum_all)):
        pre_extremum = pre_extremum_all[idx]
        post_extremum = post_extremum_all[idx]
        unit_pre = unit_pre_all[idx]
        unit_post = unit_post_all[idx]
        lag = lag_all[idx]
        unit_to_el = data_te['UNIT_TO_EL']
        results = calculate_latency(lag, exp_duration, unit_to_el, pre_extremum, post_extremum, unit_pre, unit_post, unit_ids, spikes, spikes_extremum, extremum_output=True)

        #for result in results:
        mask = (data_te['mTE']['target'].astype(int) == unit_ids.index(unit_post)) & \
    (data_te['mTE']['source'].astype(int) == unit_ids.index(unit_pre))
        validated_result = {
            'source_electrode': pre_extremum[0],
            'target_electrode': post_extremum[0],
            'source_unit_id': unit_pre,
            'target_unit_id': unit_post,
            'lag': lag,
            'latency_extremum': results[0]['latency_extremum'],
            'validation': results[0]['validation'],
            'mTE':  data_te['mTE']['te'][mask][0],
            'local_mTE': data_te['LOCAL_mTE'][unit_ids.index(unit_post)][unit_ids.index(unit_pre)]
        }
        if 'validated_results' not in data_te:
            data_te['validated_results'] = []
        data_te['validated_results'].append(validated_result)
            


        
    print('this happens')    
    # Save the data_te into a pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(data_te, f)
        print(f"Saved extended processed info metrics for {pickle_file}")


def run_for_all_files_latency_plot(save_path, filename, data_te,exp_duration,unit_ids,spikes,spikes_extremum, pre_extremum_all, post_extremum_all, unit_pre_all, unit_post_all, lag_all):

    for idx in range(len(pre_extremum_all)):
        pre_extremum = pre_extremum_all[idx]
        post_extremum = post_extremum_all[idx]
        unit_pre = unit_pre_all[idx]
        unit_post = unit_post_all[idx]
        lag = lag_all[idx]
        unit_to_el = data_te['UNIT_TO_EL']
        
        lat = plot_latency_and_location_with_extremum_both_plots(save_path, filename, lag, exp_duration, unit_to_el, pre_extremum, post_extremum, unit_pre, unit_post, unit_ids, spikes, spikes_extremum, extremum_output = post_extremum)
    


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

def check_pair(latency, lag, exp_duration):
    num_input_spikes = latency[latency['category'] == 'input'].shape[0]
    num_rel_output_spikes = np.sum((latency['category'] == 'output') & (latency['latency'] >= lag - 1) & (latency['latency'] <= lag + 1))
    if num_input_spikes / exp_duration > 0.5:
        if num_rel_output_spikes / exp_duration > 0.2:
            #print('Good', num_rel_output_spikes / exp_duration)
            return 'good'
        else:
            #print('Bad',num_rel_output_spikes / exp_duration)
            return 'bad'
    else:
        return 'bad'
    
def get_latency_with_extremum_V2(spikes_extremum, unit_ids, unit_pre, unit_post):
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


def plot_latency_and_location_with_extremum_both_plots(save_path, filename, lag, exp_duration, unit_to_el, input_ids, output_ids, unit_pre, unit_post, unit_ids, spikes, spikes_extremum, extremum_output=None):
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

                if extremum_output is not None:
                    latency_extremum = get_latency_with_extremum_V2(spikes_extremum, unit_ids, unit_pre, unit_post)
                    latency_extremum = latency_extremum[latency_extremum['latency'] < 10]
                    output_extremum = latency_extremum[latency_extremum['category'] == 'output']
                    ax1.scatter(output_extremum['latency'], output_extremum['spike time'] / 1000, s=10, label='output extremum', color='green')
                    validation = check_pair(latency_extremum, lag, exp_duration)
                
                ax1.set_title(f"Source {unit_pre} to Target {unit_post}")
                handles, labels = ax1.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())
        
        # Second subplot - Electrode location plot
        ax2.invert_yaxis()
        for electrode in electrodes_pre:
            x, y = convert_elno_to_xy(electrode)
            ax2.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)])
        for electrode in electrodes_post:
            x, y = convert_elno_to_xy(electrode)
            ax2.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)])
        
        # Mark extremum points
        pre_extremum_electrode = int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_pre)]))
        post_extremum_electrode = int(np.unique(spikes_extremum['Electrode'][spikes_extremum['UnitIdx'] == unit_ids.index(unit_post)]))
        
        x, y = convert_elno_to_xy(pre_extremum_electrode)
        ax2.scatter(y, x, color='yellow', marker='x', s=100)
        
        x, y = convert_elno_to_xy(post_extremum_electrode)
        ax2.scatter(y, x, color='green', marker='x', s=100)
        
        ax2.set_xlim(0, 220)
        ax2.set_ylim(0, 120)
        ax2.set_title(f"Pair presynaptic {unit_pre} and postsynaptic {unit_post} with lag {lag}")
        
        plt.tight_layout()
        
        # Save figures based on validation
        if validation == 'good':
            save_good = os.path.join(save_path, 'good_pairs')
            if not os.path.exists(save_good):
                os.makedirs(save_good)
            plt.savefig(os.path.join(save_good, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}_lag_{lag}_combined.pdf"), format='pdf', dpi=300)
            plt.savefig(os.path.join(save_good, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}_lag_{lag}_combined.png"), format='png')
            plt.close()
        elif validation == 'bad':
            save_bad = os.path.join(save_path, 'bad_pairs')
            if not os.path.exists(save_bad):
                os.makedirs(save_bad)
            plt.savefig(os.path.join(save_bad, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}_lag_{lag}_combined.pdf"), format='pdf', dpi=300)
            plt.savefig(os.path.join(save_bad, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}_lag_{lag}_combined.png"), format='png')
            plt.close()
        elif validation is None:
            # If no validation is performed, save in the main path
            plt.savefig(os.path.join(save_path, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}_lag_{lag}_combined.pdf"), format='pdf', dpi=300)
            plt.savefig(os.path.join(save_path, f"{filename[:-3]}_pre_{unit_pre}_post_{unit_post}_lag_{lag}_combined.png"), format='png')
            plt.close()

    return latency_extremum


def plot_color_coded_electrodes(save_path, unit_pre, unit_post, electrodes_pre, electrodes_post, filename):
    
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
        x, y = convert_elno_to_xy(electrode)
        plt.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)])
    for electrode in electrodes_post:
        x, y = convert_elno_to_xy(electrode)
        plt.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)])
    plt.xlim(0, 220)
    plt.ylim(0, 120)
    plt.title(f"Pair presynaptic {unit_pre} and postsynaptic {unit_post}")
    plt.savefig(os.path.join(save_path, f"{filename[:-3]}_color_code_pre_{unit_pre}_post_{unit_post}.pdf"), format='pdf', dpi=300)
    plt.savefig(os.path.join(save_path, f"{filename[:-3]}_color_code_pre_{unit_pre}_post_{unit_post}.png"), format='png')
    plt.close()

    return elec_to_color




def get_latency_electrode(spikes, input_electrode_number, output_electrode_number):
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



def plot_latency_and_location_simple(save_path, filename, spikes, input_ids, output_ids):
    include_input_in_latency = True

    for idx, input_id in enumerate(input_ids):
        if include_input_in_latency:
            output_ids = list(set(output_ids + [id for id in input_ids if id != input_id]))

        input_electrode = np.array([input_id], dtype=float)
        input_color = 'blue'
        input_spikes = spikes[spikes[:, 0] == input_electrode]
        output_electrodes = spikes[np.isin(spikes[:, 0], output_ids)]
        output_electrodes_filtered = np.unique(output_electrodes[:, 0])

        print(f'Plotting latencies for input {input_electrode}')

        fig1, ax1 = plt.subplots()
        ax1.set_ylabel("Experiment Time (s)")
        ax1.set_xlim((0., 10.))
        ax1.set_xlabel("Latency (ms)")
        ax1.scatter(input_spikes[:, 1], input_spikes[:, 2] / 1000, s=7, label='input', color=input_color)

        c=0
        # Plotting
        for output_electrode in output_electrodes_filtered:
            # Get the latency
            output_color = 'pink'
            latency = get_latency_electrode(spikes, input_electrode, output_electrode)

            # Separate input and output spikes
            input_before = latency[latency['category'] == 'input']
            output_before = latency[latency['category'] == 'output']

            # Plot the input spikes on the first iteration
            
            # Plot the output spikes
            ax1.scatter(output_before['latency'], output_before['spike time'] / 1000, s=7, label='output', color=output_color)

            if c == 0:
                ax1.scatter(input_before['latency'], input_before['spike time'] / 1000, s=7, color=input_color)
                c+=1
            plt.title(f"{filename[:-30]} input {input_id} and output {output_electrode}")
            plt.legend()
            plt.savefig(os.path.join(save_path, f"{filename[:-3]}_input_el_{input_id}_output_el_{int(output_electrode)}_STTRP.pdf"), format='pdf')
            plt.savefig(os.path.join(save_path, f"{filename[:-3]}_input_el_{input_id}_output_el_{int(output_electrode)}_STTRP.png"), format='png')
            #plt.show()

            plt.close()

        fig2, ax2 = plt.subplots()
        ax2.set_title(f"Color code for input electrode {input_id} output electrode {output_electrode}")
        ax2.invert_yaxis()

        for electrode in output_ids:
            x, y = convert_elno_to_xy(electrode)
            ax2.scatter(y, x, color=output_color)

        x, y = convert_elno_to_xy(input_id)
        ax2.scatter(y, x, color=input_color)
        ax2.scatter(y, x, color='goldenrod', marker='x', s=30, label=f'Input {input_id}')

        ax2.set_xlim(0, 220)
        ax2.set_ylim(0, 120)
        ax2.legend()
        plt.savefig(os.path.join(save_path, f"{filename[:-3]}_input_el_{input_id}_output_el_{output_electrode}_color_code.pdf"), format='pdf', dpi=300)
        plt.savefig(os.path.join(save_path, f"{filename[:-3]}_input_el_{input_id}_output_el_{output_electrode}_color_code.png"), format='png')
        #plt.show()

        plt.close()

    return latency



def plot_latency_and_location_simple_both_plots(save_path, filename, unit_to_el, input_ids, output_ids, unit_pre, unit_post, spikes):
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
        #latency_extremum = None
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
                latency_all = get_latency_electrode(spikes, input_id, output_electrode)
                
                latency_all = latency_all[latency_all['latency'] < 10]

                input_before = latency_all[latency_all['category'] == 'input']
                output_before = latency_all[latency_all['category'] == 'output']
                
                ax1.scatter(output_before['latency'], output_before['spike time'] / 1000, s=7, label='output', color=output_color)
                ax1.scatter(input_before['latency'], input_before['spike time'] / 1000, s=7, label='input', color=input_color)


                ax1.set_title(f"Source {unit_pre} to Target {unit_post}")
                handles, labels = ax1.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())
        
        # Second subplot - Electrode location plot
        ax2.invert_yaxis()
        for electrode in electrodes_pre:
            x, y = convert_elno_to_xy(electrode)
            ax2.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)])
        for electrode in electrodes_post:
            x, y = convert_elno_to_xy(electrode)
            ax2.scatter(y, x, color=elec_to_color['color'][elec_to_color['electrode'].index(electrode)])
        
        # Mark extremum points
        pre_extremum_electrode = int(input_id)
        post_extremum_electrode = int(output_electrode)
        
        x, y = convert_elno_to_xy(pre_extremum_electrode)
        ax2.scatter(y, x, color='yellow', marker='x', s=100)
        
        x, y = convert_elno_to_xy(post_extremum_electrode)
        ax2.scatter(y, x, color='green', marker='x', s=100)
        
        ax2.set_xlim(0, 220)
        ax2.set_ylim(0, 120)
        ax2.set_title(f"Pair pre electrode {input_id} and post electrode {output_electrode}")
        
        plt.tight_layout()
        

        # If no validation is performed, save in the main path
        plt.savefig(os.path.join(save_path, f"{filename[:-3]}_pre_{input_id}_post_{output_electrode}_combined.pdf"), format='pdf', dpi=300)
        plt.savefig(os.path.join(save_path, f"{filename[:-3]}_pre_{input_id}_post_{output_electrode}_combined.png"), format='png')
        plt.close()

    return latency_all

def get_latency(spikes, input_electrode_number, output_electrode_number):
    # Filter spikes for input and output electrodes
    input_spikes = spikes[spikes[:, 0] == input_electrode_number]
    output_spikes = spikes[spikes[:, 0] == output_electrode_number]
    
    # Sort spikes by time
    input_spikes = input_spikes[np.argsort(input_spikes[:, 1])]
    output_spikes = output_spikes[np.argsort(output_spikes[:, 1])]

    # Preallocate latency array with an upper bound on size
    max_possible_size = len(input_spikes) + len(output_spikes)
    latency = np.zeros(max_possible_size, dtype=[('input spike', 'i4'), ('spike time', 'f4'), 
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

    # Return only the populated part of the latency array
    return latency[:index]
