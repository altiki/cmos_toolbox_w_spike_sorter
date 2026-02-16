import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import cv2

def plot_channel(voltage_map, color_map, scale_factor):
    im = plt.imshow(voltage_map,
                    cmap=color_map)  # sometimes a specific colormap breaks a voltage map, use a different one if that happen
    plt.close()
    voltage_map_rgb = np.array(im.cmap(im.norm(im.get_array()))[:, :, 0:3])
    voltage_map_rgb = cv2.resize(voltage_map_rgb, (
    scale_factor * voltage_map_rgb.shape[1], scale_factor * voltage_map_rgb.shape[0]))
    initial_map = voltage_map_rgb.copy()
    prev_map = voltage_map_rgb.copy()
    cv2.namedWindow("voltage_map")
    #cv2.setMouseCallback("voltage_map", selection_pixels)

def get_latency_filtered(spikes, input_electrode_number, output_electrode_number, spike_count_threshold, broken_electrode_threshold):
    #spikes: (electrode, time, amplitude)
    latency = np.array([(0, 0, 0, "none",0)],
                    dtype=[('input spike', 'i4'), ('spike time', 'i4'), ('latency', 'f4'), ('category', 'U6'), ('voltage', 'f4')])

    input_spikes = spikes[spikes[:,0]==input_electrode_number]
    output_spikes = spikes[spikes[:,0]==output_electrode_number]

    if input_spikes.shape[0] < spike_count_threshold or output_spikes.shape[0] < spike_count_threshold:
        return -1
    
    elif input_spikes.shape[0] > broken_electrode_threshold or output_spikes.shape[0] > broken_electrode_threshold:
        return -2
    
    filtered_spikes = np.vstack([input_spikes, output_spikes])
    sorted_spikes = filtered_spikes[filtered_spikes[:,1].argsort()]

    #latency: (input_spike, spike_latency, category, amplitude)
    spike_count = sorted_spikes.shape[0]
    input_spike_count = 0
    input_time = None

   
    for spike in range(spike_count):
        if sorted_spikes[spike,0] == input_electrode_number:
            input_time = sorted_spikes[spike,1]
            latency = np.vstack([latency, np.array([(input_spike_count, input_time, 0, "input", sorted_spikes[spike,2])], 
                dtype=[('input spike', 'i4'), ('spike time', 'i4'), ('latency', 'f4'), ('category', 'U6'), ('voltage', 'f4')])])
            input_spike_count += 1
        elif sorted_spikes[spike,0] == output_electrode_number:
            if input_time is None:
                #print("Output spike before input spike")
                continue
            output_time = sorted_spikes[spike,1]
            latency = np.vstack([latency, np.array([(input_spike_count, output_time, output_time - input_time, "output", sorted_spikes[spike,2])], 
                dtype=[('input spike', 'i4'), ('spike time', 'i4'), ('latency', 'f4'), ('category', 'U6'), ('voltage', 'f4')])])

    return latency[1:,:]

def get_latency(spikes, input_electrode_number, output_electrode_number):
    # Filter spikes for input and output electrodes
    input_spikes = spikes[spikes[:, 0] == input_electrode_number]
    output_spikes = spikes[spikes[:, 0] == output_electrode_number]
    
    # Combine and sort spikes by time
    filtered_spikes = np.vstack([input_spikes, output_spikes])
    sorted_spikes = filtered_spikes[np.argsort(filtered_spikes[:, 1])]

    # Preallocate latency array
    max_possible_size = len(input_spikes) + len(output_spikes)
    latency = np.zeros(max_possible_size, dtype=[('input spike', 'i4'), ('spike time', 'i4'), 
                                                 ('latency', 'f4'), ('category', 'U6'), ('voltage', 'f4')])

    input_spike_count = 0
    input_time = None
    index = 0

    for spike in sorted_spikes:
        if spike[0] == input_electrode_number:
            input_time = spike[1]
            latency[index] = (input_spike_count, input_time, 0, "input", spike[2])
            input_spike_count += 1
            index += 1
        elif spike[0] == output_electrode_number and input_time is not None:
            output_time = spike[1]
            latency[index] = (input_spike_count, output_time, output_time - input_time, "output", spike[2])
            index += 1

    return latency[:index]


def plot_latency(latency, input_electrode_number, output_electrode_number, chip_id, network_number, figure_path, y_axis, stim='NoStim'):

    
    plt.close()

    fig, axs = plt.subplots(1)

    # Plot latency

    # Separate input and output spikes
    input_before = latency[latency['category']=='input']
    output_before = latency[latency['category']=='output']

    # y axis shows number of input spikes
    if y_axis == 'count':
        axs.scatter(input_before['latency'], input_before['input spike'], c=input_before['voltage'], cmap="viridis", s=7, label='input')
        axs.scatter(output_before['latency'], output_before['input spike'], c=output_before['voltage'], cmap="viridis", s=7, label='output')
        axs.set_ylabel("Input Spike Count")     

    # y axis shows experiment time
    elif y_axis == 'exp_time':
        axs.scatter(input_before['latency'], input_before['spike time']/1000, c=input_before['voltage'], cmap="viridis", s=7, label='input')
        axs.scatter(output_before['latency'], output_before['spike time']/1000, c=output_before['voltage'], cmap="viridis", s=7, label='output')
        axs.set_ylabel("Experiment Time (s)")

    axs.set_xlim((-1,20))
    axs.xaxis.set_major_locator(ticker.MultipleLocator(2.5))
    axs.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axs.set_xlabel("Latency (ms)")   
    axs.set_title('Latency Before')

    # Save figure
    figure_name = f'latency_ID{chip_id}_network{network_number}_in_{input_electrode_number}_out_{output_electrode_number}_{stim}'
    plt.savefig(os.path.join(figure_path, figure_name))

    return 0
