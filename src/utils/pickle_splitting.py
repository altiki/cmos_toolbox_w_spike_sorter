import pickle
import numpy as np
import os
from pathlib import Path

def split_pickle_by_time(input_file, input_dir=None, output_dir=None, segment_duration=30.0):
    """
    Split a pickle file containing spike data into smaller pickle files based on time segments.
    
    Parameters:
    -----------
    input_file : str
        Path to the input pickle file
    output_dir : str, optional
        Directory to save output files. If None, uses the same directory as input_file
    segment_duration : float, optional
        Duration of each segment in seconds (default: 30.0)
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file))
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = Path(input_file).stem
    
    # Load the pickle file
    with open(os.path.join(input_dir,input_file), 'rb') as f:
        data = pickle.load(f)
    
    # Extract needed data
    experiment_duration = data['EXPERIMENT_DURATION']
    spikemat = data['SPIKEMAT']
    spikemat_extremum = data['SPIKEMAT_EXTREMUM']
    
    # Calculate number of segments
    num_segments = int(np.ceil(experiment_duration / segment_duration))
    
    # Process each segment
    for segment_idx in range(num_segments):
        # Calculate segment time boundaries in milliseconds
        start_time_ms = segment_idx * segment_duration * 1000
        end_time_ms = min((segment_idx + 1) * segment_duration * 1000, experiment_duration * 1000)
        
        # For the first segment, remove the first 1000ms
        if segment_idx == 0:
            start_time_ms += 1000
        
        # Filter SPIKEMAT data for this segment
        segment_spikemat = spikemat[
            (spikemat['Spike_Time'] >= start_time_ms) & 
            (spikemat['Spike_Time'] < end_time_ms)
        ]
        
        # Filter SPIKEMAT_EXTREMUM data for this segment
        segment_spikemat_extremum = spikemat_extremum[
            (spikemat_extremum['Spike_Time'] >= start_time_ms) & 
            (spikemat_extremum['Spike_Time'] < end_time_ms)
        ]
        
        # Prepare segment data
        segment_data = data.copy()  # Copy all fields from original data
        segment_data['SPIKEMAT'] = segment_spikemat
        segment_data['SPIKEMAT_EXTREMUM'] = segment_spikemat_extremum
        segment_data['EXPERIMENT_DURATION'] = segment_duration  # Set to 30 seconds
        
        # For the first segment, adjust experiment duration if needed
        if segment_idx == 0 and end_time_ms - start_time_ms < segment_duration * 1000:
            segment_data['EXPERIMENT_DURATION'] = (end_time_ms - start_time_ms) / 1000
        
        # Save segment data to a new pickle file
        output_file = os.path.join(output_dir, f"{base_filename}_segment_{segment_idx+1}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(segment_data, f)
        
        print(f"Saved segment {segment_idx+1}/{num_segments} to {output_file}")
        print(f"  Time range: {start_time_ms/1000:.3f}s - {end_time_ms/1000:.3f}s")
        print(f"  SPIKEMAT entries: {len(segment_spikemat)}")
        print(f"  SPIKEMAT_EXTREMUM entries: {len(segment_spikemat_extremum)}")


    