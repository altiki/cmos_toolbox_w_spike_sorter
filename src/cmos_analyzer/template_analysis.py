import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_template_files(template_folder):
    """
    Load all template files from the specified folder.
    
    Parameters
    ----------
    template_folder : str
        Path to the folder containing template files.
        
    Returns
    -------
    dict
        Dictionary with template types as keys and data as values.
    """
    templates = {}
    for file in os.listdir(template_folder):
        if file.endswith('.npy') and 'templates_' in file:
            template_type = file.split('_')[1].split('.')[0]
            template_path = os.path.join(template_folder, file)
            templates[template_type] = np.load(template_path)
            print(f"Loaded {template_type} with shape {templates[template_type].shape}")
    return templates

def visualize_templates(templates, output_folder):
    """
    Visualize templates for each unit.
    
    Parameters
    ----------
    templates : dict
        Dictionary containing template data.
    output_folder : str
        Folder to save visualization plots.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Visualize templates by type
    for template_type, template_data in templates.items():
        plt.figure(figsize=(12, 8))
        
        # If template is 3D (units, channels, time)
        if template_data.ndim == 3:
            n_units = template_data.shape[0]
            for i in range(min(n_units, 10)):  # Plot up to 10 units
                # Find channel with max amplitude for this unit
                channel_amplitudes = np.ptp(template_data[i], axis=1)
                max_channel = np.argmax(channel_amplitudes)
                plt.plot(template_data[i, max_channel], 
                         label=f'Unit {i}, Ch {max_channel}')
        
        # If template is 2D (could be units×time or channels×time)
        elif template_data.ndim == 2:
            if template_type == 'average':
                # Likely units × time
                n_units = template_data.shape[0]
                for i in range(min(n_units, 10)):  # Plot up to 10 units
                    plt.plot(template_data[i], label=f'Unit {i}')
            else:
                # Plot each row
                for i in range(min(template_data.shape[0], 10)):
                    plt.plot(template_data[i], label=f'Series {i}')
        
        # If template is 1D (single waveform)
        elif template_data.ndim == 1:
            plt.plot(template_data, label='Template')
        
        plt.title(f'Template: {template_type}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"template_{template_type}.png"))
        plt.close()

def analyze_percentile_templates(templates, output_folder):
    """
    Analyze percentile templates to compare different spike waveform characteristics.
    
    Parameters
    ----------
    templates : dict
        Dictionary containing template data.
    output_folder : str
        Folder to save analysis plots.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if percentile templates exist
    percentile_keys = [k for k in templates.keys() if 'percentile' in k]
    if not percentile_keys:
        print("No percentile templates found.")
        return
    
    # Plot percentile templates for comparison
    plt.figure(figsize=(12, 8))
    
    for key in percentile_keys:
        template_data = templates[key]
        
        # If template is 3D, take the first unit and the max channel
        if template_data.ndim == 3:
            channel_amplitudes = np.ptp(template_data[0], axis=1)
            max_channel = np.argmax(channel_amplitudes)
            plt.plot(template_data[0, max_channel], label=f'{key}')
        
        # If template is 2D, take the first row
        elif template_data.ndim == 2:
            plt.plot(template_data[0], label=f'{key}')
        
        # If template is 1D
        elif template_data.ndim == 1:
            plt.plot(template_data, label=f'{key}')
    
    plt.title('Comparison of Percentile Templates')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "percentile_comparison.png"))
    plt.close()

def main(template_folder, output_folder):
    """
    Main function to analyze template files.
    
    Parameters
    ----------
    template_folder : str
        Folder containing template files.
    output_folder : str
        Folder to save outputs.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Load templates
    print(f"Loading templates from {template_folder}...")
    templates = load_template_files(template_folder)
    
    if not templates:
        print("No template files found.")
        return
    
    # Visualize templates
    print("Visualizing templates...")
    visualize_templates(templates, os.path.join(output_folder, "template_plots"))
    
    # Analyze percentile templates
    print("Analyzing percentile templates...")
    analyze_percentile_templates(templates, os.path.join(output_folder, "percentile_analysis"))
    
    print("Template analysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze template files from spikeinterface.')
    parser.add_argument('--templates', type=str, required=True, 
                        help='Path to the folder containing template files.')
    parser.add_argument('--output', type=str, required=True, 
                        help='Path to save output visualizations.')
    
    args = parser.parse_args()
    main(args.templates, args.output)