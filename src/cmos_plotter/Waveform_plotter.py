import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.multicomp as mc

def extract_metadata_from_filename(filename):
    """
    Extract metadata from filename using regex
    Expected format: ID1103_N10_DIV17_DATE20240419_0915_spontaneous_CTRL
    """
    chip_id = filename.split('_')[0].replace('ID', '')
    div = filename.split('_')[2].replace('DIV', '')
    cell_type = filename.split('_')[-1].split(".")[0] # remove file extension
    network_id = filename.split('_')[1].replace('N', '')
    if cell_type == 'CTRl':
        cell_type = 'CTRL'
    print(chip_id, div, cell_type, network_id)
  
    return {
        'chip_id': int(chip_id),
        'div': int(div),
        'cell_type': cell_type,
        'network_id': int(network_id)
    }

def load_and_merge_waveforms(filepath,filename):
    """
    Load and merge waveform data from a pickle file
    
    Parameters:
    filepath (str): Full path to the pickle file
    
    Returns:
    pandas.DataFrame: Merged waveform data
    """
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(filename)
    print(metadata)
    # Load pickle file
    with open(os.path.join(filepath,'waveforms.pkl'), 'rb') as f:
        waveform_dict = pickle.load(f)
    
    # Check if the dictionary is empty
    if not waveform_dict:
        print(f"Skipping empty pickle file: {filename}")
        return None
    waveform_dict = pd.DataFrame(waveform_dict)
    # Process each unit in the pickle file
    all_waveforms = []
    for i in range(len(waveform_dict)):
        unit_id = waveform_dict['unit_ids'][i]
        for j in range(len(waveform_dict['waveforms_best_channel'][i])):
            waveform = waveform_dict['waveforms_best_channel'][i][j]
            waveform = np.array(waveform)
            # Create a DataFrame for the current unit
            df = pd.DataFrame({
                'unit_id': [unit_id],
                'waveform': [waveform],
                'chip_id': [metadata['chip_id']],
                'div': [metadata['div']],
                'cell_type': [metadata['cell_type']],
                'area': [metadata['network_id']]
            })
            all_waveforms.append(df)
    
    return pd.concat(all_waveforms, ignore_index=True)



def load_and_process_waveform_metrics(filepath, filename):
    """
    Load a specific pickle file and create a comprehensive DataFrame
    
    Parameters:
    filepath (str): Full path to the pickle file
    
    Returns:
    pandas.DataFrame or None: Processed metrics with metadata, or None if no data
    """
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(filename)
    if not metadata:
        print(f"Unable to extract metadata from {filename}")
        return None
    
    # Load pickle file
    with open(os.path.join(filepath,'waveform_metrics.pkl'), 'rb') as f:
        waveform_dict = pickle.load(f)
    
    #rename waveform_dict['amplitude'] to waveform_dict['amplitude uV']
    for unit_id, metrics in waveform_dict.items():
        metrics['amplitude uV'] = metrics.pop('amplitude')

    # Check if the dictionary is empty
    if not waveform_dict:
        print(f"Skipping empty pickle file: {filename}")
        return None
    
    # Process each unit in the pickle file
    all_metrics = []
    for unit_id, metrics in waveform_dict.items():
        # Combine metadata with metrics
        unit_data = {
            'unit_id': unit_id,
            **metadata,
            **{k: v for k, v in metrics.items() if k != 'template'}
        }
        
        # Store the template separately (it's large and we don't want to plot it directly)
        unit_data['template'] = metrics.get('template')
        
        all_metrics.append(unit_data)
    
    # Check if any metrics were extracted
    if not all_metrics:
        print(f"No valid metrics found in {filename}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    return df

def plot_metric_violin_plots(df, metric, div=None, figsize=(12, 6)):
    """
    Create violin plots for a specific metric, optionally filtering by DIV
    """
    plt.figure(figsize=figsize)
    
    # Filter by DIV if specified
    if div is not None:
        df = df[df['div'] == div]
    
    # Create violin plot
    sns.violinplot(x='cell_type', y=metric, data=df)
    plt.title(f'{metric} Distribution by Cell Type' + (f' (DIV {div})' if div is not None else ''))
    plt.xlabel('Cell Type')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_metrics_violin_plots_together(df, metrics_to_plot, div=None, figsize=(20, 10)):
    """
    Create a single figure with violin plots for multiple metrics
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with metrics
    metrics_to_plot (list): List of metrics to plot
    div (int, optional): Specific DIV to filter
    figsize (tuple): Size of the entire figure
    """
    # Filter by DIV if specified
    if div is not None:
        df = df[df['div'] == div]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize, squeeze=False)
    
    # Add a main title
    fig.suptitle(f'Waveform Metrics Violin Plots' + (f' (DIV {div})' if div is not None else ''), fontsize=16)
    
    # Plot each metric in a separate subplot
    for i, metric in enumerate(metrics_to_plot):
        # Select the subplot (note the [0, i] indexing due to squeeze=False)
        ax = axes[0, i]
        
        # Create violin plot
        sns.violinplot(x='cell_type', y=metric, data=df, ax=ax, palette='Set2', hue='cell_type', legend=False)
        ax.set_title(metric)
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.show()

def plot_mean_templates(df):
    """
    Plot mean waveform templates for each cell type
    """
    plt.figure(figsize=(12, 6))
    
    # Group by cell type and calculate mean template
    cell_type_templates = df.groupby('cell_type')['template'].apply(list)
    
    # Calculate mean template for each cell type
    mean_templates = {}
    for cell_type, templates in cell_type_templates.items():
        # Stack templates and calculate mean
        stacked_templates = np.stack(templates)
        mean_templates[cell_type] = np.mean(stacked_templates, axis=0)
    
    # Plot mean templates
    for cell_type, template in mean_templates.items():
        plt.plot(template, label=cell_type)
    
    plt.title('Mean Waveform Templates by Cell Type')
    plt.xlabel('Time Point')
    plt.ylabel('Amplitude uV')
    plt.legend()
    plt.tight_layout()
    plt.show()


from scipy import stats
import statsmodels.stats.multicomp as mc
import numpy as np

def perform_tukey_test(df, metric):
    """
    Perform Tukey's HSD test for a given metric
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    metric (str): Metric to analyze
    
    Returns:
    tuple: Tukey's HSD results
    """
    # Prepare data for Tukey's test
    cell_types = df['cell_type'].unique()
    
    # Perform one-way ANOVA first
    groups = [df[df['cell_type'] == cell_type][metric] for cell_type in cell_types]
    f_statistic, p_value = stats.f_oneway(*groups)
    
    # If ANOVA is significant, perform Tukey's HSD
    if p_value < 0.05:
        # Prepare data for Tukey's test
        tukey_data = df[[metric, 'cell_type']]
        tukey = mc.pairwise_tukeyhsd(endog=tukey_data[metric], 
                                     groups=tukey_data['cell_type'], 
                                     alpha=0.05)
        return tukey, True
    
    return None, False

def add_significance_annotations(ax, tukey_results, cell_types, metric_range):
    """
    Add significance annotations to the plot
    
    Parameters:
    ax (matplotlib.axes.Axes): Axes to annotate
    tukey_results (statsmodels.stats.multicomp.TukeyHSDResults): Tukey's HSD results
    cell_types (list): List of cell types in order
    metric_range (tuple): Range of the metric for positioning annotations
    """
    if tukey_results is None:
        return
    
    # Vertical position for annotations
    y_min, y_max = metric_range
    y_step = (y_max - y_min) * 0.05
    
    # Get the results table data
    results_table = tukey_results._results_table.data
    
    # Annotate significant differences
    sig_comparisons = []
    for idx, row in enumerate(results_table[1:], 1):
        try:
            # Extract group1, group2, and p-value
            # Handling different possible row formats
            if len(row) >= 6:
                group1, group2, _, p_val, _, _ = row[:6]
            else:
                continue
            
            # Check if the difference is significant
            if p_val < 0.05:
                # Determine number of stars
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                else:
                    stars = '*'
                
                sig_comparisons.append({
                    'group1': group1,
                    'group2': group2,
                    'stars': stars,
                    'p_val': p_val
                })
        except Exception as e:
            print(f"Error processing row {row}: {e}")
    
    # Plot annotations
    for i, comp in enumerate(sig_comparisons, 1):
        try:
            # Find indices of groups
            group1_idx = cell_types.index(comp['group1'])
            group2_idx = cell_types.index(comp['group2'])
            
            # Vertical position
            y_pos = y_max + i * y_step
            
            # Add annotation line and stars
            ax.plot([group1_idx, group2_idx], 
                    [y_pos, y_pos], 
                    color='black', 
                    linewidth=1)
            ax.text((group1_idx + group2_idx) / 2, 
                    y_pos, 
                    comp['stars'], 
                    ha='center', 
                    va='bottom')
        except ValueError:
            print(f"Could not find groups {comp['group1']} or {comp['group2']} in cell types")
            
def plot_metrics_violin_plots_w_test(df, metrics_to_plot, div=None, figsize=(20, 10)):
    """
    Create a single figure with violin plots for multiple metrics
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with metrics
    metrics_to_plot (list): List of metrics to plot
    div (int, optional): Specific DIV to filter
    figsize (tuple): Size of the entire figure
    """
    # Filter by DIV if specified
    if div is not None:
        df = df[df['div'] == div]
    
    # Get unique cell types in order (preserve full names)
    cell_types = df['cell_type'].unique().tolist()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize, squeeze=False)
    
    # Add a main title
    fig.suptitle(f'Waveform Metrics Violin Plots' + (f' (DIV {div})' if div is not None else ''), fontsize=16)
    
    # Plot each metric in a separate subplot
    for i, metric in enumerate(metrics_to_plot):
        # Select the subplot (note the [0, i] indexing due to squeeze=False)
        ax = axes[0, i]
        
        # Create violin plot
        sns.violinplot(x='cell_type', y=metric, data=df, ax=ax, 
                       order=cell_types, 
                       # Use first occurrence of each cell type
                       hue='cell_type', dodge=False)
        
        ax.set_title(metric)
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Value')

        ax.tick_params(axis='x', rotation=45)
        
        # Perform Tukey's test and add annotations
        tukey_results, is_significant = perform_tukey_test(df, metric)
        
        if is_significant:
            # Get current y-axis limits for positioning annotations
            y_min, y_max = ax.get_ylim()
            try:
                add_significance_annotations(ax, tukey_results, cell_types, (y_min, y_max))
                
                # Adjust y-axis to make room for annotations
                ax.set_ylim(y_min, y_max * 3.)
            except Exception as e:
                print(f"Error adding annotations for {metric}: {e}")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.show()


def plot_mean_templates_fixed(df):
    """
    Plot mean waveform templates for each cell type
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing template data
    """
    plt.figure(figsize=(12, 6))
    
    # Group by cell type and calculate mean template
    cell_types = df['cell_type'].unique()
    
    # Ensure templates are numeric numpy arrays
    def safe_template_conversion(template):
        """
        Convert template to numeric numpy array
        """
        # If already a numpy array, convert to float
        if isinstance(template, np.ndarray):
            return template.astype(np.float32)
        
        # If it's a list, convert to numpy array
        if isinstance(template, list):
            return np.array(template, dtype=np.float32)
        
        # If it's a pandas Series, convert to numpy array
        if hasattr(template, 'to_numpy'):
            return template.to_numpy(dtype=np.float32)
        
        # Last resort conversion
        try:
            return np.array(template, dtype=np.float32)
        except Exception as e:
            print(f"Could not convert template: {e}")
            return None
    
    # Collect and process templates
    mean_templates = {}
    for cell_type in cell_types:
        # Filter templates for this cell type
        cell_templates = df[df['cell_type'] == cell_type]['template']
        
        # Convert and filter out None values
        valid_templates = [
            safe_template_conversion(template) 
            for template in cell_templates 
            if safe_template_conversion(template) is not None
        ]
        
        # Ensure we have valid templates
        if not valid_templates:
            print(f"No valid templates found for {cell_type}")
            continue
        
        # Stack and calculate mean
        try:
            stacked_templates = np.stack(valid_templates)
            mean_templates[cell_type] = np.mean(stacked_templates, axis=0)
        except Exception as e:
            print(f"Error processing templates for {cell_type}: {e}")
            continue
    
    # Plot mean templates
    for cell_type, template in mean_templates.items():
        plt.plot(template, label=cell_type)
    
    plt.title('Mean Waveform Templates by Cell Type')
    plt.xlabel('Time Point')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_metrics_over_weeks(df, metrics_to_plot):
    """
    Plot metrics over weeks in culture with standard deviation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'cell_type', 'div', and metrics columns
    metrics_to_plot : list
        List of metric column names to plot
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Map DIV to weeks
    def map_div_to_week(div):
        if 0 <= div <= 6:
            return "Week 1"
        elif 7 <= div <= 13:
            return "Week 2"
        elif 14 <= div <= 20:
            return "Week 3"
        elif 21 <= div <= 27:
            return "Week 4"
        elif 28 <= div <= 34:
            return "Week 5"
        else:
            return "Other"
    
    # Add week column
    df_copy['week'] = df_copy['div'].apply(map_div_to_week)
    
    # Filter out any 'Other' weeks if needed
    df_copy = df_copy[df_copy['week'] != 'Other']
    
    # Define the order of weeks for plotting
    week_order = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]
    
    # Get unique cell types
    cell_types = df_copy['cell_type'].unique()
    
    # Set up the figure size based on the number of metrics
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4 * len(metrics_to_plot)), sharex=True)
    
    # If there's only one metric, axes won't be an array, wrap it in a list
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    # Plotting each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # For each cell type
        for cell_type in cell_types:
            # Filter data for this cell type
            cell_data = df_copy[df_copy['cell_type'] == cell_type]
            
            # Group by week and calculate mean and std
            grouped = cell_data.groupby('week')[metric].agg(['mean', 'std']).reset_index()
            
            # Reorder according to week_order
            grouped['week_num'] = grouped['week'].apply(lambda x: week_order.index(x) if x in week_order else -1)
            grouped = grouped.sort_values('week_num')
            grouped = grouped[grouped['week_num'] != -1]  # Remove weeks not in order
            
            # Extract the data
            weeks = grouped['week']
            means = grouped['mean']
            stds = grouped['std']
            
            # Plot the line with error bars for standard deviation
            ax.errorbar(weeks, means, yerr=stds, marker='o', label=cell_type, capsize=5)
        
        # Set labels and title
        ax.set_title(f'{metric} Over Weeks in Culture')
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Set the x-axis label only on the bottom subplot
    axes[-1].set_xlabel('Weeks in Culture')
    
    # Adjust spacing
    plt.tight_layout()
    
    return fig