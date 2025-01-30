import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import json, os

def plot_confidence_boxplot(df: pl.DataFrame, save_path: str):
    # Prepare data for pLDDT plotting
    df = df.drop_nans()
    plddt_data = pl.DataFrame({
        'Score': pl.concat([df['pLDDT']]),
        'Metric': ['pLDDT'] * (len(df))
    })
    
    # Prepare data for pTM plotting
    ptm_data = pl.DataFrame({
        'Score': pl.concat([df['pTM']]),
        'Metric': ['pTM'] * (len(df))
    })
    
    # Prepare data for ipTM plotting
    iptm_data = pl.DataFrame({
        'Score': pl.concat([df['ipTM']]),
        'Metric': ['ipTM'] * (len(df))
    })
    plot_data = pl.concat([plddt_data, ptm_data, iptm_data])
    
    # Set style
    colors = ['#2ecc71']  # Green for AF2, Red for AF3
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create box plot
    bp = sns.boxplot(data=plot_data,
                    x='Metric',
                    y='Score',
                    palette=colors,
                    width=0.7,
                    linewidth=2)
    
    # Add individual points with jitter
    sns.stripplot(data=plot_data,
                 x='Metric',
                 y='Score',
                 dodge=True,
                 size=4,
                 alpha=0.3,
                 palette=colors,
                 jitter=0.2)
    
    # Customize the plot
    plt.title('Confidence Metrics',
             pad=20,
             fontsize=16,
             fontweight='bold')
    
    plt.xlabel('Confidence Metric', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    
    # Set y-axis limits from 0 to 100
    plt.ylim(0, 1)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # Add statistics as text
    stats_text = (
        f"Median Scores:\n"
        f"pLDDT:\n"
        f"{df['pLDDT'].median():.2f}\n"
        f"pTM:\n"
        f"{df['pTM'].median():.2f}\n"
        f"ipTM:\n"
        f"{df['ipTM'].median():.2f}\n\n"
        f"Mean Scores:\n"
        f"pLDDT:\n"
        f"{df['pLDDT'].mean():.2f} ± {df['pLDDT'].std():.2f}\n"
        f"pTM:\n"
        f"{df['pTM'].mean():.2f} ± {df['pTM'].std():.2f}\n"
        f"ipTM:\n"
        f"{df['ipTM'].median():.2f} ± {df['ipTM'].std():.2f}\n\n"
    )
    
    plt.text(1.15, 0.95, stats_text,
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10,
             verticalalignment='top')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def collect_statistics(bait_list, prey_list, complex_dir):
    # Initialize empty lists to store results
    results = []
    for bait in bait_list:
        for prey in prey_list:
            name = f"{bait}-{prey}"
            result = {
            'bait': bait,
            "prey": prey,
            'pTM': np.nan,
            'pLDDT': np.nan,
            'ipTM': np.nan
            }
            # Find matching folder
            matching_folders = [d for d in os.listdir(complex_dir) 
                            if d.startswith(name)]
            
            if matching_folders:
                folder = matching_folders[0]
                json_file = os.path.join(complex_dir, folder, 
                                    f"{folder}_summary_confidences.json")
                
                if os.path.exists(json_file):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            result['pTM'] = data['ptm']
                            result['ipTM'] = data['iptm']
                    except:
                        pass
                
                atom_met_file = os.path.join(complex_dir, folder, 
                                        f"{folder}_confidences.json")
                if os.path.exists(atom_met_file):
                    try:
                        with open(atom_met_file, 'r') as f:
                            data = json.load(f)
                            atom_plddts = data['atom_plddts']
                            average_plddt = sum(atom_plddts) / (len(atom_plddts) * 100)
                            result['pLDDT'] = average_plddt
                    except:
                        pass
            results.append(result)
    df = pl.DataFrame(results)
    return df