#!/usr/bin/env python3
"""
Script to generate and save all plots for the project.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import TrafficVisualizer

def save_all_plots(data_path='data/processed/passenger_data_daily.csv',
                   output_dir='reports/figures'):
    """
    Generate and save all possible plots.
    """
    print(f"📊 Generating all plots from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Initialize visualizer
    viz = TrafficVisualizer(df)
    
    # Dictionary of plots to generate
    plots_to_generate = {
        'time_series': {
            'function': viz.plot_time_series,
            'kwargs': {'title': 'Passenger Traffic Over Time'}
        },
        'distribution': {
            'function': viz.plot_distribution,
            'kwargs': {'target_col': 'total', 'title': 'Traffic Distribution'}
        },
        'correlation': {
            'function': viz.plot_correlation_heatmap,
            'kwargs': {'title': 'Feature Correlations'}
        },
        'seasonality': {
            'function': viz.plot_seasonality,
            'kwargs': {'title': 'Seasonal Patterns'}
        }
    }
    
    # Add control point plots if column exists
    if 'control_point' in df.columns:
        plots_to_generate.update({
            'control_points': {
                'function': viz.plot_control_point_comparison,
                'kwargs': {'title': 'Control Point Comparison'}
            },
            'cp_time_series': {
                'function': viz.plot_control_point_time_series,
                'kwargs': {'title': 'Control Point Time Series'}
            },
            'traffic_heatmap': {
                'function': viz.plot_traffic_heatmap,
                'kwargs': {'title': 'Traffic Heatmap'}
            }
        })
    
    # Generate and save all plots
    saved_files = []
    
    for plot_name, plot_info in plots_to_generate.items():
        try:
            print(f"🖼️  Generating {plot_name}...")
            
            # Generate plot
            fig = plot_info['function'](**plot_info['kwargs'])
            
            # Save as PNG
            png_path = os.path.join(output_dir, f"{plot_name}_{timestamp}.png")
            fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
            saved_files.append(png_path)
            print(f"   ✅ PNG: {png_path}")
            
            # Save as PDF (higher quality for reports)
            pdf_path = os.path.join(output_dir, f"{plot_name}_{timestamp}.pdf")
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
            saved_files.append(pdf_path)
            print(f"   ✅ PDF: {pdf_path}")
            
            plt.close(fig)  # Close to free memory
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Create summary file
    summary_path = os.path.join(output_dir, f'plot_summary_{timestamp}.md')
    with open(summary_path, 'w') as f:
        f.write(f"# Plot Generation Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data source: {data_path}\n")
        f.write(f"Total plots generated: {len(saved_files)//2}\n\n")
        f.write("## Generated Files\n\n")
        for file in saved_files:
            f.write(f"- `{os.path.basename(file)}`\n")
    
    print(f"\n📋 Summary saved to: {summary_path}")
    print(f"✅ Total plots saved: {len(saved_files)} files")
    
    return saved_files

if __name__ == "__main__":
    # Run with command line arguments
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'data/processed/passenger_data_daily.csv'
    
    save_all_plots(data_path)