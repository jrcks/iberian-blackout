#!/usr/bin/env python3
"""
ASN Change Analyzer
Analyzes changes in RTT, Hop Count and Active Host Count between April 28th and 29th, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import glob
import numpy as np
from collections import defaultdict
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from config import Config

# Set style for better looking plots
plt.style.use(Config.PLOT_STYLE)
sns.set_palette("husl")

class ASNChangeAnalyzer:
    def __init__(self, data_dir=None, output_dir=None):
        self.data_dir = data_dir or str(Config.FILTERED_DATA_DIR)
        self.output_dir = Path(output_dir) if output_dir else (Config.PLOTS_DIR / 'asn_plots')

        # Target dates for comparison
        self.before_date = Config.BEFORE_BLACKOUT_DATE
        self.after_date = Config.DURING_BLACKOUT_DATE

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data_for_date(self, target_date, measurement_type):
        """Loads ASN data for a specific date and measurement type"""
        print(f"Loading {measurement_type} data from {target_date.strftime('%B %d, %Y')}...")

        # Dictionary to combine data by ASN
        asn_combined_data = defaultdict(list)
        asn_metadata = {}

        # Find the specific file for target date and measurement type
        target_date_str = target_date.strftime("%Y_%m_%d")
        filename = f"filtered_{measurement_type}_{target_date_str}.csv"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: File {filename} not found")
            return {}

        print(f"Processing {filename}...")

        try:
            df = pd.read_csv(filepath)

            # Group by ASN and collect data
            for asn, group in df.groupby('asn'):
                # Store metadata (from first occurrence)
                if asn not in asn_metadata:
                    first_entry = group.iloc[0]
                    asn_metadata[asn] = {
                        'country': first_entry['country'],
                        'operator': first_entry['operator']
                    }
                # Add all measurements for this ASN
                asn_combined_data[asn].extend(group[['rtt', 'hops', 'target', 'receiver', 'sender']].values.tolist())

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}

        # Calculate statistics for each ASN
        daily_data = {}
        for asn, data_list in asn_combined_data.items():
            if not data_list:
                continue

            # Convert to DataFrame for easier processing
            combined_df = pd.DataFrame(data_list, columns=['rtt', 'hops', 'target', 'receiver', 'sender'])

            # Remove duplicates - keep only first occurrence of each receiver-sender-target combination
            unique_df = combined_df.drop_duplicates(subset=['receiver', 'sender', 'target'], keep='first')

            # Calculate statistics using only unique hosts (first occurrence)
            daily_data[asn] = {
                'asn': asn,
                'country': asn_metadata[asn]['country'],
                'operator': asn_metadata[asn]['operator'],
                'rtt_mean': round(unique_df['rtt'].mean(), 3),
                'rtt_count': len(combined_df),  # Total measurements (including duplicates)
                'hops_mean': round(unique_df['hops'].mean(), 3),
                'host_count': len(unique_df),  # Number of unique hosts
                'rtt_median': round(unique_df['rtt'].median(), 3),
                'rtt_std': round(unique_df['rtt'].std(), 3)
            }

        print(f"Data for {len(daily_data)} ASNs from {target_date.strftime('%B %d, %Y')} loaded")
        return daily_data

    def calculate_changes(self, measurement_type):
        """Calculates changes between the two days for a specific measurement type"""
        print(f"\n=== Calculating {measurement_type} Changes: April 28 → April 29 ===")

        # Load data for both dates
        before_data = self.load_data_for_date(self.before_date, measurement_type)
        after_data = self.load_data_for_date(self.after_date, measurement_type)

        if not before_data or not after_data:
            print(f"No data available for {measurement_type}")
            return []

        # Find ASNs that exist in both datasets
        common_asns = set(before_data.keys()) & set(after_data.keys())
        print(f"Found: {len(common_asns)} ASNs in both datasets")

        change_data = []

        # Calculate changes for each ASN
        for asn in common_asns:
            before = before_data[asn]
            after = after_data[asn]

            # Calculate changes
            rtt_change = after['rtt_mean'] - before['rtt_mean']
            hops_change = after['hops_mean'] - before['hops_mean']
            host_change = after['host_count'] - before['host_count']

            # Calculate percentage changes (avoid division by zero)
            rtt_pct_change = (rtt_change / before['rtt_mean'] * 100) if before['rtt_mean'] > 0 else 0
            hops_pct_change = (hops_change / before['hops_mean'] * 100) if before['hops_mean'] > 0 else 0
            host_pct_change = (host_change / before['host_count'] * 100) if before['host_count'] > 0 else 0

            change_record = {
                'asn': asn,
                'country': before['country'],
                'operator': before['operator'],
                'rtt_before': before['rtt_mean'],
                'rtt_after': after['rtt_mean'],
                'rtt_change': round(rtt_change, 3),
                'rtt_pct_change': round(rtt_pct_change, 1),
                'hops_before': before['hops_mean'],
                'hops_after': after['hops_mean'],
                'hops_change': round(hops_change, 3),
                'hops_pct_change': round(hops_pct_change, 1),
                'host_before': before['host_count'],
                'host_after': after['host_count'],
                'host_change': host_change,
                'host_pct_change': round(host_pct_change, 1)
            }

            change_data.append(change_record)

        print(f"Changes calculated for {len(change_data)} ASNs")
        return change_data

    def get_measurement_title(self, measurement_type):
        """Returns the appropriate title for the measurement type"""
        titles = {
            'manycast_v4': 'IPv4 Anycast',
            'manycast_v6': 'IPv6 Anycast',
            'unicast_v4': 'IPv4 Unicast',
            'unicast_v6': 'IPv6 Unicast'
        }
        return titles.get(measurement_type, measurement_type)

    def create_change_plots(self, change_data, measurement_type):
        """Creates plots of the biggest changes for a specific measurement type"""
        if not change_data:
            print(f"No data available for {measurement_type} change plots!")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(change_data)

        # Get appropriate title
        measurement_title = self.get_measurement_title(measurement_type)

        # Create figure with subplots - custom grid layout
        fig = plt.figure(figsize=(40, 24))
        fig.suptitle('AS Changes: April 28 → April 29',
                    fontsize=36, fontweight='bold', y=0.82)

        # Add subtitle for measurement type - larger font
        fig.text(0.5, 0.79, measurement_title,
                 ha='center', va='center', fontsize=28, style='italic')

        # Create subplot layout: 2 very wide plots on top, 1 centered below (wider)
        ax1 = plt.subplot(2, 6, (1, 3))  # Top left (spans 3 columns)
        ax2 = plt.subplot(2, 6, (4, 6))  # Top right (spans 3 columns)
        ax3 = plt.subplot(2, 6, (8, 10))  # Bottom center (spans 3 columns, centered)

        # 1. RTT Changes (Top 5 increases)
        top_5_rtt_increase = df.nlargest(5, 'rtt_change')
        y_positions = range(len(top_5_rtt_increase))[::-1]

        # Use same blue color as snapshot plotter for RTT
        bars1 = ax1.barh(y_positions, top_5_rtt_increase['rtt_change'],
                        color='#1f77b4', alpha=0.7, height=0.6)

        # Set x-axis limit with 20% extra space
        max_rtt = top_5_rtt_increase['rtt_change'].max()
        ax1.set_xlim(0, max_rtt * 1.2)

        ax1.set_yticks(y_positions)
        ax1.set_yticklabels([f"$\\bf{{AS{row['asn']}}}$\n{row['operator'][:30]}..." if len(row['operator']) > 30
                           else f"$\\bf{{AS{row['asn']}}}$\n{row['operator']}" for _, row in top_5_rtt_increase.iterrows()],
                          fontsize=20)
        ax1.set_xlabel('RTT Change (ms)', fontsize=24, labelpad=15)
        ax1.set_title('Top 5 RTT Increases', fontsize=32, fontweight='bold', pad=30)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax1.tick_params(axis='x', labelsize=24)        # Add values on bars with percentage - right of the bars
        for i, (bar, (_, row)) in enumerate(zip(bars1, top_5_rtt_increase.iterrows())):
            width = bar.get_width()
            # Position text to the right of the bar
            x_pos = width + (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.02  # Small offset from bar end
            ax1.text(x_pos,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:+.1f}ms\n({row["rtt_pct_change"]:+.1f}%)',
                    ha='left',
                    va='center', fontsize=22, fontweight='bold', color='black')

        # 2. Hop Count Changes (Top 5 increases)
        top_5_hops_increase = df.nlargest(5, 'hops_change')
        y_positions = range(len(top_5_hops_increase))[::-1]

        # Use same orange color as snapshot plotter for Hops
        bars2 = ax2.barh(y_positions, top_5_hops_increase['hops_change'],
                        color='#ff7f0e', alpha=0.7, height=0.6)

        # Set x-axis limit with 20% extra space
        max_hops = top_5_hops_increase['hops_change'].max()
        ax2.set_xlim(0, max_hops * 1.2)

        ax2.set_yticks(y_positions)
        ax2.set_yticklabels([f"$\\bf{{AS{row['asn']}}}$\n{row['operator'][:30]}..." if len(row['operator']) > 30
                           else f"$\\bf{{AS{row['asn']}}}$\n{row['operator']}" for _, row in top_5_hops_increase.iterrows()],
                          fontsize=20)
        ax2.set_xlabel('Hop Count Change', fontsize=24, labelpad=15)
        ax2.set_title('Top 5 Hop Count Increases', fontsize=32, fontweight='bold', pad=30)
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax2.tick_params(axis='x', labelsize=24)        # Add values on bars with percentage - right of the bars
        for i, (bar, (_, row)) in enumerate(zip(bars2, top_5_hops_increase.iterrows())):
            width = bar.get_width()
            # Position text to the right of the bar
            x_pos = width + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.02  # Small offset from bar end
            ax2.text(x_pos,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:+.1f}\n({row["hops_pct_change"]:+.1f}%)',
                    ha='left',
                    va='center', fontsize=22, fontweight='bold', color='black')        # 3. Host Count Changes (Top 5 decreases - negative changes, but mirrored for consistent direction)
        top_5_hosts_decrease = df.nsmallest(5, 'host_change')
        y_positions = range(len(top_5_hosts_decrease))[::-1]

        # Mirror the negative values to positive for consistent bar direction
        mirrored_host_changes = -top_5_hosts_decrease['host_change']  # Convert negative to positive

        # Use same green color as snapshot plotter for Host Count
        bars3 = ax3.barh(y_positions, mirrored_host_changes,
                        color='#2ca02c', alpha=0.7, height=0.6)

        # Set x-axis limit with 20% extra space
        max_hosts = mirrored_host_changes.max()
        ax3.set_xlim(0, max_hosts * 1.2)

        ax3.set_yticks(y_positions)
        ax3.set_yticklabels([f"$\\bf{{AS{row['asn']}}}$\n{row['operator'][:30]}..." if len(row['operator']) > 30
                           else f"$\\bf{{AS{row['asn']}}}$\n{row['operator']}" for _, row in top_5_hosts_decrease.iterrows()],
                          fontsize=20)
        ax3.set_xlabel('Active Host Decrease', fontsize=24, labelpad=15)
        ax3.set_title('Top 5 Host Count Decreases', fontsize=32, fontweight='bold', pad=30)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax3.tick_params(axis='x', labelsize=24)        # Add values on bars with percentage - right of the bars, now all bars go right
        for i, (bar, (_, row)) in enumerate(zip(bars3, top_5_hosts_decrease.iterrows())):
            width = bar.get_width()  # This is now positive (mirrored)
            original_change = row['host_change']  # Keep original negative value for display
            # Position text to the right of the bar
            x_pos = width + (ax3.get_xlim()[1] - ax3.get_xlim()[0]) * 0.02  # Small offset from bar end
            ax3.text(x_pos,
                    bar.get_y() + bar.get_height()/2,
                    f'{int(original_change):+d}\n({row["host_pct_change"]:+.1f}%)',
                    ha='left',
                    va='center', fontsize=22, fontweight='bold', color='black')

        # Adjust layout with more space for y-labels - increased spacing for larger fonts and new layout
        plt.tight_layout(rect=[0, 0, 1, 0.88], pad=10.0, h_pad=8.0, w_pad=6.0)

        # Save plot with measurement type in filename
        filename = f"asn_changes_28_to_29_april_{measurement_type}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=800, bbox_inches='tight', pad_inches=0.3)
        plt.close()

        return filepath

    def print_summary_statistics(self, change_data, measurement_type):
        """Prints summary statistics for a measurement type"""
        if not change_data:
            return

        df = pd.DataFrame(change_data)
        measurement_title = self.get_measurement_title(measurement_type)

        print(f"\n=== {measurement_title} Analysis Results ===")
        print(f"Number of compared ASNs: {len(df)}")
        print(f"\nRTT Changes:")
        print(f"  Average change: {df['rtt_change'].mean():.2f} ms")
        print(f"  Median change: {df['rtt_change'].median():.2f} ms")
        print(f"  Largest increase: {df['rtt_change'].max():.2f} ms (AS{df.loc[df['rtt_change'].idxmax(), 'asn']})")
        print(f"  Largest decrease: {df['rtt_change'].min():.2f} ms (AS{df.loc[df['rtt_change'].idxmin(), 'asn']})")

        print(f"\nHop Count Changes:")
        print(f"  Average change: {df['hops_change'].mean():.2f}")
        print(f"  Median change: {df['hops_change'].median():.2f}")
        print(f"  Largest increase: {df['hops_change'].max():.2f} (AS{df.loc[df['hops_change'].idxmax(), 'asn']})")
        print(f"  Largest decrease: {df['hops_change'].min():.2f} (AS{df.loc[df['hops_change'].idxmin(), 'asn']})")

        print(f"\nActive Host Changes:")
        print(f"  Average change: {df['host_change'].mean():.1f}")
        print(f"  Median change: {df['host_change'].median():.1f}")
        print(f"  Largest increase: {df['host_change'].max():.0f} (AS{df.loc[df['host_change'].idxmax(), 'asn']})")
        print(f"  Largest decrease: {df['host_change'].min():.0f} (AS{df.loc[df['host_change'].idxmin(), 'asn']})")

        # ASNs with significant changes
        print(f"\nASNs with significant RTT increases (>50ms):")
        significant_rtt = df[df['rtt_change'] > 50].sort_values('rtt_change', ascending=False)
        for _, row in significant_rtt.head(10).iterrows():
            print(f"  AS{row['asn']:5d} {row['operator'][:30]:30s} +{row['rtt_change']:6.1f}ms ({row['rtt_pct_change']:+5.1f}%)")

    def run_analysis(self):
        """Runs the complete change analysis for all measurement types"""
        print("=== ASN Change Analysis: April 28 → April 29, 2025 ===")

        # Define all measurement types to analyze
        measurement_types = ["manycast_v4", "manycast_v6", "unicast_v4", "unicast_v6"]

        all_results = {}

        for measurement_type in measurement_types:
            print(f"\n{'='*60}")
            print(f"Processing {measurement_type}")
            print(f"{'='*60}")

            # Calculate changes for this measurement type
            change_data = self.calculate_changes(measurement_type)

            if not change_data:
                print(f"No data available for {measurement_type} - skipping!")
                continue

            all_results[measurement_type] = change_data

            # Create change plot for this measurement type
            print(f"\nCreating change plots for {measurement_type}...")
            change_plot = self.create_change_plots(change_data, measurement_type)

            # Print summary statistics
            self.print_summary_statistics(change_data, measurement_type)

            if change_plot:
                print(f"\nChange Plot: {change_plot}")

        print(f"\n{'='*60}")
        print("Analysis completed for all measurement types")
        print(f"{'='*60}")


def main():
    """Main function"""
    # Create necessary directories
    Config.create_directories()

    analyzer = ASNChangeAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
