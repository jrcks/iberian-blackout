#!/usr/bin/env python3
"""
ASN Timeline Plotter
Creates timeline plots for RTT, Hop Count and Active Host Count for a specific ASN over multiple days
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import glob
from collections import defaultdict
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

class ASNTimelinePlotter:
    def __init__(self, data_dir="filtered_data", output_dir="plots/asn_plots"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.asn_data = defaultdict(list)
        self.asn_metadata = {}

        # Define date range for filtering
        self.start_date = datetime(2025, 4, 25)
        self.end_date = datetime(2025, 5, 4)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, target_asn):
        """Loads all relevant CSV files and extracts data for the target ASN"""
        print(f"Loading CSV files for ASN {target_asn} (April 25 - May 4)...")

        # Define all measurement types to load
        measurement_types = ["manycast_v4"]

        # Dictionary to store data by date
        daily_data = defaultdict(list)

        for measurement_type in measurement_types:
            # Find all filtered files for this measurement type
            pattern = os.path.join(self.data_dir, f"filtered_{measurement_type}_*.csv")
            csv_files = glob.glob(pattern)
            csv_files.sort()

            print(f"Gefunden: {len(csv_files)} CSV-Dateien für {measurement_type}")

            for csv_file in csv_files:
                # Extract date from filename
                filename = os.path.basename(csv_file)
                # filtered_manycast_v4_2025_04_24.csv -> 2025_04_24
                date_str = filename.replace(f"filtered_{measurement_type}_", "").replace(".csv", "")
                try:
                    date = datetime.strptime(date_str, "%Y_%m_%d")
                except ValueError:
                    print(f"Warnung: Konnte Datum aus {filename} nicht extrahieren")
                    continue

                # Filter by date range
                if date < self.start_date or date > self.end_date:
                    continue

                print(f"Verarbeite {filename} (Datum: {date.strftime('%Y-%m-%d')})")

                # Load CSV
                try:
                    df = pd.read_csv(csv_file)

                    # Filter for target ASN
                    asn_df = df[df['asn'] == target_asn]

                    if asn_df.empty:
                        continue

                    # Store metadata for the ASN (from first occurrence)
                    if target_asn not in self.asn_metadata:
                        first_entry = asn_df.iloc[0]
                        self.asn_metadata[target_asn] = {
                            'country': first_entry['country'],
                            'operator': first_entry['operator']
                        }

                    # Store all data for this date to combine later
                    # Include measurement_type to handle potential duplicates across different measurement types better
                    for _, row in asn_df.iterrows():
                        daily_data[date].append([row['rtt'], row['hops'], row['target'], row['receiver'], row['sender'], measurement_type])

                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
                    continue

        # Now calculate combined statistics for each date
        for date, data_list in daily_data.items():
            if not data_list:
                continue

            # Convert to DataFrame for easier processing
            combined_df = pd.DataFrame(data_list, columns=['rtt', 'hops', 'target', 'receiver', 'sender', 'measurement_type'])

            # Remove duplicates based on receiver, sender, target - keep first occurrence
            # This ensures we don't have duplicate measurements from the same receiver-sender-target combination
            initial_count = len(combined_df)
            combined_df_dedup = combined_df.drop_duplicates(subset=['receiver', 'sender', 'target'], keep='first')
            duplicates_removed = initial_count - len(combined_df_dedup)

            if duplicates_removed > 0:
                print(f"  Removed {duplicates_removed} duplicates for ASN {target_asn} on {date.strftime('%m/%d')} ({len(combined_df_dedup)} remaining)")

            # Calculate statistics combining all measurement types (without duplicates)
            stats = {
                'rtt_mean': combined_df_dedup['rtt'].mean(),
                'rtt_count': len(combined_df_dedup),
                'hops_mean': combined_df_dedup['hops'].mean(),
                'host_count': combined_df_dedup['target'].nunique()
            }

            # Add date and store data
            self.asn_data[target_asn].append({
                'date': date,
                'date_str': date.strftime('%m/%d'),
                'rtt_mean': round(stats['rtt_mean'], 3),
                'rtt_count': stats['rtt_count'],
                'hops_mean': round(stats['hops_mean'], 3),
                'host_count': stats['host_count']
            })

        if target_asn in self.asn_data:
            print(f"Data for ASN {target_asn} loaded over {len(self.asn_data[target_asn])} days (April 25 - May 4)")
            print(f"Combined data from all measurement types: manycast_v4, manycast_v6, unicast_v4, unicast_v6")
        else:
            print(f"No data found for ASN {target_asn} in the period April 25 - May 4!")

    def create_asn_plot(self, asn, data):
        """Creates a plot for a specific ASN"""
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(data)
        df = df.sort_values('date')

        # Get ASN metadata
        metadata = self.asn_metadata.get(asn, {'country': 'Unknown', 'operator': 'Unknown'})

        # Define formatter functions (matching plotter.py style)
        def thousands_formatter(x, pos):
            if x >= 1000:
                return f'{x:,.0f}'.replace(',', '.')
            else:
                return f'{x:.0f}'

        def hop_formatter(x, pos):
            return f'{x:.1f}'

        def host_formatter(x, pos):
            if x >= 1000:
                return f'{x:,.0f}'.replace(',', '.')
            else:
                return f'{x:.0f}'

        thousands_formatter_func = FuncFormatter(thousands_formatter)
        hop_formatter_func = FuncFormatter(hop_formatter)
        host_formatter_func = FuncFormatter(host_formatter)

        # Define event time (matching plotter.py)
        event_time = datetime(2025, 4, 28, 12, 33)
        event_date_str = event_time.strftime('%m/%d')

        # Create figure (matching plotter.py style)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'AS{asn} - {metadata["operator"]} ({metadata["country"]})',
                     fontsize=16, fontweight='bold', y=0.89)  # Further reduced y position

        # Add subtitle for measurement type
        fig.text(0.5, 0.83, 'IPv4 Anycast',
                 ha='center', va='center', fontsize=12, style='italic')  # Further reduced y position

        # Find position for event line
        date_labels = df['date_str'].tolist()
        event_position = None

        if '28.04.' in date_labels and '29.04.' in date_labels:
            idx_28 = date_labels.index('28.04.')
            idx_29 = date_labels.index('29.04.')
            event_position = idx_28 + 0.5
        elif '28.04.' in date_labels:
            idx_28 = date_labels.index('28.04.')
            event_position = idx_28 + 0.5
        elif len(date_labels) > 0:
            for i, date_str in enumerate(date_labels):
                if date_str >= '28.04.':
                    event_position = i + 0.5
                    break

        # Plot RTT on primary y-axis
        ax1.plot(df['date_str'], df['rtt_mean'], color='#1f77b4', marker='o', linewidth=1.5, markersize=4, label='Avg. RTT (ms)')
        ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=12)
        ax1.yaxis.set_major_formatter(thousands_formatter_func)
        # Set dynamic range with 250ms buffer above max value
        rtt_max = df['rtt_mean'].max() + 100
        ax1.set_ylim(0, rtt_max)
        ax1.grid(True, alpha=0.3)

        # Add vertical line for event
        if event_position is not None:
            ax1.axvline(x=event_position, color='dimgray', linestyle='--', linewidth=1.5, alpha=0.7)
            # Add "Blackout" text above the event line
            ax1.text(event_position - 0.08, ax1.get_ylim()[1] * 0.98, 'Blackout',
                    rotation=90, ha='right', va='top', fontsize=10,
                    color='dimgray', fontweight='bold')

        # Create secondary y-axis for active hosts
        ax2 = ax1.twinx()
        ax2.plot(df['date_str'], df['host_count'], color='#2ca02c', marker='o', linewidth=1.5, markersize=4, label='Active Hosts')
        ax2.tick_params(axis='y', labelcolor='#2ca02c', labelsize=12)
        ax2.yaxis.set_major_formatter(host_formatter_func)
        # Set dynamic range with 5% buffer below min and 15% buffer above max
        host_min = df['host_count'].min()
        host_max = df['host_count'].max()
        host_range = host_max - host_min
        if host_range > 0:
            host_y_min = max(0, host_min - 0.05 * host_range)
            host_y_max = host_max + 0.29 * host_range
        else:
            # If all values are the same, add fixed padding
            host_y_min = max(0, host_min - 1)
            host_y_max = host_max + 3
        ax2.set_ylim(host_y_min, host_y_max)

        # Create tertiary y-axis for hops
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(df['date_str'], df['hops_mean'], color='#ff7f0e', marker='o', linewidth=1.5, markersize=4, label='Avg. Hops')
        ax3.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=12)
        ax3.yaxis.set_major_formatter(hop_formatter_func)
        # Set dynamic range with 5 hops buffer around actual values
        hop_min = df['hops_mean'].min() - 5
        hop_max = df['hops_mean'].max() + 5
        ax3.set_ylim(max(0, hop_min), hop_max)  # Ensure minimum is not below 0

        # Add legend (matching plotter.py style)
        lines_labels = [
            (ax1.get_lines()[0], "Avg. RTT (ms)"),
            (ax2.get_lines()[0], "Active Hosts"),
            (ax3.get_lines()[0], "Avg. Hops")
        ]
        lines, labels = zip(*lines_labels)
        ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                  ncol=1, fontsize=12, frameon=True, fancybox=True, shadow=False,
                  facecolor='white', edgecolor='black', framealpha=1.0)

        # Adjust layout (matching plotter.py style)
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.set_xticks(range(len(df['date_str'])))
        ax1.set_xticklabels(df['date_str'])
        fig.tight_layout(rect=[0, 0, 1, 0.88])

        # Save plot
        filename = f"asn_{asn}_timeline.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_asn(self, target_asn):
        """Lädt Daten und erstellt Plot für eine spezifische ASN"""
        # Load data for the target ASN
        self.load_data(target_asn)

        if target_asn not in self.asn_data or not self.asn_data[target_asn]:
            print(f"Keine ausreichenden Daten für ASN {target_asn} gefunden!")
            return None

        # Create plot
        filepath = self.create_asn_plot(target_asn, self.asn_data[target_asn])

        # Print summary
        metadata = self.asn_metadata.get(target_asn, {'country': 'Unknown', 'operator': 'Unknown'})
        data = self.asn_data[target_asn]
        total_measurements = sum(entry['rtt_count'] for entry in data)

        print(f"\n=== Plot erstellt ===")
        print(f"ASN: {target_asn}")
        print(f"Operator: {metadata['operator']}")
        print(f"Land: {metadata['country']}")
        print(f"Anzahl Tage: {len(data)}")
        print(f"Gesamte Messungen: {total_measurements}")
        print(f"Plot gespeichert: {filepath}")

        return filepath


def main():
    """Hauptfunktion"""
    # HIER DIE ASN EINSTELLEN DIE GEPLOTTET WERDEN SOLL
    TARGET_ASN = 42863  # Ändere diese Zahl für eine andere ASN #3352 #42863

    print("=== ASN Timeline Plotter ===")
    print(f"Erstelle Plot für ASN {TARGET_ASN}")

    # Initialize plotter
    plotter = ASNTimelinePlotter()

    # Plot the specific ASN
    result = plotter.plot_asn(TARGET_ASN)

    if result:
        print(f"\nFertig! Plot wurde erfolgreich erstellt.")
    else:
        print(f"\nFehler beim Erstellen des Plots für ASN {TARGET_ASN}")


if __name__ == "__main__":
    main()
