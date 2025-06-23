import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from datetime import datetime
import seaborn as sns
from matplotlib.ticker import FuncFormatter

def load_filtered_data():
    """
    Load all filtered data files from the local filtered_data directory.
    """
    data_dir = Path("filtered_data")
    measurement_types = ["manycast_v4", "unicast_v4", "manycast_v6", "unicast_v6"]

    all_data = []

    for measurement_type in measurement_types:
        # Find files matching the pattern: filtered_{measurement_type}_{year}_{month}_{day}.csv
        pattern = f"filtered_{measurement_type}_*.csv"
        files = glob.glob(str(data_dir / pattern))

        if not files:
            print(f"Warning: No filtered data files found for {measurement_type}")
            continue

        for file in sorted(files):
            # Extract date from filename (e.g., filtered_manycast_v4_2025_04_27.csv)
            filename = Path(file).stem
            parts = filename.split('_')
            if len(parts) >= 6:
                year = int(parts[3])
                month = int(parts[4])
                day = int(parts[5])
                date = datetime(year, month, day)

                # Load data
                df = pd.read_csv(file)

                # Select only the columns we need for plotting
                required_columns = ['receiver', 'sender', 'target', 'rtt', 'ttl', 'hops', 'country']

                # Check if all required columns exist
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"Warning: Missing columns {missing_columns} in {file}")
                    continue

                # Keep only required columns and add metadata
                df = df[required_columns].copy()
                df['date'] = date
                df['date_str'] = date.strftime('%d.%m.')
                df['measurement_type'] = measurement_type

                all_data.append(df)
                print(f"Loaded {len(df):,} records from {file} ({measurement_type})")

    if not all_data:
        raise ValueError("No valid data files found in either manycast_v4 or unicast_v4 folders")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total combined records: {len(combined_df):,}")

    return combined_df

def calculate_daily_metrics(df):
    """
    Calculate daily metrics for RTT, hops, and active hosts by country and measurement type.
    """
    # Filter data to only include dates from 25.04. to 04.05.
    start_date = datetime(2025, 4, 25)
    end_date = datetime(2025, 5, 4)
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    print(f"Filtered data to {start_date.strftime('%d.%m.')} - {end_date.strftime('%d.%m.')}: {len(df_filtered):,} records")

    daily_metrics = []

    for date in df_filtered['date'].unique():
        date_data = df_filtered[df_filtered['date'] == date]
        date_str = date_data['date_str'].iloc[0]

        for measurement_type in ['manycast_v4', 'unicast_v4', 'manycast_v6', 'unicast_v6']:
            for country in ['Spain', 'Portugal']:
                country_data = date_data[
                    (date_data['country'] == country) &
                    (date_data['measurement_type'] == measurement_type)
                ]

                if len(country_data) > 0:
                    # Remove duplicates based on key fields (receiver, sender, target) - keep first occurrence
                    initial_count = len(country_data)
                    country_data_dedup = country_data.drop_duplicates(subset=['receiver', 'sender', 'target'], keep='first')
                    duplicates_removed = initial_count - len(country_data_dedup)

                    if duplicates_removed > 0:
                        print(f"  Removed {duplicates_removed} duplicates for {country} {measurement_type} on {date_str} ({len(country_data_dedup)} remaining)")

                    metrics = {
                        'date': date,
                        'date_str': date_str,
                        'country': country,
                        'measurement_type': measurement_type,
                        'avg_rtt': country_data_dedup['rtt'].mean(),
                        'median_rtt': country_data_dedup['rtt'].median(),
                        'avg_hops': country_data_dedup['hops'].mean(),
                        'median_hops': country_data_dedup['hops'].median(),
                        'active_hosts': len(country_data_dedup['target'].unique()),
                        'total_measurements': len(country_data_dedup)
                    }
                else:
                    # No data for this country/type on this date
                    metrics = {
                        'date': date,
                        'date_str': date_str,
                        'country': country,
                        'measurement_type': measurement_type,
                        'avg_rtt': None,
                        'median_rtt': None,
                        'avg_hops': None,
                        'median_hops': None,
                        'active_hosts': 0,
                        'total_measurements': 0
                    }

                daily_metrics.append(metrics)

    return pd.DataFrame(daily_metrics)

def create_plots(daily_metrics_df):
    """
    Create combined plots with 3 y-axes for RTT, hops, and active hosts.
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Define formatter for thousand separators with dots
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{x:,.0f}'.replace(',', '.')
        else:
            return f'{x:.0f}'

    # Define formatter for hop count values (preserves decimal places)
    def hop_formatter(x, pos):
        return f'{x:.1f}'

    # Define formatter for host count values
    def host_formatter(x, pos):
        if x >= 1000:
            return f'{x:,.0f}'.replace(',', '.')
        else:
            return f'{x:.0f}'

    thousands_formatter_func = FuncFormatter(thousands_formatter)
    hop_formatter_func = FuncFormatter(hop_formatter)
    host_formatter_func = FuncFormatter(host_formatter)

    # Define event time
    event_time = datetime(2025, 4, 28, 12, 33)
    event_date_str = event_time.strftime('%d.%m.')

    # Define combinations for IPv4 and IPv6
    combinations = [
        ('Portugal', 'unicast_v4', 'Unicast IPv4 Portugal'),
        ('Spain', 'unicast_v4', 'Unicast IPv4 Spain'),
        ('Portugal', 'manycast_v4', 'Anycast IPv4 Portugal'),
        ('Spain', 'manycast_v4', 'Anycast IPv4 Spain'),
        ('Portugal', 'unicast_v6', 'Unicast IPv6 Portugal'),
        ('Spain', 'unicast_v6', 'Unicast IPv6 Spain'),
        ('Portugal', 'manycast_v6', 'Anycast IPv6 Portugal'),
        ('Spain', 'manycast_v6', 'Anycast IPv6 Spain')
    ]

    for country, measurement_type, title in combinations:
        # Filter data for this combination
        data = daily_metrics_df[
            (daily_metrics_df['country'] == country) &
            (daily_metrics_df['measurement_type'] == measurement_type)
        ].sort_values('date')

        if len(data) == 0:
            print(f"No data found for {title}")
            continue

        # Create figure
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'{country}',
                     fontsize=16, fontweight='bold', y=0.89)  # Main title with country

        # Add subtitle for measurement type
        formatted_measurement_type = measurement_type.replace("manycast", "Anycast").replace("unicast", "Unicast").replace("_v4", " IPv4").replace("_v6", " IPv6")
        fig.text(0.5, 0.83, formatted_measurement_type,
                 ha='center', va='center', fontsize=12, style='italic')  # Subtitle with formatted measurement type

        # Find position between 28.04. and 29.04.
        date_labels = data['date_str'].tolist()
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
        ax1.plot(data['date_str'], data['avg_rtt'], color='#1f77b4', marker='o', linewidth=1.5, markersize=4, label='Avg. RTT (ms)')
        ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=12)
        ax1.yaxis.set_major_formatter(thousands_formatter_func)
        ax1.set_ylim(20, 200)  # Set fixed RTT range from 20 to 100 ms
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
        ax2.plot(data['date_str'], data['active_hosts'], color='#2ca02c', marker='o', linewidth=1.5, markersize=4, label='Active Hosts')
        ax2.tick_params(axis='y', labelcolor='#2ca02c', labelsize=12)
        ax2.yaxis.set_major_formatter(host_formatter_func)
        # Set dynamic range with 5% buffer below min and 20% buffer above max
        host_min = data['active_hosts'].min()
        host_max = data['active_hosts'].max()
        host_range = host_max - host_min
        if host_range > 0:
            host_y_min = max(0, host_min - 0.05 * host_range)
            host_y_max = host_max + 0.23 * host_range
        else:
            # If all values are the same, add fixed padding
            host_y_min = max(0, host_min - 1)
            host_y_max = host_max + 3
        ax2.set_ylim(host_y_min, host_y_max)

        # Create tertiary y-axis for hops
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(data['date_str'], data['avg_hops'], color='#ff7f0e', marker='o', linewidth=1.5, markersize=4, label='Avg. Hops')
        ax3.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=12)
        ax3.yaxis.set_major_formatter(hop_formatter_func)
        # Set dynamic range with 5 hops buffer around actual values
        hop_min = data['avg_hops'].min() - 5
        hop_max = data['avg_hops'].max() + 5
        ax3.set_ylim(max(0, hop_min), hop_max)  # Ensure minimum is not below 0

        # Add legend
        lines_labels = [
            (ax1.get_lines()[0], "Avg. RTT (ms)"),
            (ax2.get_lines()[0], "Active Hosts"),
            (ax3.get_lines()[0], "Avg. Hops")
        ]
        lines, labels = zip(*lines_labels)
        ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                  ncol=1, fontsize=12, frameon=True, fancybox=True, shadow=False,
                  facecolor='white', edgecolor='black', framealpha=1.0)

        # Adjust layout
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.set_xticks(range(len(data['date_str'])))  # Show tick marks for all entries
        ax1.set_xticklabels(data['date_str'])  # Show all date labels
        fig.tight_layout(rect=[0, 0, 1, 0.88])

        # Save plot
        if 'v4' in measurement_type:
            ip_version = 'ipv4'
            mtype = measurement_type.replace('_v4', '')
        else:
            ip_version = 'ipv6'
            mtype = measurement_type.replace('_v6', '')

        filename = f"{country.lower()}_{mtype}_{ip_version}.png"
        filepath = plots_dir / filename
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")

    print(f"All plots saved to: {plots_dir}")

def print_summary_statistics(daily_metrics_df):
    """
    Print summary statistics for the analysis period by country and measurement type.
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for measurement_type in ['manycast_v4', 'unicast_v4', 'manycast_v6', 'unicast_v6']:
        # Format the measurement type name for display
        if 'v4' in measurement_type:
            display_name = measurement_type.upper().replace('_V4', ' IPv4')
        else:
            display_name = measurement_type.upper().replace('_V6', ' IPv6')

        print(f"\n{display_name} RESULTS:")
        print("-" * 40)

        for country in ['Spain', 'Portugal']:
            country_data = daily_metrics_df[
                (daily_metrics_df['country'] == country) &
                (daily_metrics_df['measurement_type'] == measurement_type)
            ]

            if len(country_data) > 0:
                print(f"\n{country}:")
                print(f"  Average RTT: {country_data['avg_rtt'].mean():.2f} ms")
                print(f"  Average Hops: {country_data['avg_hops'].mean():.2f}")
                print(f"  Average Active Hosts: {country_data['active_hosts'].mean():.0f}")
                print(f"  Total Measurements: {country_data['total_measurements'].sum():,}")

                # Find min/max values and dates
                min_rtt_idx = country_data['avg_rtt'].idxmin()
                max_rtt_idx = country_data['avg_rtt'].idxmax()

                if not pd.isna(min_rtt_idx):
                    print(f"  Lowest RTT: {country_data.loc[min_rtt_idx, 'avg_rtt']:.2f} ms on {country_data.loc[min_rtt_idx, 'date_str']}")
                if not pd.isna(max_rtt_idx):
                    print(f"  Highest RTT: {country_data.loc[max_rtt_idx, 'avg_rtt']:.2f} ms on {country_data.loc[max_rtt_idx, 'date_str']}")
            else:
                print(f"\n{country}: No data available")

def main():
    """
    Main function to run the complete analysis and plotting.
    """
    try:
        print("Loading filtered data from filtered_data directory (IPv4 and IPv6)...")
        df = load_filtered_data()

        print("Calculating daily metrics...")
        daily_metrics = calculate_daily_metrics(df)

        print("Creating plots...")
        create_plots(daily_metrics)

        print_summary_statistics(daily_metrics)

        # Save metrics to CSV for further analysis
        metrics_path = Path("analysis_results")
        metrics_path.mkdir(exist_ok=True)

        daily_metrics.to_csv(metrics_path / "daily_metrics.csv", index=False)
        print(f"\nDaily metrics saved to: {metrics_path / 'daily_metrics.csv'}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
