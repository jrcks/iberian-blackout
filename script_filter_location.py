import pandas as pd
import numpy as np
import ipaddress
from pathlib import Path
import gzip
import re
import os
import glob

pattern = r"ID:\s*(\d+)\s*,\s*hostname:\s*([\w-]+)"
pd.set_option('display.max_rows', None)

def read_gzipped_comment_lines(filepath):
    """
    Reads initial comment lines (starting with '#') from a gzipped file.
    """
    comment_lines = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File not found at '{filepath}'")

    try:
        with gzip.open(filepath, mode='rt', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    comment_lines.append(line.rstrip())
                else:
                    # found the first non-comment line, stop reading
                    break
    except gzip.BadGzipFile:
        print(f"Error: '{filepath}' is not a valid gzip file.")
        raise
    except Exception as e:
        print(f"An error occurred while reading '{filepath}': {e}")
        raise

    return comment_lines

# function to create a hostname mapping (Client ID -> hostname)
def create_hostname_mapping(comment_lines):
    """
    Create a mapping of Client ID to hostname from the comment lines.
    """
    matches = [re.search(pattern, line) for line in comment_lines if re.search(pattern, line)]
    return {int(match.group(1)): match.group(2) for match in matches}

def read_csv_data(filepath, comment_lines_count, tx_worker_id=None, nrows=None, chunksize=50000):
    """
    Read CSV data with optional filtering by tx_worker_id. Always uses chunked reading for efficiency.
    """
    columns_needed = ['rx_worker_id', 'tx_worker_id', 'reply_src_addr', 'rx_time', 'tx_time', 'ttl']

    dtype_dict = {
        'rx_worker_id': 'int32',
        'tx_worker_id': 'int32',
        'reply_src_addr': 'object',
        'rx_time': 'int64',
        'tx_time': 'int64',
        'ttl': 'int16'
    }

    print(f"Processing CSV in chunks (chunksize: {chunksize:,})")
    if tx_worker_id is not None:
        print(f"Filtering for tx_worker_id: {tx_worker_id}")

    chunks = []
    total_rows_read = 0
    rows_kept = 0

    try:
        chunk_reader = pd.read_csv(
            filepath,
            skiprows=comment_lines_count,
            compression='gzip',
            usecols=columns_needed,
            dtype=dtype_dict,
            engine='c',
            chunksize=chunksize,
            na_filter=False,
            nrows=nrows,
            memory_map=True,
            low_memory=False
        )
    except pd.errors.ParserError as e:
        if "overflow" in str(e).lower():
            print(f"Overflow detected, retrying with string dtype for reply_src_addr...")
            # Retry with string dtype for problematic column
            dtype_dict_safe = dtype_dict.copy()
            dtype_dict_safe['reply_src_addr'] = 'str'
            chunk_reader = pd.read_csv(
                filepath,
                skiprows=comment_lines_count,
                compression='gzip',
                usecols=columns_needed,
                dtype=dtype_dict_safe,
                engine='c',
                chunksize=chunksize,
                na_filter=False,
                nrows=nrows,
                memory_map=True,
                low_memory=False
            )
        else:
            raise

    for chunk in chunk_reader:
        total_rows_read += len(chunk)

        # Apply filtering if tx_worker_id is specified
        if tx_worker_id is not None:
            filtered_chunk = chunk[chunk['tx_worker_id'] == tx_worker_id]
            if len(filtered_chunk) > 0:
                chunks.append(filtered_chunk)
                rows_kept += len(filtered_chunk)
        else:
            # No filtering, keep all data
            chunks.append(chunk)
            rows_kept += len(chunk)

        if total_rows_read % 500000 == 0:
            print(f"Processed {total_rows_read:,} rows, kept {rows_kept:,} rows")

    if chunks:
        result_df = pd.concat(chunks, ignore_index=True)
        print(f"Final: Processed {total_rows_read:,} rows, kept {len(result_df):,} rows")
    else:
        result_df = pd.DataFrame(columns=columns_needed)
        for col, dtype in dtype_dict.items():
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(dtype)

    return result_df

# Pre-load and optimize IP lookup database
def load_ip_lookup_database():
    """
    Load IP lookup database and create optimized structures for IPv4 and IPv6.
    """
    print("Loading IP2Location database...")

    # Load IPv4 database
    ipv4_lookup_df = pd.read_csv(
        Path("/Users/markusbecker/Desktop/Sonstiges/manycast_data") / "IP2LOCATION-LITE-DB5.CSV",
        header=None,
        usecols=[0, 1, 3],
        names=['ip_from', 'ip_to', 'country_name'],
        encoding='utf-8',
        engine='c',
        dtype={
            'ip_from': 'int64',
            'ip_to': 'int64',
            'country_name': 'string'
        }
    )

    # Sort and optimize for binary search
    ipv4_lookup_df = ipv4_lookup_df.sort_values(by='ip_from').reset_index(drop=True)

    # Pre-filter for Spain and Portugal to speed up lookups
    iberian_mask = ipv4_lookup_df['country_name'].isin(['Spain', 'Portugal'])
    iberian_ipv4_df = ipv4_lookup_df[iberian_mask].copy()

    # Load IPv6 database
    print("Loading IPv6 country database...")
    ipv6_lookup_df = pd.read_csv(
        Path("/Users/markusbecker/Desktop/Sonstiges/manycast_data") / "ip2country-v6.tsv",
        sep='\t',
        header=None,
        names=['ipv6_from', 'ipv6_to', 'country_code'],
        dtype={
            'ipv6_from': 'string',
            'ipv6_to': 'string',
            'country_code': 'string'
        }
    )

    # Filter IPv6 for Spain (ES) and Portugal (PT)
    iberian_ipv6_df = ipv6_lookup_df[ipv6_lookup_df['country_code'].isin(['ES', 'PT'])].copy()

    # Convert IPv6 addresses to integers for faster lookup
    def ipv6_to_int(ipv6_str):
        try:
            return int(ipaddress.IPv6Address(ipv6_str))
        except:
            return None

    iberian_ipv6_df['ipv6_from_int'] = iberian_ipv6_df['ipv6_from'].apply(ipv6_to_int)
    iberian_ipv6_df['ipv6_to_int'] = iberian_ipv6_df['ipv6_to'].apply(ipv6_to_int)

    # Remove rows where conversion failed
    iberian_ipv6_df = iberian_ipv6_df.dropna(subset=['ipv6_from_int', 'ipv6_to_int']).copy()
    iberian_ipv6_df = iberian_ipv6_df.sort_values(by='ipv6_from_int').reset_index(drop=True)

    # Map country codes to full names for consistency
    country_mapping = {'ES': 'Spain', 'PT': 'Portugal'}
    iberian_ipv6_df['country_name'] = iberian_ipv6_df['country_code'].map(country_mapping)

    print(f"Loaded IPv4 database: {len(ipv4_lookup_df):,} total ranges, {len(iberian_ipv4_df):,} Iberian ranges")
    print(f"Loaded IPv6 database: {len(iberian_ipv6_df):,} Iberian ranges")

    return ipv4_lookup_df, iberian_ipv4_df, iberian_ipv6_df


def load_asn_database():
    """
    Load ASN database from ip2asn-combined.tsv for both IPv4 and IPv6.
    """
    print("Loading ASN database...")

    try:
        # Load the combined ASN database
        asn_df = pd.read_csv(
            Path("/Users/markusbecker/Desktop/Sonstiges/manycast_data") / "ip2asn-combined.tsv",
            sep='\t',
            header=None,
            names=['ip_from', 'ip_to', 'asn', 'country_code', 'operator'],
            dtype={
                'ip_from': 'string',
                'ip_to': 'string',
                'asn': 'int64',
                'country_code': 'string',
                'operator': 'string'
            }
        )

        # Separate IPv4 and IPv6 entries
        ipv4_asn_df = []
        ipv6_asn_df = []

        for _, row in asn_df.iterrows():
            try:
                # Try to parse as IPv4 first
                ip_from = ipaddress.ip_address(row['ip_from'])
                ip_to = ipaddress.ip_address(row['ip_to'])

                if isinstance(ip_from, ipaddress.IPv4Address):
                    ipv4_asn_df.append({
                        'ip_from_int': int(ip_from),
                        'ip_to_int': int(ip_to),
                        'asn': row['asn'],
                        'operator': row['operator']
                    })
                elif isinstance(ip_from, ipaddress.IPv6Address):
                    ipv6_asn_df.append({
                        'ip_from_int': int(ip_from),
                        'ip_to_int': int(ip_to),
                        'asn': row['asn'],
                        'operator': row['operator']
                    })
            except:
                continue

        # Convert to DataFrames and sort for binary search
        ipv4_asn_df = pd.DataFrame(ipv4_asn_df).sort_values(by='ip_from_int').reset_index(drop=True)
        ipv6_asn_df = pd.DataFrame(ipv6_asn_df).sort_values(by='ip_from_int').reset_index(drop=True)

        print(f"Loaded ASN database: {len(ipv4_asn_df):,} IPv4 ranges, {len(ipv6_asn_df):,} IPv6 ranges")

        return ipv4_asn_df, ipv6_asn_df

    except FileNotFoundError:
        print("Warning: ip2asn-combined.tsv not found. ASN lookup will be disabled.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"Error loading ASN database: {e}")
        return pd.DataFrame(), pd.DataFrame()

def lookup_iberian_locations(ip_networks, iberian_ipv4_df, iberian_ipv6_df):
    """
    Lookup countries for IP networks, focusing on Iberian Peninsula.
    Supports both IPv4 and IPv6 networks.
    """
    # Separate IPv4 and IPv6 networks
    ipv4_networks = []
    ipv6_networks = []
    ipv4_indices = []
    ipv6_indices = []

    for i, net in enumerate(ip_networks):
        if isinstance(net.network_address, ipaddress.IPv4Address):
            ipv4_networks.append(net)
            ipv4_indices.append(i)
        elif isinstance(net.network_address, ipaddress.IPv6Address):
            ipv6_networks.append(net)
            ipv6_indices.append(i)

    # Initialize results
    countries = np.full(len(ip_networks), None, dtype=object)

    # Process IPv4 networks
    if ipv4_networks and len(iberian_ipv4_df) > 0:
        ipv4_ints = np.array([int(net.network_address) for net in ipv4_networks])

        iberian_indices = iberian_ipv4_df['ip_from'].searchsorted(ipv4_ints, side='right') - 1
        valid_iberian_mask = (iberian_indices >= 0) & (iberian_indices < len(iberian_ipv4_df))

        if np.any(valid_iberian_mask):
            valid_iberian_indices = iberian_indices[valid_iberian_mask]
            valid_ip_ints = ipv4_ints[valid_iberian_mask]

            # Check ranges
            ip_from_vals = iberian_ipv4_df.iloc[valid_iberian_indices]['ip_from'].values
            ip_to_vals = iberian_ipv4_df.iloc[valid_iberian_indices]['ip_to'].values

            iberian_range_mask = (ip_from_vals <= valid_ip_ints) & (valid_ip_ints <= ip_to_vals)

            if np.any(iberian_range_mask):
                final_indices = valid_iberian_indices[iberian_range_mask]
                original_positions_in_ipv4 = np.where(valid_iberian_mask)[0][iberian_range_mask]

                # Map back to original indices
                original_positions = np.array(ipv4_indices)[original_positions_in_ipv4]

                countries[original_positions] = iberian_ipv4_df.iloc[final_indices]['country_name'].values

    # Process IPv6 networks
    if ipv6_networks and len(iberian_ipv6_df) > 0:
        ipv6_ints = np.array([int(net.network_address) for net in ipv6_networks])

        iberian_indices = iberian_ipv6_df['ipv6_from_int'].searchsorted(ipv6_ints, side='right') - 1
        valid_iberian_mask = (iberian_indices >= 0) & (iberian_indices < len(iberian_ipv6_df))

        if np.any(valid_iberian_mask):
            valid_iberian_indices = iberian_indices[valid_iberian_mask]
            valid_ip_ints = ipv6_ints[valid_iberian_mask]

            # Check ranges
            ipv6_from_vals = iberian_ipv6_df.iloc[valid_iberian_indices]['ipv6_from_int'].values
            ipv6_to_vals = iberian_ipv6_df.iloc[valid_iberian_indices]['ipv6_to_int'].values

            iberian_range_mask = (ipv6_from_vals <= valid_ip_ints) & (valid_ip_ints <= ipv6_to_vals)

            if np.any(iberian_range_mask):
                final_indices = valid_iberian_indices[iberian_range_mask]
                original_positions_in_ipv6 = np.where(valid_iberian_mask)[0][iberian_range_mask]

                # Map back to original indices
                original_positions = np.array(ipv6_indices)[original_positions_in_ipv6]

                countries[original_positions] = iberian_ipv6_df.iloc[final_indices]['country_name'].values

    return countries


def lookup_asn_info(ip_networks, ipv4_asn_df, ipv6_asn_df):
    """
    Lookup ASN and operator information for IP networks.
    Supports both IPv4 and IPv6 networks.
    """
    # Separate IPv4 and IPv6 networks
    ipv4_networks = []
    ipv6_networks = []
    ipv4_indices = []
    ipv6_indices = []

    for i, net in enumerate(ip_networks):
        if isinstance(net.network_address, ipaddress.IPv4Address):
            ipv4_networks.append(net)
            ipv4_indices.append(i)
        elif isinstance(net.network_address, ipaddress.IPv6Address):
            ipv6_networks.append(net)
            ipv6_indices.append(i)

    # Initialize results
    asns = np.full(len(ip_networks), None, dtype=object)
    operators = np.full(len(ip_networks), None, dtype=object)

    # Process IPv4 networks
    if ipv4_networks and len(ipv4_asn_df) > 0:
        ipv4_ints = np.array([int(net.network_address) for net in ipv4_networks])

        asn_indices = ipv4_asn_df['ip_from_int'].searchsorted(ipv4_ints, side='right') - 1
        valid_asn_mask = (asn_indices >= 0) & (asn_indices < len(ipv4_asn_df))

        if np.any(valid_asn_mask):
            valid_asn_indices = asn_indices[valid_asn_mask]
            valid_ip_ints = ipv4_ints[valid_asn_mask]

            # Check ranges
            ip_from_vals = ipv4_asn_df.iloc[valid_asn_indices]['ip_from_int'].values
            ip_to_vals = ipv4_asn_df.iloc[valid_asn_indices]['ip_to_int'].values

            asn_range_mask = (ip_from_vals <= valid_ip_ints) & (valid_ip_ints <= ip_to_vals)

            if np.any(asn_range_mask):
                final_indices = valid_asn_indices[asn_range_mask]
                original_positions_in_ipv4 = np.where(valid_asn_mask)[0][asn_range_mask]

                # Map back to original indices
                original_positions = np.array(ipv4_indices)[original_positions_in_ipv4]

                asns[original_positions] = ipv4_asn_df.iloc[final_indices]['asn'].values
                operators[original_positions] = ipv4_asn_df.iloc[final_indices]['operator'].values

    # Process IPv6 networks
    if ipv6_networks and len(ipv6_asn_df) > 0:
        ipv6_ints = np.array([int(net.network_address) for net in ipv6_networks])

        asn_indices = ipv6_asn_df['ip_from_int'].searchsorted(ipv6_ints, side='right') - 1
        valid_asn_mask = (asn_indices >= 0) & (asn_indices < len(ipv6_asn_df))

        if np.any(valid_asn_mask):
            valid_asn_indices = asn_indices[valid_asn_mask]
            valid_ip_ints = ipv6_ints[valid_asn_mask]

            # Check ranges
            ipv6_from_vals = ipv6_asn_df.iloc[valid_asn_indices]['ip_from_int'].values
            ipv6_to_vals = ipv6_asn_df.iloc[valid_asn_indices]['ip_to_int'].values

            asn_range_mask = (ipv6_from_vals <= valid_ip_ints) & (valid_ip_ints <= ipv6_to_vals)

            if np.any(asn_range_mask):
                final_indices = valid_asn_indices[asn_range_mask]
                original_positions_in_ipv6 = np.where(valid_asn_mask)[0][asn_range_mask]

                # Map back to original indices
                original_positions = np.array(ipv6_indices)[original_positions_in_ipv6]

                asns[original_positions] = ipv6_asn_df.iloc[final_indices]['asn'].values
                operators[original_positions] = ipv6_asn_df.iloc[final_indices]['operator'].values

    return asns, operators

def calculate_hops_from_ttl(ttl_values):
    """
    Calculate number of hops from TTL values using vectorized operations.
    Assumes common default TTL values: Linux=64, Windows=128, Cisco=255.
    """
    ttl_array = np.array(ttl_values, dtype=np.int16)

    # Common default TTL values
    default_ttls = np.array([64, 128, 255], dtype=np.int16)

    # Calculate offsets for each default TTL (broadcast to handle all values at once)
    offsets = default_ttls[:, np.newaxis] - ttl_array[np.newaxis, :]

    # Only consider positive offsets (valid hops) and filter out > 50 hops
    # Use np.inf instead of -1 to properly handle minimum calculation
    valid_offsets = np.where((offsets > 0) & (offsets <= 50), offsets, np.inf)

    # Find minimum valid offset for each TTL value
    min_offsets = np.min(valid_offsets, axis=0)

    # Replace inf with NaN for cases where no valid default TTL was found
    # Convert valid offsets to integers, keep NaN for invalid cases
    hops = np.where(np.isinf(min_offsets), np.nan, min_offsets)

    return pd.array(hops, dtype=pd.Int64Dtype())  # Pandas nullable integer array

def process_data(filepath, comment_lines_count, hostname_map, tx_worker_id=None, nrows=None):
    """
    Main data processing pipeline.
    """
    print("=== Starting data processing ===")

    # Load IP lookup database
    full_ipv4_df, iberian_ipv4_df, iberian_ipv6_df = load_ip_lookup_database()

    # Load ASN database
    ipv4_asn_df, ipv6_asn_df = load_asn_database()

    # Load CSV data
    print("Loading CSV data...")
    result_df = read_csv_data(filepath, comment_lines_count, tx_worker_id=tx_worker_id, nrows=nrows)

    if len(result_df) == 0:
        print("No data after filtering")
        return pd.DataFrame()

    print(f"Processing {len(result_df):,} rows...")

    # Convert IPs to networks - handle both IPv4 and IPv6
    print("Converting IPs to networks...")
    def create_network(ip_int):
        try:
            # Handle both string and numeric IP addresses
            if isinstance(ip_int, str):
                # Try to parse as IP address string first
                try:
                    ip_addr = ipaddress.ip_address(ip_int)
                except ValueError:
                    # If that fails, try to convert to int first
                    ip_addr = ipaddress.ip_address(int(ip_int))
            else:
                # Convert numeric value to IP address
                # Handle very large IPv6 addresses that might overflow
                try:
                    if ip_int > 2**64:  # Likely IPv6
                        ip_addr = ipaddress.ip_address(int(ip_int))
                    else:
                        ip_addr = ipaddress.ip_address(int(ip_int))
                except (ValueError, OverflowError):
                    return None

            if isinstance(ip_addr, ipaddress.IPv4Address):
                # IPv4: use /24 network
                network_int = int(ip_addr) & 0xFFFFFF00
                return ipaddress.ip_network(f"{ipaddress.ip_address(network_int)}/24")
            else:
                # IPv6: use /64 network
                return ipaddress.ip_network(f"{ip_addr}/64", strict=False)
        except (ValueError, OverflowError, TypeError):
            return None

    result_df['target'] = [create_network(ip) for ip in result_df['reply_src_addr'].values]

    # Remove rows where network creation failed
    result_df = result_df[result_df['target'].notna()].copy()

    # Map hostnames
    print("Mapping hostnames...")
    result_df['receiver'] = result_df['rx_worker_id'].map(hostname_map)
    result_df['sender'] = result_df['tx_worker_id'].map(hostname_map)

    # Calculate RTT
    print("Calculating RTT...")
    result_df['rtt'] = (result_df['rx_time'] - result_df['tx_time']) / 1e6

    # Calculate hops from TTL
    print("Calculating hops from TTL...")
    result_df['hops'] = calculate_hops_from_ttl(result_df['ttl'].values)

    # Location lookup
    print("Performing location lookup...")
    countries = lookup_iberian_locations(result_df['target'].values, iberian_ipv4_df, iberian_ipv6_df)

    # ASN lookup
    print("Performing ASN lookup...")
    asns, operators = lookup_asn_info(result_df['target'].values, ipv4_asn_df, ipv6_asn_df)

    # Add location and ASN data
    result_df = result_df.copy()
    result_df['country'] = countries
    result_df['asn'] = asns
    result_df['operator'] = operators

    # Filter for Iberian Peninsula
    print("Filtering for Spain and Portugal...")
    iberian_df = result_df[result_df['country'].isin(['Spain', 'Portugal'])].copy()

    # Select final columns
    final_df = iberian_df[['receiver', 'sender', 'target', 'rtt', 'ttl', 'hops', 'country', 'asn', 'operator']]

    print(f"=== Processing complete: {len(final_df):,} Iberian targets found ===")
    return final_df

def get_worker_id_by_hostname(hostname_map, hostname):
    """
    Get worker ID by hostname from the hostname mapping.
    Returns the worker ID if found, otherwise raises ValueError.
    """
    # Create reverse mapping: hostname -> worker_id
    reverse_map = {hostname: worker_id for worker_id, hostname in hostname_map.items()}

    if hostname not in reverse_map:
        available_hostnames = list(reverse_map.keys())
        raise ValueError(f"Hostname '{hostname}' not found. Available hostnames: {available_hostnames}")

    return reverse_map[hostname]

def extract_date_from_filename(filename):
    """
    Extract date components from filename.
    Handles both GCD_ICMPv4YYYY-MM-DD and MAnycast_ICMPv4YYYY-MM-DD formats.
    """
    # Match patterns like GCD_ICMPv42025-04-24 or MAnycast_ICMPv42025-04-24
    match = re.search(r'(GCD|MAnycast)_ICMPv[46](\d{4})-(\d{2})-(\d{2})', filename)
    if match:
        return int(match.group(2)), int(match.group(3)), int(match.group(4))  # year, month, day
    return None, None, None

def get_all_csv_files():
    """
    Get all CSV files from the manycast_data subdirectories with their metadata.
    Returns list of tuples: (filepath, subdirectory, year, month, day)
    """
    base_path = Path("/Users/markusbecker/Desktop/Sonstiges/manycast_data")
    subdirectories = ['manycast_v4', 'manycast_v6', 'unicast_v4', 'unicast_v6']
    all_files = []

    if not base_path.exists():
        print(f"Warning: Directory {base_path} does not exist!")
        return all_files

    for subdir in subdirectories:
        subdir_path = base_path / subdir
        if not subdir_path.exists():
            print(f"Warning: Directory {subdir_path} does not exist, skipping...")
            continue

        # Find all .csv.gz files in subdirectory
        pattern_files = glob.glob(str(subdir_path / "*.csv.gz"))

        for filepath in pattern_files:
            filename = os.path.basename(filepath)
            year, month, day = extract_date_from_filename(filename)

            if year and month and day:
                all_files.append((filepath, subdir, year, month, day))
            else:
                print(f"Warning: Could not parse date from filename: {filename}")

    return sorted(all_files, key=lambda x: (x[2], x[3], x[4], x[1]))  # Sort by date, then subdirectory

def process_single_file(filepath, subdir, year, month, day, target_hostname):
    """
    Process a single CSV file and save the filtered result.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"Subdirectory: {subdir}, Date: {year}-{month:02d}-{day:02d}")
    print(f"{'='*60}")

    # Create output directory structure and check if file already exists
    output_base_dir = Path("filtered_data")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename with subdirectory info
    output_filename = f"filtered_{subdir}_{year}_{month:02d}_{day:02d}.csv"
    output_path = output_base_dir / output_filename

    # Check if filtered file already exists
    if output_path.exists():
        print(f"SKIPPED: Filtered file already exists at {output_path}")
        print(f"To reprocess, delete the existing file first.")
        return True  # Return True to count as successful (already processed)

    try:
        # Read metadata
        comment_lines = read_gzipped_comment_lines(filepath)
        print(comment_lines)
        hostname_map = create_hostname_mapping(comment_lines)
        print(hostname_map)

        # Get worker ID for target hostname
        try:
            tx_worker_id = get_worker_id_by_hostname(hostname_map, target_hostname)
            print(f"Using worker ID {tx_worker_id} for hostname '{target_hostname}'")
        except ValueError as e:
            print(f"Error: {e}")
            return False

        # Process data
        result_df = process_data(filepath, len(comment_lines), hostname_map, tx_worker_id=tx_worker_id, nrows=None)

        if len(result_df) > 0:
            # Save filtered data
            print(f"Saving {len(result_df):,} filtered records to: {output_path}")
            result_df.to_csv(output_path, index=False)

            # Print summary
            if 'country' in result_df.columns:
                country_counts = result_df['country'].value_counts()
                print("Breakdown by country:")
                for country, count in country_counts.items():
                    print(f"  {country}: {count:,}")

            print(f"Successfully processed {os.path.basename(filepath)}")
            return True
        else:
            print(f"No Iberian targets found in {os.path.basename(filepath)}")
            return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

##### SCRIPT START #####

# Configuration
target_hostname = "de-fra-manycast"  # Change this to desired hostname

print("Starting batch processing of all CSV files...")
print(f"Target hostname: {target_hostname}")

# Get all CSV files
all_files = get_all_csv_files()

if not all_files:
    print("No CSV files found in the subdirectories!")
    exit(1)

print(f"\nFound {len(all_files)} CSV files to process:")
for filepath, subdir, year, month, day in all_files:
    print(f"  {subdir}: {os.path.basename(filepath)} ({year}-{month:02d}-{day:02d})")

# Process all files
successful_files = 0
total_files = len(all_files)

for i, (filepath, subdir, year, month, day) in enumerate(all_files, 1):
    print(f"\n[{i}/{total_files}] Processing file {i} of {total_files}")

    success = process_single_file(filepath, subdir, year, month, day, target_hostname)
    if success:
        successful_files += 1

print(f"\n{'='*60}")
print("BATCH PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Total files processed: {total_files}")
print(f"Successfully processed: {successful_files}")
print(f"Failed: {total_files - successful_files}")

if successful_files > 0:
    print(f"\nFiltered data saved in 'filtered_data/' directory:")
    output_base = Path("filtered_data")
    if output_base.exists():
        csv_files = list(output_base.glob("*.csv"))
        print(f"  Total filtered files: {len(csv_files)}")

        # Count by type
        type_counts = {}
        for csv_file in csv_files:
            if 'manycast_v4' in csv_file.name:
                type_counts['manycast_v4'] = type_counts.get('manycast_v4', 0) + 1
            elif 'manycast_v6' in csv_file.name:
                type_counts['manycast_v6'] = type_counts.get('manycast_v6', 0) + 1
            elif 'unicast_v4' in csv_file.name:
                type_counts['unicast_v4'] = type_counts.get('unicast_v4', 0) + 1
            elif 'unicast_v6' in csv_file.name:
                type_counts['unicast_v6'] = type_counts.get('unicast_v6', 0) + 1

        for file_type, count in type_counts.items():
            print(f"    {file_type}: {count} files")
        for csv_file in csv_files:
            if 'manycast_v4' in csv_file.name:
                type_counts['manycast_v4'] = type_counts.get('manycast_v4', 0) + 1
            elif 'manycast_v6' in csv_file.name:
                type_counts['manycast_v6'] = type_counts.get('manycast_v6', 0) + 1
            elif 'unicast_v4' in csv_file.name:
                type_counts['unicast_v4'] = type_counts.get('unicast_v4', 0) + 1
            elif 'unicast_v6' in csv_file.name:
                type_counts['unicast_v6'] = type_counts.get('unicast_v6', 0) + 1

        for file_type, count in type_counts.items():
            print(f"    {file_type}: {count} files")
