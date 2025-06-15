# imports
from typing import Generator
from pathlib import Path
import os
import gzip
import re
import ipaddress
import zipfile
import logging
import time

import requests
import pandas as pd
import swifter
from dataclasses import dataclass

import boto3
from botocore.utils import fix_s3_host
from botocore.config import Config
from botocore.exceptions import EndpointConnectionError


logger = logging.getLogger("Iberian_Logger")

# configure logger
logging_config = {
    "version": 1,
    # dont log from 3rd party modules
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(name)s | %(levelname)s]:\n%(message)s\n"
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "Iberian_Logger": {
            "level": "DEBUG",
            "handlers": ["stdout"]
        }
    }
}


@dataclass(frozen=True, slots=True)
class DataDate:
    year: int
    month: int
    day: int
    def __repr__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

# load environment variables from .env
def load_env_file(env_file: Path):
    """
    Load variables from a .env style file into os.environ to use in the current context

    Args:
        env_file (Path): Path to the .env-style file
    """
    with open(env_file, "r") as f:
        for line in f:
            if line.strip() == '' or line.strip().startswith('#'):
                continue
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

def get_bucket():
    """
    Creates a bucket based on provided environment variables

    Make sure AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET, AWS_ENDPOINT_URL, AWS_BUCKET_NAME
    are set in os.environ before creating the bucket
    """
        # Create boto3 resource using environment variables
    S3 = boto3.resource(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_ACCESS_KEY_SECRET'],
        endpoint_url=os.environ['AWS_ENDPOINT_URL'],
        # Change timeouts in case we are uploading large files
        config=Config(
            connect_timeout=3, 
            read_timeout=900, 
            retries={"max_attempts":0}
        )
    )

    # Unregister to ensure requests don’t go to AWS
    S3.meta.client.meta.events.unregister('before-sign.s3', fix_s3_host)

    # Use bucket name from environment
    return S3.Bucket(os.environ['AWS_BUCKET_NAME'])

def download_minio_file(bucket, data_dir: Path, date: DataDate, anycast=True, ipv6=False) -> Generator:
    """Download a (Manycast/Unicast, IPv4/IPv6) file from MinIO if not already present."""
    # Build object prefix based on date
    prefix = f"manycast/{date.year}/{date.month:02}/{date.day:02}/"

    # Choose file pattern based on anycast and IP version
    protocol = "ICMPv6" if ipv6 else "ICMPv4"
    base_pattern = f"MAnycast_{protocol}" if anycast else f"GCD_{protocol}"

    # Search for matching file in bucket
    try:
        for obj in bucket.objects.filter(Prefix=prefix):
            # Replace invalid Windows characters in filenames
            filename = re.sub(r'[:<>"/\\|?*]', '_', obj.key[len(prefix):])
            filepath = os.path.join(data_dir, filename)

            if filename.startswith(base_pattern) and filename.endswith('.csv.gz'):
                logger.debug(f"Found file: {filename} (bucket key: {obj.key})")
                # Check if file already exists locally
                if os.path.exists(filepath):
                    logger.debug(f"File {filename} already exists. Skipping download.")
                else:
                    logger.debug(f"Downloading {filename} from bucket...")
                    os.makedirs(data_dir, exist_ok=True)
                    bucket.download_file(obj.key, filepath)

                yield filepath
    except EndpointConnectionError:
        logger.info("Could not access the Bucket, falling back to local data.")
        for file in os.listdir(data_dir):
            if file.startswith(base_pattern + f"{date}"):
                yield data_dir / file 

    logger.info("No matching file found.")

def download_ip2location_db(ip_geoloc_db: Path, ip_geoloc_dl_path: Path, ip_geoloc_dl_url: str):
    # retrieve ip2location database
    if not ip_geoloc_db.exists():
        # download
        response = requests.get(ip_geoloc_dl_url, allow_redirects=True)
        if not response.ok:
            logger.error("Download of IP2Location database failed!")
            return
        with open(ip_geoloc_dl_path, "wb") as f:
            f.write(response.content)

        # extract
        with zipfile.ZipFile(ip_geoloc_dl_path, "r") as zipf:
            zipf.extractall(ip_geoloc_db.parent)

        # remove downloaded zip file
        os.remove(ip_geoloc_dl_path)

# function to create a hostname mapping (Client ID -> hostname)
def extract_hostname_map(comment_lines: list[str]) -> dict[int, str]:
    """Map Client ID to hostname from comment lines."""
    pattern = r"ID:\s*(\d+)\s*,\s*hostname:\s*([\w-]+)"
    mapping = {}

    for line in comment_lines:
        if (match := re.search(pattern, line)):
            client_id = int(match.group(1))       # Extract Client ID
            hostname = match.group(2)             # Extract hostname
            mapping[client_id] = hostname

    return mapping

# MAnycastR files have metadata that give information about the measurement
def read_gzipped_comment_lines(filepath, comment_char='#'):
    """Read initial comment lines from a gzipped file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")

    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            comment_lines = []
            # Read lines until the first non-comment line
            for line in f:
                if line.startswith(comment_char):
                    comment_lines.append(line.rstrip())
                else:
                    # Stop at first non-comment line
                    break
            return comment_lines

    except gzip.BadGzipFile:
        raise gzip.BadGzipFile(f"Invalid gzip file: '{filepath}'")
    except Exception as e:
        raise RuntimeError(f"Error reading file '{filepath}': {e}")

# credit julian
def process_chunk(
    chunk: pd.DataFrame,
    hostname_map: dict[int, str],
    tx_worker_id: int | None = None
) -> pd.DataFrame:
    """
    Filter by tx_worker_id and process a DataFrame chunk to add hostname.
    """
    # Avoid SettingWithCopyWarning
    chunk = chunk.copy()

    # Filter rows on tx_worker_id if supplied
    if tx_worker_id is not None:
        chunk = chunk[chunk['tx_worker_id'] == tx_worker_id]

    # Map hostnames for sender and receiver
    chunk['receiver'] = chunk['rx_worker_id'].map(hostname_map)
    chunk['sender'] = chunk['tx_worker_id'].map(hostname_map)

    # Convert IP-number to ip network /24
    chunk['target'] = chunk['reply_src_addr'].apply(
        lambda x: ipaddress.ip_network(f"{ipaddress.ip_address(int(x))}/24", strict=False) # type: ignore
    )

    # Calculate RTT in seconds
    chunk['rtt'] = (chunk['rx_time'] - chunk['tx_time']) / 1e6

    # Return only the needed columns
    return chunk[['receiver', 'sender', 'target', 'reply_src_addr',  'rtt', 'ttl']]

def csv_to_df(
    filepath: str,
    hostname_map: dict[int, str],
    chunksize: int = 1_000_000,
    tx_worker: str | None = None,
    nrows_to_read: int | None = None
) -> pd.DataFrame:
    """
    Load a large, gzipped CSV file in chunks and filter rows by a given tx_worker_id.
    Prints progress and a summary upon completion.

    Returns:
        (DataFrame, read_time_sec, process_time_sec)
    """
    filtered_chunks = []
    chunk_num: int = 0

    # Get tx_worker_id if tx_worker name is supplied
    tx_worker_id = None
    if tx_worker is not None:
        try:
            tx_worker_id = next(k for k, v in hostname_map.items() if v.startswith(tx_worker))
        except StopIteration:
            raise ValueError(f"No tx_worker_id found for tx_worker name starting with: {tx_worker}")

    # Read the CSV file in chunks
    logger.info(f"Reading file: {filepath} in chunks of {chunksize:,} rows...")
    t0 = time.time()
    chunks = pd.read_csv(
        filepath,
        compression='gzip',
        comment='#',
        usecols=['rx_worker_id', 'tx_worker_id', 'reply_src_addr', 'rx_time', 'tx_time', 'ttl'],
        dtype={
            'rx_worker_id': 'uint8',
            'tx_worker_id': 'uint8',
            'reply_src_addr': 'uint32',
            'rx_time': 'float64',
            'tx_time': 'float64',
            'ttl': 'uint8'
        },
        chunksize=chunksize,
        # maybe for parallel reading?
        #        iterator=True
        nrows=nrows_to_read
    )

    # Process each chunk
    t1 = time.time()
    for chunk in chunks:
        filtered_chunk = process_chunk(chunk, hostname_map, tx_worker_id)
        filtered_chunks.append(filtered_chunk)
        chunk_num += 1

        # Print progress
        logger.info(f"Read {chunk_num} chunks")

    # Processing complete
    t2 = time.time()
    logger.debug(f"\nProcessing complete! Time taken: {t1 - t0:.2f}s (reading) + {t2 - t1:.2f}s (processing)")
    logger.info(f"Processed {sum(len(c) for c in filtered_chunks):,} entries!")

    return pd.concat(filtered_chunks, ignore_index=True)


def preproc_network_data(manycast_data_path: Path, tx_worker_name: str | None = None, nrows_to_read: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Process a MAnycast csv.gz file

    Returns:
        Tuple[pd.DataFrame, str]: A resulting dataframe containing the data and the metadata as a string
    """
    # Preprocessing of MAnycast data
    comment_lines = read_gzipped_comment_lines(manycast_data_path)
    hostname_map = extract_hostname_map(comment_lines)

    # load in data as a pandas dataframe
    # manycast_df = pd.read_csv(manycast_data_path, skiprows=len(comment_lines), nrows=ROWS, compression='gzip',
    #                           usecols=['rx_worker_id', 'tx_worker_id', 'reply_src_addr', 'rx_time', 'tx_time', 'ttl'])

    network_df = csv_to_df(str(manycast_data_path),
                           hostname_map,
                           tx_worker=tx_worker_name,
                           nrows_to_read=nrows_to_read)

    # convert IP-number to ip network
    # network_df['target'] = network_df['reply_src_addr'].swifter.apply(
    #     # can be sped up with swifter => uv pip install swifter "swifter[groupby] swifter[notebook]"
    #     lambda x: ipaddress.ip_network(f"{ipaddress.ip_address(x)}/24", strict=False)
    # )

    # # get receiving anycast site
    # network_df['receiver'] = network_df['rx_worker_id'].map(hostname_map)
    # # get sending anycast site
    # network_df['sender'] = network_df['tx_worker_id'].map(hostname_map)
    # # calculate rtt
    # network_df['rtt'] = ((network_df['rx_time'] - network_df['tx_time']) / 1e6)

    # make huebsch
    network_df = network_df.rename(columns={"reply_src_addr": "encoded_target_addr"})

    return network_df, comment_lines

def preproc_ip2location(ip_geoloc_db: Path) -> pd.DataFrame:
    # read the ip2location database
    return pd.read_csv(ip_geoloc_db,
                       usecols=[0, 1, 6, 7],
                       names=["ip_from",
                              "ip_to",
                              # "country_code",
                              # "country_name",
                              # "region",
                              # "city",
                              "lat",
                              "lon"],
                        dtype={
                            "ip_from": "uint32",
                            "ip_to": "uint32",
                            "lat": "float64",
                            "lon": "float64",
                        }
                       )

def get_manycast_geolocated(manycast_df: pd.DataFrame, ip_geoloc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: the manycast_df, with added lat and lon values corresponding to encoded_target_addr
    """
    sorted_ip_geoloc_df = ip_geoloc_df.sort_values("ip_from")
    sorted_manycast_df = manycast_df.sort_values("encoded_target_addr")

    # merge_asof for efficient range join
    merged = pd.merge_asof(
        sorted_manycast_df,
        sorted_ip_geoloc_df,
        left_on="encoded_target_addr",
        right_on="ip_from",
        direction="backward"
    )

    # filter entry when reply_src_addr > ip_to
    merged: pd.DataFrame = merged[merged["encoded_target_addr"] <= merged["ip_to"]]

    sorted_manycast_df.insert(len(sorted_manycast_df.columns), "lat", 0.0)
    sorted_manycast_df.insert(len(sorted_manycast_df.columns), "lon", 0.0)

    sorted_manycast_df["lat"] = merged["lat"].values
    sorted_manycast_df["lon"] = merged["lon"].values

    return sorted_manycast_df
