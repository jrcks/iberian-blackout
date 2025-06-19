from typing import Callable
from pathlib import Path
import os
from dataclasses import dataclass
import logging.config
import commons as cm

import pandas as pd
import swifter
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as mplcolors
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN

import geopandas as gpd
import plotly.express as px
from shapely.geometry import Point, box

@dataclass(frozen=True, slots=True)
class BasemapArea:
    bottom_left_lat_deg: float | None = None
    bottom_left_lon_deg: float | None = None
    upper_right_lat_deg: float | None = None
    upper_right_lon_deg: float | None = None

def get_densest_location(group):
    counts = group.groupby(["lat", "lon"]).size()
    return counts.idxmax()

def cluster_points(geo_data: pd.DataFrame, basemap_area: BasemapArea, lat_bins=20, lon_bins=20) -> tuple[pd.DataFrame, npt.NDArray, npt.NDArray]:
    # using grid discretization and putting the points in the center
    coords =  geo_data[["lat", "lon"]].to_numpy()

    lat_edges = np.linspace(basemap_area.bottom_left_lat_deg, basemap_area.upper_right_lat_deg, lat_bins + 1)
    lon_edges = np.linspace(basemap_area.bottom_left_lon_deg, basemap_area.upper_right_lon_deg, lat_bins + 1)

    lat_idx = np.digitize(coords[:, 0], lat_edges) - 1
    lon_idx = np.digitize(coords[:, 1], lon_edges) - 1

    mask_oob = (
        (lat_idx >= 0) & (lat_idx < lat_bins) &
        (lon_idx >= 0) & (lon_idx < lon_bins)
    )

    df = pd.DataFrame(
        {"lat_idx": lat_idx[mask_oob],
         "lon_idx": lon_idx[mask_oob],
         "lat": coords[mask_oob][:, 0],
         "lon": coords[mask_oob][:, 1]
        }
    )

    # lat_lon_without_dupes = df.drop_duplicates(subset=["lat_idx", "lon_idx"],
    #                                            keep="first")[["lat_idx", "lon_idx"]].values

    # # create a mapping, which essentially assigns an id based on the lat_idx and lon_idx
    # mapping ={tuple(k): v for k, v in zip(lat_lon_without_dupes, lat_idx + lat_bins * lon_idx)}

    df["cluster"] = df.set_index(["lat_idx", "lon_idx"]).index.map(lambda x: x[0] + lat_bins * x[1])

    # DBSCAN approach
    # kms_per_radian = 6371.0088
    # base_eps_km = 500
    # db = DBSCAN(eps=base_eps_km / kms_per_radian, min_samples=1, metric="haversine")
    # geo_data["cluster"] = db.fit_predict(coords)

    return df, lat_edges, lon_edges

def map_centers_sizes(clustered_geo_data: pd.DataFrame) -> pd.DataFrame:
    cluster_sizes  = clustered_geo_data.groupby("cluster").size()
    # find for each cluster, one entry that contains the most occurences (lat, long)-wise
    centers = clustered_geo_data.swifter.groupby("cluster").apply(get_densest_location)

    lats, lons = tuple(zip(*centers.values))

    # create new dataframe containing the cluster centers and magnitudes
    center_magnitudes = pd.DataFrame({"lat": np.array(lats), "lon": np.array(lons), "size": cluster_sizes.values, "cluster_id": cluster_sizes.index})
    cm.logger.debug(center_magnitudes)
    return center_magnitudes

def cluster_geoloc_data(geoloc_df: pd.DataFrame,
                        basemap_area: BasemapArea,
                        rx_worker_name: str,
                        lat_bins=20,
                        lon_bins=20,
                        ):
    # Drop rows with missing or zero coordinates
    geo_df = geoloc_df[(geoloc_df["lat"] != 0.0) & (geoloc_df["lon"] != 0.0)]
    cm.logger.info(f"Dropped {len(geoloc_df) - len(geo_df)} from being plotted (due to (lat, long) == (0, 0))")

    # tx_sender should get handled on csv read, only show hosts, which packets get received aat madrid
    geo_df_madrid = geo_df[geo_df["receiver"] == rx_worker_name]

    # retrieve clusters
    clustered_df, lat_edges, lon_edges = cluster_points(geo_df_madrid, basemap_area, lat_bins=lat_bins, lon_bins=lon_bins)
    centers_sizes = map_centers_sizes(clustered_df)

    # get amount of clusters, where size==1
    cm.logger.debug(f"Amount of clusters, where size == 1: {len(centers_sizes[centers_sizes["size"] == 1])}")

    return centers_sizes, lat_edges, lon_edges

def plot_target_geoloc_circles(centers_sizes: pd.DataFrame,
                               basemap_area: BasemapArea,
                               rx_worker_pretty: str,
                               save_path: Path,
                               date: cm.DataDate,
                               lat_edges: npt.NDArray,
                               lon_edges: npt.NDArray,
                               big_thresh=800,
                               size_modifier_func: Callable | None=None,
                               vmax=None,
                               vmin=None):

    # Set up figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Initialize Basemap
    m = Basemap(projection='merc',
                llcrnrlat=basemap_area.bottom_left_lat_deg,
                urcrnrlat=basemap_area.upper_right_lat_deg,
                llcrnrlon=basemap_area.bottom_left_lon_deg,
                urcrnrlon=basemap_area.upper_right_lon_deg,
                resolution='l',
                ax=ax)

    # Draw map details
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='white', lake_color='lightblue')

    # sizes = np.round(counts, decimals=0)
    counts = centers_sizes["size"].values

    if vmax is None:
        vmax = np.max(counts)
    if vmin is None:
        vmin = np.min(counts)

    # vmax = np.max(counts)
    cm.logger.debug(centers_sizes)

    x, y = m(centers_sizes["lon"].values, centers_sizes["lat"])

    # filter rows by sizes
    x_big = x[counts >= big_thresh]
    y_big = y[counts >= big_thresh]

    # draw grid to indicate the "bins"
    for i in range(0, len(lat_edges) - 1):
        for j in range(0, len(lon_edges) - 1):
            lat0, lat1 = lat_edges[i], lat_edges[i + 1]
            lon0, lon1 = lon_edges[j], lon_edges[j + 1]

            x0, y0 = m(lon0, lat0)
            x1, y1 = m(lon1, lat1)
            width = x1 - x0
            height = y1 - y0

            rect = Rectangle((x0, y0), width, height,
                            linewidth=.8, edgecolor="gray", linestyle="--", fill=False, facecolor=None, alpha=.4, zorder=7)
            ax.add_patch(rect)


    # scatter center points for big circles
    ax.scatter(x_big,
               y_big,
               c="red",
               s=10,
               alpha=1,
               marker="o",
               edgecolors='none',
               zorder=6)

    # make sure counts for size is atleast 10 big
    sizes = counts
    sizes[counts < 10] = 10

    sc = ax.scatter(x,
                    y,
                    c=counts,
                    cmap="viridis",
                    s=sizes,
                    alpha=0.6,
                    marker="o",
                    edgecolors='none',
                    norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax),
                    zorder=5)

    cbar = fig.colorbar(sc, label="Frequency")
    cbar.ax.tick_params(labelsize = 16)
    ax.set_title(f"Geolocation of Targets replying to {rx_worker_pretty}", fontsize=20)
    ax.set_ylabel(f"Anycast data from Frankfurt to {rx_worker_pretty} at {date}", fontsize=16)
    fig.savefig(save_path, bbox_inches="tight", dpi=400)

def calc_diff_in_clusters(from_centers_sizes: pd.DataFrame,
                          to_centers_sizes: pd.DataFrame):
    """
    This will essentially calculate small vs big differences
    """
    center_sizes_merged = pd.merge(from_centers_sizes, to_centers_sizes, how="outer", on="cluster_id", suffixes=("_from", "_to"))
    center_sizes_merged.fillna(0, inplace=True)
    # calculate size_diff
    center_sizes_merged["size_diff"] = center_sizes_merged["size_to"] - center_sizes_merged["size_from"]

    return center_sizes_merged

def plot_diff_geoloc_circles(diffed_centers_sizes: pd.DataFrame,
                             basemap_area: BasemapArea,
                             rx_worker_pretty: str,
                             save_path: Path,
                             date_from: cm.DataDate,
                             date_to: cm.DataDate,
                             lat_edges: npt.NDArray,
                             lon_edges: npt.NDArray,
                             big_thresh=100,
                             vmin=None,
                             vmax=None):
    # Set up figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Initialize Basemap
    m = Basemap(projection='merc',
                llcrnrlat=basemap_area.bottom_left_lat_deg,
                urcrnrlat=basemap_area.upper_right_lat_deg,
                llcrnrlon=basemap_area.bottom_left_lon_deg,
                urcrnrlon=basemap_area.upper_right_lon_deg,
                resolution='l',
                ax=ax)

    # Draw map details
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='white', lake_color='lightblue')

    # sizes = np.round(counts, decimals=0)
    diffs = diffed_centers_sizes["size_diff"].values

    cm.logger.debug(diffs)
    if vmax is None:
        vmax = np.max(diffs)
    if vmin is None:
        vmin = np.min(diffs)

    # vmax = np.max(counts)
    cm.logger.debug(diffed_centers_sizes)


    big_mask = diffed_centers_sizes["size_from"] >= diffed_centers_sizes["size_to"]
    # take the biggest cluster for lon
    lon_big = np.where(big_mask, diffed_centers_sizes["lon_from"], diffed_centers_sizes["lon_to"])
    lat_big = np.where(big_mask, diffed_centers_sizes["lat_from"], diffed_centers_sizes["lat_to"])

    x, y = m(lon_big, lat_big)

    # filter rows by sizes
    x_big = x[np.abs(diffs) >= big_thresh]
    y_big = y[np.abs(diffs) >= big_thresh]

    # draw grid to indicate the "bins"
    for i in range(0, len(lat_edges) - 1):
        for j in range(0, len(lon_edges) - 1):
            lat0, lat1 = lat_edges[i], lat_edges[i + 1]
            lon0, lon1 = lon_edges[j], lon_edges[j + 1]

            x0, y0 = m(lon0, lat0)
            x1, y1 = m(lon1, lat1)
            width = x1 - x0
            height = y1 - y0

            rect = Rectangle((x0, y0), width, height,
                            linewidth=.8, edgecolor="gray", linestyle="--", fill=False, facecolor=None, alpha=.4, zorder=7)
            ax.add_patch(rect)


    # scatter center points for big circles
    ax.scatter(x_big,
               y_big,
               c="black",
               s=10,
               alpha=1,
               marker="o",
               edgecolors='none',
               zorder=6)

    # make sure counts for size is atleast 10 big otherwise the points cannot be seen on the map
    sizes = np.abs(diffs)
    sizes[sizes < 10] = 10
    cm.logger.debug(f"{vmin}, {vmax}")

    sc = ax.scatter(x,
                    y,
                    c=diffs,
                    cmap="managua_r",
                    s=sizes,
                    alpha=0.4,
                    marker="o",
                    edgecolors='none',
                    norm=mplcolors.SymLogNorm(10, vmin=vmin, vmax=vmax),
                    zorder=5)

    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])

    cbar = fig.colorbar(sc, label="Frequency difference", cax=cax)
    cbar.ax.tick_params(labelsize = 16)
    ax.set_title(f"Replying targets difference from {date_from} to {date_to}", fontsize=20)
    ax.set_ylabel(f"Anycast data from Frankfurt to {rx_worker_pretty}", fontsize=16)
    fig.savefig(save_path, bbox_inches="tight", dpi=400)

def plotly_geocord_colors(geoloc_manycast_df: pd.DataFrame,
                          regions,
                          rx_worker_name: str,
                          save_path: Path,
                          basemap_area: BasemapArea):

    # Drop rows with missing or zero coordinates
    geo_df = geoloc_manycast_df[(geoloc_manycast_df["lat"] != 0.0) & (geoloc_manycast_df["lon"] != 0.0)]
    cm.logger.info(f"Dropped {len(geoloc_manycast_df) - len(geo_df)} from being plotted (due to (lat, long) == (0, 0))")

    # tx_sender should get handled on csv read, only show hosts, which packets get received aat madrid
    geo_df_madrid = geo_df[geo_df["receiver"] == rx_worker_name]
    cm.logger.debug(geo_df_madrid)

    # Create GeoDataFrame from the input coordinates
    geometry = [Point(xy) for xy in zip(geo_df_madrid['lon'], geo_df_madrid['lat'])]
    gdf_points = gpd.GeoDataFrame(geo_df_madrid, geometry=geometry, crs=regions.crs)

    bbox = box(basemap_area.bottom_left_lon_deg, basemap_area.bottom_left_lat_deg, basemap_area.upper_right_lon_deg, basemap_area.upper_right_lat_deg)

    regions = regions[regions.intersects(bbox)]

    # Spatial join to assign each point to a region
    joined = gpd.sjoin(gdf_points, regions, how='left', predicate='within')
    cm.logger.debug(joined)

    # simplify regions
    # regions["geometry"] = regions["geometry"].simplify(tolerance=0.001)
    # cm.logger.debug(regions)

    # Count points per region (using 'name' for country names)
    counts = joined.groupby('shapeID').size().reset_index(name='count')

    # Merge counts with regions
    regions_counts = regions.merge(counts, on='shapeID', how='left').fillna({'count': 0})
    cm.logger.debug(regions_counts)

    # Plot Choropleth using Plotly
    fig = px.choropleth(regions_counts,
                        geojson=regions_counts.geometry,
                        locations=regions_counts.index,
                        projection="natural earth")

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title='Coordinate Count by Region')
    # fig.write_html(save_path)
    fig.write_image(save_path)

def main():
    analyze_days = [cm.DataDate(2025, 4, 28),
                    cm.DataDate(2025, 4, 29),
                    cm.DataDate(2025, 4, 30),
                    cm.DataDate(2025, 5, 1),
                    cm.DataDate(2025, 5, 2)]
                    ]
    # analyze_days = set([cm.DataDate(2025, 4, 30)])

    recv_site_name = "fr-cdg-manycast"
    recv_site_pretty_name = "Paris"
    file_recv_name = "cdg"

    # define which differences to plot
    analyze_combinations = [(0, 1), (1, 2), (2, 3), (3, 4)]

    analyze_days_stats = {}

    logging.config.dictConfig(config=cm.logging_config)

    # prepare environment
    cm.load_env_file(Path(__file__).parent / ".env")

    # coordinates madrid
    # lat=40.416775, long=-3.703790

    # define parameters
    basemap_area = BasemapArea(30, -11, 60, 40)

    # world map (almost world like in https://matplotlib.org/basemap/stable/users/merc.html)
    # basemap_area = BasemapArea(-80, -180, 80, 180)

    # Set up global constants for paths
    data_dir = Path("data/")
    ip_geolocation_db_path = data_dir / "IP2LOCATION-LITE-DB5.CSV"
    ip_geolocation_download_path = data_dir / "ip2location-lite-db5.zip"
    ip_geolocation_url = f"https://www.ip2location.com/download/?token={os.environ["IP2LOCATION_LITE_TOKEN"]}&file=DB5LITECSV"

    # make sure results dir exists
    results_path = data_dir / "results"
    os.makedirs(results_path, exist_ok=True)

    # process data
    bucket = cm.get_bucket()

    # process ip2location db
    cm.download_ip2location_db(ip_geolocation_db_path, ip_geolocation_download_path, ip_geolocation_url)
    ip2geolocation_df = cm.preproc_ip2location(ip_geolocation_db_path)

    vmin = None
    vmax = None

    for date in analyze_days:
        manycast_file = next(cm.download_minio_file(bucket, data_dir, date))

        # for testing limit the number of rows
        # manycast_df, meta_inf = cm.preproc_network_data(manycast_file, "de-fra-manycast", nrows_to_read=1_000_000)
        manycast_df, meta_inf = cm.preproc_network_data(manycast_file, "de-fra-manycast")

        # cm.logger.debug(f"{manycast_df}")
        # cm.logger.debug(meta_inf)

        # TODO: check logging context
        lat0_lon0_amount = ((ip2geolocation_df["lon"] == 0.0) & (ip2geolocation_df["lat"] == 0.0)).sum()
        cm.logger.debug(f"invalid lat long values found in ip2location database: {lat0_lon0_amount}")

        manycast_df = cm.get_manycast_geolocated(manycast_df, ip2geolocation_df)

        cm.logger.debug(manycast_df)

        centers_sizes, lat_edges, lon_edges = cluster_geoloc_data(manycast_df,
                                                                  basemap_area,
                                                                  recv_site_name,
                                                                  lat_bins=20,
                                                                  lon_bins=28)

        # save the stats for each date
        analyze_days_stats[date] = (centers_sizes, lat_edges, lon_edges)

        counts = centers_sizes["size"].values
        max_cnt = np.max(counts)
        min_cnt = np.min(counts)

        if vmin is None or vmin > min_cnt:
            vmin = min_cnt
        if vmax is None or vmax < max_cnt:
            vmax = max_cnt

    # now create a plot for each date but determine vmax beforehand
    for date, (centers_sizes, lat_edges, lon_edges) in analyze_days_stats.items():
        plot_target_geoloc_circles(centers_sizes,
                                   basemap_area,
                                   recv_site_pretty_name,
                                   results_path / f"fra-{file_recv_name}_{repr(date)}.png",
                                   date,
                                   lat_edges,
                                   lon_edges,
                                   big_thresh=800,
                                   vmin=vmin,
                                   vmax=vmax)


    from_lat_edges = None
    from_lon_edges = None

    sizes_list = []

    vmin = None
    vmax = None

    for idx_from, idx_to in analyze_combinations:
        date_from = analyze_days[idx_from]
        date_to = analyze_days[idx_to]
        centers_from, from_lat_edges, from_lon_edges = analyze_days_stats[date_from]
        centers_to, _, _ = analyze_days_stats[date_to]

        centers_diffs = calc_diff_in_clusters(centers_from, centers_to)

        max_cnt = np.max(centers_diffs)
        min_cnt = np.min(centers_diffs)

        if vmin is None or vmin > min_cnt:
            vmin = min_cnt
        if vmax is None or vmax < max_cnt:
            vmax = max_cnt

        sizes_list.append(centers_diffs)

    for (idx_from, idx_to), size_diffs in zip(analyze_combinations, sizes_list):
        date_from = analyze_days[idx_from]
        date_to = analyze_days[idx_to]
        plot_diff_geoloc_circles(size_diffs,
                                 basemap_area,
                                 recv_site_pretty_name,
                                 results_path / f"fra-{file_recv_name}_{repr(date_from)}_to_{repr(date_to)}.png",
                                 date_from,
                                 date_to,
                                 from_lat_edges,
                                 from_lon_edges,
                                 big_thresh=500,
                                 vmin=vmin,
                                 vmax=vmax)


    # Load a GeoDataFrame with regional boundaries (e.g., US states, world countries)
    # For global countries:
    # regions = gpd.read_file(data_dir / "geoBoundariesCGAZ_ADM2.geojson")

    # plotly_geocord_colors(manycast_df, regions, "es-mad-manycast", results_path / f"fra-mad_plotly_{date}.pdf", basemap_area)

if __name__ == "__main__":
    main()
