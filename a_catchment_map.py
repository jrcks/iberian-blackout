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

    lat_lon_without_dupes = df.drop_duplicates(subset=["lat_idx", "lon_idx"],
                                               keep="first")[["lat_idx", "lon_idx"]].values

    # create a mapping
    mapping ={tuple(k): v for k, v in zip(lat_lon_without_dupes,
                                          np.arange(0, lat_lon_without_dupes.shape[0], 1))}

    df["cluster"] = df.set_index(["lat_idx", "lon_idx"]).index.map(mapping)

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

def plot_target_geoloc_circles(geoloc_manycast_df: pd.DataFrame,
                               rx_worker_name: str,
                               rx_worker_pretty: str,
                               save_path: Path,
                               basemap_area: BasemapArea,
                               date: cm.DataDate,
                               lat_bins=20,
                               lon_bins=20,
                               big_thresh=800):

    # Drop rows with missing or zero coordinates
    geo_df = geoloc_manycast_df[(geoloc_manycast_df["lat"] != 0.0) & (geoloc_manycast_df["lon"] != 0.0)]
    cm.logger.info(f"Dropped {len(geoloc_manycast_df) - len(geo_df)} from being plotted (due to (lat, long) == (0, 0))")

    # tx_sender should get handled on csv read, only show hosts, which packets get received aat madrid
    geo_df_madrid = geo_df[geo_df["receiver"] == rx_worker_name]

    # retrieve clusters
    clustered_df, lat_edges, lon_edges = cluster_points(geo_df_madrid, basemap_area, lat_bins=lat_bins, lon_bins=lon_bins)
    centers_sizes = map_centers_sizes(clustered_df)

    # get amount of clusters, where size==1
    cm.logger.debug(f"Amount of clusters, where size == 1: {len(centers_sizes[centers_sizes["size"] == 1])}")

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

    vmax = np.max(counts)
    vmin = 10

    # vmax = np.max(counts)
    cm.logger.debug(centers_sizes)

    x, y = m(centers_sizes["lon"].values, centers_sizes["lat"])

    upper_percentile = np.percentile(counts ,80)

    # filter rows by sizes
    x_big = x[counts >= upper_percentile]
    y_big = y[counts >= upper_percentile]

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
                    s=counts,
                    alpha=0.6,
                    marker="o",
                    edgecolors='none',
                    norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax),
                    zorder=5)

    fig.colorbar(sc, label="Frequency")
    ax.set_title(f"Geolocation of Targets replying to {rx_worker_pretty}")
    ax.set_xlabel(f"Anycast data from Frankfurt to {rx_worker_pretty} at {date}")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")

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
    logging.config.dictConfig(config=cm.logging_config)

    # prepare environment
    cm.load_env_file(Path(__file__).parent / ".env")

    # coordinates madrid
    # lat=40.416775, long=-3.703790

    # define parameters
    date = cm.DataDate(2025, 4, 28)
    basemap_area = BasemapArea(30, -20, 50, 20)

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

    # TODO: retrieve for multiple days
    manycast_file = next(cm.download_minio_file(bucket, data_dir, date))

    # for testing limit the number of rows
    manycast_df, meta_inf = cm.preproc_network_data(manycast_file, "de-fra-manycast", nrows_to_read=1_000_000) 
    # manycast_df, meta_inf = cm.preproc_network_data(manycast_file, "de-fra-manycast")

    cm.logger.debug(f"{manycast_df}")
    cm.logger.debug(meta_inf)

    # process ip2location db
    cm.download_ip2location_db(ip_geolocation_db_path, ip_geolocation_download_path, ip_geolocation_url)
    ip2geolocation_df = cm.preproc_ip2location(ip_geolocation_db_path)

    # TODO: check logging context
    lat0_lon0_amount = ((ip2geolocation_df["lon"] == 0.0) & (ip2geolocation_df["lat"] == 0.0)).sum()
    cm.logger.debug(f"invalid lat long values found in ip2location database: {lat0_lon0_amount}")

    manycast_df = cm.get_manycast_geolocated(manycast_df, ip2geolocation_df)

    cm.logger.debug(manycast_df)

    # check for amount of "same" targets [304] maximum at 2025-04-28
    # cm.logger.debug(manycast_df.value_counts(["target"]))
    # cm.logger.debug(manycast_df.value_counts(["encoded_target_addr"]))
    # cm.logger.debug(manycast_df.value_counts(["probe_dst_addr"]))

    plot_target_geoloc_circles(manycast_df,
                              "es-mad-manycast",
                              "Madrid",
                              results_path / f"fra-mad_{date}.pdf",
                              basemap_area,
                              date,
                              lat_bins=16,
                              lon_bins=24,
                              big_thresh=800)



    # Load a GeoDataFrame with regional boundaries (e.g., US states, world countries)
    # For global countries:
    # regions = gpd.read_file(data_dir / "geoBoundariesCGAZ_ADM2.geojson")

    # plotly_geocord_colors(manycast_df, regions, "es-mad-manycast", results_path / f"fra-mad_plotly_{date}.pdf", basemap_area)

if __name__ == "__main__":
    main()
