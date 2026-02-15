# T07 - carte folium

import os
import sys
import json
import time
import io
import base64
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource

from config import (
    MAPS_DIR, FIGURES_DIR, RGI_DIR, DEM_DIR, DERIVED_DIR,
    DEM_RESOLUTION, NODATA_VALUE, UHD_DPI,
    BBOX_WGS84, CRS_L93, CRS_WGS84,
    START_POINT_WGS84, END_POINT_WGS84,
    MAP_CENTER_WGS84, MAP_DEFAULT_ZOOM,
)

#  Chargement

def load_geojson(variant):
    """Charge un GeoJSON WGS84 (full ou simplified)."""
    fname = f"route_requin_to_aiguille_{DEM_RESOLUTION}m_wgs84_{variant}.geojson"
    path = os.path.join(MAPS_DIR, fname)
    if not os.path.exists(path):
        print(f"[error] introuvable: {path}")
        print("  -> lancer T06 d'abord")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    feature = data["features"][0]
    coords = feature["geometry"]["coordinates"]  # [lon, lat, elev]
    props = feature.get("properties", {})
    print(f"[load] {variant}: {len(coords)} points")
    return coords, props


def load_glaciers_wgs84():
    """Charge les polygones RGI, filtre bbox, reproject WGS84."""
    shp_paths = []
    for root, dirs, files in os.walk(RGI_DIR):
        for f in files:
            if f.endswith('.shp'):
                shp_paths.append(os.path.join(root, f))

    if not shp_paths:
        print("[warn] pas de shapefiles RGI, pas de couche glacier")
        return None

    gdf = gpd.read_file(shp_paths[0])

    # reproject WGS84 si necessaire
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(CRS_WGS84)

    bb = BBOX_WGS84
    gdf = gdf.cx[bb['lon_min']:bb['lon_max'], bb['lat_min']:bb['lat_max']]
    print(f"[glaciers] {len(gdf)} polygones dans la bbox")
    return gdf

#  Prfil altimetrique en base64
def make_elevation_profile_b64(coords):
    """Genere le profil d'altitude, retourne PNG en base64."""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    elevs = [c[2] for c in coords]

    # distance cumulee
    cum_dist = [0]
    for i in range(1, len(coords)):
        dlat = (lats[i] - lats[i-1]) * 111320
        dlon = (lons[i] - lons[i-1]) * 111320 * np.cos(np.radians(lats[i]))
        d = np.sqrt(dlat**2 + dlon**2)
        cum_dist.append(cum_dist[-1] + d)

    dist_km = np.array(cum_dist) / 1000
    elevs = np.array(elevs)

    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.fill_between(dist_km, elevs, elevs.min() - 30, color='#3498db', alpha=0.3)
    ax.plot(dist_km, elevs, color='#2c3e50', linewidth=1.2)

    ax.set_xlabel('Distance (km)', fontsize=8)
    ax.set_ylabel('Alt. (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, dist_km[-1])
    ax.set_ylim(elevs.min() - 50, elevs.max() + 50)

    ax.annotate(f'{elevs[0]:.0f}m', xy=(dist_km[0], elevs[0]),
                fontsize=7, color='#2ecc71', fontweight='bold')
    ax.annotate(f'{elevs[-1]:.0f}m', xy=(dist_km[-1], elevs[-1]),
                fontsize=7, color='#e74c3c', fontweight='bold', ha='right')

    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    print(f"[profile] image base64: {len(b64)//1024} KB")
    return b64

#  Construction folium

def build_map(coords_full, coords_simp, props, glaciers_gdf, profile_b64):
    m = folium.Map(
        location=MAP_CENTER_WGS84,
        zoom_start=MAP_DEFAULT_ZOOM,
        control_scale=True,
    )

    # --- basemaps ---
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap',
        name='OpenTopoMap',
        max_zoom=17,
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        max_zoom=18,
    ).add_to(m)

    # --- route gradiet par altitude ---
    route_fg = folium.FeatureGroup(name='Route', show=True)

    lats = [c[1] for c in coords_full]
    lons = [c[0] for c in coords_full]
    elevs = [c[2] for c in coords_full]
    positions = list(zip(lats, lons))

    elev_min, elev_max = min(elevs), max(elevs)
    norm = Normalize(vmin=elev_min, vmax=elev_max)
    cmap = plt.colormaps['RdYlGn_r']  # vert bas -> rouge haut

    # ColorLine : decoupe en segments colores
    # folium.ColorLine pas dispo partout, test a la main avec des PolyLine
    n = len(positions)
    step = max(1, n // 200)
    for i in range(0, n - step, step):
        j = min(i + step, n - 1)
        seg_latlng = positions[i:j+1]
        avg_elev = np.mean(elevs[i:j+1])
        rgba = cmap(norm(avg_elev))
        color = '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        folium.PolyLine(
            locations=seg_latlng,
            color=color,
            weight=4,
            opacity=0.85,
        ).add_to(route_fg)

    route_fg.add_to(m)

    # --- couche glaciers ---
    if glaciers_gdf is not None and len(glaciers_gdf) > 0:
        glacier_fg = folium.FeatureGroup(name='Glaciers', show=True)

        name_col = None
        for col in ['glac_name', 'name', 'Name', 'GLAC_NAME']:
            if col in glaciers_gdf.columns:
                name_col = col
                break

        style_fn = lambda x: {
            'fillColor': '#a8d8ea',
            'color': '#0077be',
            'fillOpacity': 0.3,
            'weight': 1.5,
        }

        for _, row in glaciers_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            name = ''
            if name_col and row[name_col] and str(row[name_col]).strip():
                name = str(row[name_col])
            tooltip = name if name else 'Glacier (RGI 7.0)'
            folium.GeoJson(
                geom.__geo_interface__,
                style_function=style_fn,
                tooltip=tooltip,
            ).add_to(glacier_fg)

        glacier_fg.add_to(m)

    # --- markers depart / arrivee ---
    elev_start = coords_full[0][2]
    elev_end = coords_full[-1][2]

    folium.Marker(
        location=[coords_full[0][1], coords_full[0][0]],
        popup=folium.Popup(
            f"<b>Depart - Refuge du Requin</b><br>"
            f"Alt: {elev_start:.0f}m<br>"
            f"({coords_full[0][1]:.5f}, {coords_full[0][0]:.5f})",
            max_width=250,
        ),
        icon=folium.Icon(color='green', icon='play', prefix='fa'),
    ).add_to(m)

    folium.Marker(
        location=[coords_full[-1][1], coords_full[-1][0]],
        popup=folium.Popup(
            f"<b>Arrivee - Aiguille du Midi</b><br>"
            f"Alt: {elev_end:.0f}m<br>"
            f"({coords_full[-1][1]:.5f}, {coords_full[-1][0]:.5f})",
            max_width=250,
        ),
        icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa'),
    ).add_to(m)

    # --- plugins ---
    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.Fullscreen().add_to(m)

    # --- panneau stats HTML ---
    dist_km = props.get('distance_km', '?')
    dplus = props.get('dplus_m', '?')
    dminus = props.get('dminus_m', '?')
    time_h = props.get('time_tobler_h', '?')
    glacier_pct = props.get('glacier_pct', '?')
    n_pts = props.get('n_points', '?')

    stats_html = f"""
    <div style="
        position: fixed;
        bottom: 30px; left: 10px;
        background: rgba(255,255,255,0.92);
        border: 1px solid #888;
        border-radius: 6px;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 12px;
        z-index: 9999;
        line-height: 1.6;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    ">
        <b>Requin &rarr; Aig. du Midi</b><br>
        Dist: {dist_km} km | D+ {dplus}m | D- {dminus}m<br>
        Temps Tobler: ~{time_h}h<br>
        Glacier: {glacier_pct}% | Pts: {n_pts}<br>
        Res: {DEM_RESOLUTION}m
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    # --- profil altimetrique collapsible ---
    profile_html = f"""
    <div id="profile-panel" style="
        position: fixed;
        bottom: 30px; right: 60px;
        background: rgba(255,255,255,0.95);
        border: 1px solid #888;
        border-radius: 6px;
        padding: 4px;
        z-index: 9998;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        max-width: 420px;
    ">
        <div style="text-align:right; margin-bottom:2px;">
            <button onclick="
                var img = document.getElementById('elev-img');
                img.style.display = img.style.display === 'none' ? 'block' : 'none';
            " style="font-size:11px; cursor:pointer; border:1px solid #aaa;
                     border-radius:3px; padding:1px 6px; background:#f0f0f0;">
                Profil
            </button>
        </div>
        <img id="elev-img" src="data:image/png;base64,{profile_b64}"
             style="width:400px; display:block;" />
    </div>
    """
    m.get_root().html.add_child(folium.Element(profile_html))

    # --- layer control ---
    folium.LayerControl(collapsed=False).add_to(m)

    return m

def export_html(m):
    os.makedirs(MAPS_DIR, exist_ok=True)
    fname = f"route_interactive_{DEM_RESOLUTION}m.html"
    path = os.path.join(MAPS_DIR, fname)
    m.save(path)
    size_kb = os.path.getsize(path) / 1024
    print(f"[html] {path} ({size_kb:.0f} KB)")
    return path

def export_static_png(coords_full, props, glaciers_gdf):
    """Figure matplotlib avec hillshade, route, glaciers."""
    import rasterio

    dem_path = os.path.join(DEM_DIR, f"dem_aiguille_du_midi_{DEM_RESOLUTION}m.tif")
    if not os.path.exists(dem_path):
        print("[warn] DEM introuvable, skip PNG statique")
        return None

    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float32)
        transform = ds.transform
        bounds = ds.bounds

    dem_display = np.where(dem == NODATA_VALUE, np.nan, dem)
    dem_filled = np.where(np.isnan(dem_display), 0, dem_display)
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem_filled, vert_exag=2,
                              dx=DEM_RESOLUTION, dy=DEM_RESOLUTION)

    extent_l93 = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    from pyproj import Transformer # route en L93 via pyproj
    proj = Transformer.from_crs(CRS_WGS84, CRS_L93, always_xy=True)

    route_lons = [c[0] for c in coords_full]
    route_lats = [c[1] for c in coords_full]
    route_elevs = [c[2] for c in coords_full]
    rx, ry = proj.transform(route_lons, route_lats)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(hillshade, cmap='gray', alpha=0.5, extent=extent_l93, origin='upper')
    dem_masked = np.where(dem == NODATA_VALUE, np.nan, dem)
    ax.imshow(dem_masked, cmap='terrain', alpha=0.4, extent=extent_l93, origin='upper')

    if glaciers_gdf is not None and len(glaciers_gdf) > 0:
        gdf_l93 = glaciers_gdf.to_crs(CRS_L93)
        gdf_l93.plot(ax=ax, facecolor='#a8d8ea', edgecolor='#0077be',
                     alpha=0.35, linewidth=0.8)

    elevs_arr = np.array(route_elevs)
    norm = Normalize(vmin=elevs_arr.min(), vmax=elevs_arr.max())
    cmap_route = plt.colormaps['RdYlGn_r']

    sc = ax.scatter(rx, ry, c=route_elevs, cmap='RdYlGn_r', s=2, zorder=5,
                    norm=norm, edgecolors='none')
    ax.plot(rx, ry, color='white', linewidth=3.5, alpha=0.5, zorder=4)

    ax.plot(rx[0], ry[0], 'o', color='#2ecc71', markersize=12,
            markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.plot(rx[-1], ry[-1], 's', color='#e74c3c', markersize=12,
            markeredgecolor='white', markeredgewidth=2, zorder=10)

    ax.annotate('Requin', (rx[0], ry[0]), fontsize=8, color='white',
                fontweight='bold', ha='left', va='bottom',
                xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', fc='#2ecc71', alpha=0.8))
    ax.annotate('Aig. du Midi', (rx[-1], ry[-1]), fontsize=8, color='white',
                fontweight='bold', ha='right', va='bottom',
                xytext=(-5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', fc='#e74c3c', alpha=0.8))

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Altitude (m)', fontsize=9)

    dist_km = props.get('distance_km', '?')
    dplus = props.get('dplus_m', '?')
    time_h = props.get('time_tobler_h', '?')
    ax.set_title(f"Route Requin -> Aig. du Midi ({DEM_RESOLUTION}m)\n"
                 f"{dist_km} km | D+ {dplus}m | ~{time_h}h Tobler", fontsize=11)

    ax.set_xlabel('X Lambert-93 (m)')
    ax.set_ylabel('Y Lambert-93 (m)')
    ax.ticklabel_format(style='plain')

    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_150 = os.path.join(FIGURES_DIR, "map_overview.png")
    fig.savefig(out_150, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[png] {out_150}")

    uhd_dir = os.path.join(FIGURES_DIR, "uhd")
    os.makedirs(uhd_dir, exist_ok=True)
    out_uhd = os.path.join(uhd_dir, "map_overview.png")
    fig.savefig(out_uhd, dpi=UHD_DPI, bbox_inches='tight', facecolor='white')
    print(f"[png] {out_uhd} (UHD {UHD_DPI} dpi)")

    plt.close(fig)
    return out_150

def validate_html(html_path, coords_full):
    print("\n--- Validation ---")
    ok = True

    if not os.path.exists(html_path):
        print("  [FAIL] HTML non genere")
        return False

    size = os.path.getsize(html_path)
    if size == 0:
        print("  [FAIL] HTML vide")
        return False

    size_kb = size / 1024
    print(f"  taille: {size_kb:.0f} KB")

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # verif basemaps
    has_osm = 'openstreetmap' in content.lower() or 'tile.openstreetmap' in content
    has_topo = 'opentopomap' in content.lower()
    has_sat = 'arcgisonline' in content.lower() or 'World_Imagery' in content

    n_basemaps = sum([has_osm, has_topo, has_sat])
    print(f"  basemaps: {n_basemaps}/3 (OSM={'ok' if has_osm else 'MISS'}, "
          f"Topo={'ok' if has_topo else 'MISS'}, Sat={'ok' if has_sat else 'MISS'})")

    # verif layers
    has_route = 'Route' in content
    has_glacier = 'Glacier' in content
    has_minimap = 'MiniMap' in content or 'minimap' in content.lower()
    has_fullscreen = 'fullscreen' in content.lower()

    print(f"  layers: route={'ok' if has_route else 'MISS'}, "
          f"glaciers={'ok' if has_glacier else 'MISS'}")
    print(f"  plugins: minimap={'ok' if has_minimap else 'MISS'}, "
          f"fullscreen={'ok' if has_fullscreen else 'MISS'}")

    # verif profil
    has_profile = 'data:image/png;base64' in content
    print(f"  profil altitude: {'ok' if has_profile else 'MISS'}")

    # verif bbox
    lats = [c[1] for c in coords_full]
    lons = [c[0] for c in coords_full]
    lat_ok = 45.8 < min(lats) < 46.0
    lon_ok = 6.8 < min(lons) < 7.0
    print(f"  bbox Chamonix: {'ok' if (lat_ok and lon_ok) else 'WARN'}")

    if n_basemaps < 2:
        ok = False
    if not has_route:
        ok = False

    return ok


def main():
    t0 = time.time()

    print("=" * 55)
    print("T07 - Carte interactive (folium)")
    print(f"  resolution: {DEM_RESOLUTION}m")
    print("=" * 55)

    print("\n--- Chargement GeoJSON WGS84 ---")
    coords_full, props_full = load_geojson("full")
    coords_simp, props_simp = load_geojson("simplified")

    print("\n--- Chargement glaciers RGI ---")
    glaciers_gdf = load_glaciers_wgs84()

    print("\n--- Profil altimetrique (base64) ---")
    profile_b64 = make_elevation_profile_b64(coords_full)

    print("\n--- Construction carte folium ---")
    m = build_map(coords_full, coords_simp, props_full, glaciers_gdf, profile_b64)

    print("\n--- Export HTML ---")
    html_path = export_html(m)

    print("\n--- Export PNG statique ---")
    export_static_png(coords_full, props_full, glaciers_gdf)

    validate_html(html_path, coords_full)

    dt = time.time() - t0
    print(f"\nDone! ({dt:.1f}s)")


if __name__ == "__main__":
    main()
