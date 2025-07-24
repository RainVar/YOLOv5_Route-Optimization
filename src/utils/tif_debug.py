import rasterio
from rasterio.plot import show
from shapely.geometry import Point
import geopandas as gpd

# Coordinates of Tisa, Cebu City
tisa_coords = (123.871968, 10.299848)  # (lon, lat)

# Load the raster
raster_path = 'data/srtm.tif'
with rasterio.open(raster_path) as src:
    bounds = src.bounds
    print(f"Raster bounds: {bounds}")
    
    # Check if the Tisa point is within raster bounds
    point = Point(tisa_coords)
    if bounds.left <= point.x <= bounds.right and bounds.bottom <= point.y <= bounds.top:
        print("✅ Tisa, Cebu City is within the SRTM raster coverage.")
    else:
        print("❌ Tisa, Cebu City is NOT covered by the SRTM raster.")

    # Optional: plot raster with point
    try:
        import matplotlib.pyplot as plt
        show(src, title="SRTM Raster")
        plt.plot(point.x, point.y, 'ro')  # Mark Tisa point
        plt.show()
    except:
        print("Matplotlib not available or plotting failed.")
