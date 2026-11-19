import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def create_hillshade(heightmap, azimuth=315, altitude=45):
    """Calculate hillshade from a 2D array of elevation values."""
    azimuth = np.radians(azimuth)
    altitude = np.radians(altitude)
    
    dy, dx = np.gradient(heightmap)
    slope = np.pi/2. - np.arctan(np.sqrt(dx*dx + dy*dy))
    aspect = np.arctan2(-dx, dy)
    
    shaded = np.sin(altitude) * np.sin(slope) + \
             np.cos(altitude) * np.cos(slope) * \
             np.cos(azimuth - aspect)
             
    return (shaded - shaded.min()) / (shaded.max() - shaded.min() + 1e-8)

def generate_terrain():
    output_dir = os.path.join('static', 'assets', 'terrain')
    os.makedirs(output_dir, exist_ok=True)
    
    height, width = 600, 800
    
    # 1. Open Terrain (flat, light greens, minimal elevation)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    np.random.seed(42)
    base_noise = np.random.randn(height, width)
    elevation = gaussian_filter(base_noise, sigma=40)
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    hillshade = create_hillshade(elevation)
    
    # Light green colormap blend
    open_cmap = plt.cm.get_cmap('YlGn')
    img = open_cmap(elevation * 0.4 + 0.3)  # light greens
    # Apply hillshade shading
    img[:, :, :3] *= (0.7 + 0.3 * hillshade[:, :, np.newaxis])
    
    ax.imshow(img)
    plt.savefig(os.path.join(output_dir, 'open.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    # 2. Hilly Terrain (browns/greens, high elevation ridges, subtle blue river)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    elevation_hilly = gaussian_filter(np.random.randn(height, width), sigma=15) + \
                      gaussian_filter(np.random.randn(height, width), sigma=35) * 1.5
    elevation_hilly = (elevation_hilly - elevation_hilly.min()) / (elevation_hilly.max() - elevation_hilly.min())
    hillshade_hilly = create_hillshade(elevation_hilly)
    
    hilly_cmap = plt.cm.get_cmap('gist_earth')
    img_hilly = hilly_cmap(elevation_hilly * 0.6 + 0.2)
    img_hilly[:, :, :3] *= (0.6 + 0.4 * hillshade_hilly[:, :, np.newaxis])
    
    # Add a subtle blue river curve across middle
    x = np.linspace(0, 4 * np.pi, width)
    river_y = (height // 2 + np.sin(x) * 60 + np.cos(x*0.5) * 30).astype(int)
    for w_idx in range(width):
        ry = river_y[w_idx]
        for r_offset in range(-8, 9):
            py = ry + r_offset
            if 0 <= py < height:
                dist = abs(r_offset) / 8.0
                alpha = 1.0 - dist
                img_hilly[py, w_idx, 0] = img_hilly[py, w_idx, 0] * (1 - alpha) + 0.1 * alpha  # R
                img_hilly[py, w_idx, 1] = img_hilly[py, w_idx, 1] * (1 - alpha) + 0.3 * alpha  # G
                img_hilly[py, w_idx, 2] = img_hilly[py, w_idx, 2] * (1 - alpha) + 0.7 * alpha  # B
    
    ax.imshow(img_hilly)
    plt.savefig(os.path.join(output_dir, 'hilly.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    # 3. Forest Terrain (deep greens, textured canopy look)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    canopy_noise = np.random.randn(height, width)
    elevation_forest = gaussian_filter(canopy_noise, sigma=5) * 0.5 + gaussian_filter(canopy_noise, sigma=25) * 1.0
    elevation_forest = (elevation_forest - elevation_forest.min()) / (elevation_forest.max() - elevation_forest.min())
    hillshade_forest = create_hillshade(elevation_forest)
    
    forest_cmap = plt.cm.get_cmap('Greens')
    img_forest = forest_cmap(elevation_forest * 0.5 + 0.45)
    img_forest[:, :, :3] *= (0.65 + 0.35 * hillshade_forest[:, :, np.newaxis])
    
    ax.imshow(img_forest)
    plt.savefig(os.path.join(output_dir, 'forest.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    # 4. Desert Terrain (tans/yellows, subtle dune ripples)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    x_grid, y_grid = np.meshgrid(np.linspace(0, 10, width), np.linspace(0, 10, height))
    dunes = np.sin(x_grid * 4 + y_grid * 2) * 0.3 + gaussian_filter(np.random.randn(height, width), sigma=30)
    dunes = (dunes - dunes.min()) / (dunes.max() - dunes.min())
    hillshade_desert = create_hillshade(dunes, azimuth=300, altitude=35)
    
    desert_cmap = plt.cm.get_cmap('YlOrBr')
    img_desert = desert_cmap(dunes * 0.5 + 0.2)
    img_desert[:, :, :3] *= (0.7 + 0.3 * hillshade_desert[:, :, np.newaxis])
    
    ax.imshow(img_desert)
    plt.savefig(os.path.join(output_dir, 'desert.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    # 5. Urban Terrain (grays, grid-like abstract structures)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    grid_img = np.ones((height, width)) * 0.25
    # draw grid lines
    grid_img[::25, :] = 0.5
    grid_img[:, ::25] = 0.5
    # add block elevation / structures
    blocks = np.random.rand(height // 25 + 1, width // 25 + 1)
    for gh in range(0, height, 25):
        for gw in range(0, width, 25):
            val = blocks[gh // 25, gw // 25]
            if val > 0.4:
                grid_img[gh+2:gh+23, gw+2:gw+23] = val * 0.6 + 0.3
                
    grid_img = gaussian_filter(grid_img, sigma=0.8)
    hillshade_urban = create_hillshade(grid_img)
    
    urban_cmap = plt.cm.get_cmap('bone')
    img_urban = urban_cmap(grid_img)
    img_urban[:, :, :3] *= (0.7 + 0.3 * hillshade_urban[:, :, np.newaxis])
    
    ax.imshow(img_urban)
    plt.savefig(os.path.join(output_dir, 'urban.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    # 6. Mixed Terrain (combination of all)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    mixed_elevation = (elevation * 0.3 + elevation_hilly * 0.4 + elevation_forest * 0.3)
    hillshade_mixed = create_hillshade(mixed_elevation)
    
    mixed_cmap = plt.cm.get_cmap('terrain')
    img_mixed = mixed_cmap(mixed_elevation)
    img_mixed[:, :, :3] *= (0.65 + 0.35 * hillshade_mixed[:, :, np.newaxis])
    
    # Add subtle blue river
    for w_idx in range(width):
        ry = river_y[w_idx]
        for r_offset in range(-6, 7):
            py = ry + r_offset
            if 0 <= py < height:
                dist = abs(r_offset) / 6.0
                alpha = 1.0 - dist
                img_mixed[py, w_idx, 0] = img_mixed[py, w_idx, 0] * (1 - alpha) + 0.1 * alpha
                img_mixed[py, w_idx, 1] = img_mixed[py, w_idx, 1] * (1 - alpha) + 0.4 * alpha
                img_mixed[py, w_idx, 2] = img_mixed[py, w_idx, 2] * (1 - alpha) + 0.8 * alpha
                
    ax.imshow(img_mixed)
    plt.savefig(os.path.join(output_dir, 'mixed.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

    print("Terrain PNGs generated successfully in static/assets/terrain/")

if __name__ == '__main__':
    generate_terrain()
