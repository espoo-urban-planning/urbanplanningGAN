import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cartopy
import numpy as np 
import matplotlib as mpl        
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

import matplotlib.pyplot as plt
#plt.plot([1,2,3,4])
#plt.ylabel('some numbers')
#plt.show()

"""
Web tile imagery
----------------

This example demonstrates how imagery from a tile
providing web service can be accessed.

"""
import matplotlib.pyplot as plt

from cartopy.io.img_tiles import StamenTerrain


def main():
    terrain = cimgt.MapboxTiles('pk.eyJ1IjoiZWVzYWhlIiwiYSI6ImNqdGFjMTI1ODA1ZGc0NHRmN28wcG5ybnEifQ.ZBSMcdbWkjfjRgDVQLjfnw', 'satellite')
    mycrs = terrain.crs

    go = ccrs.Mercator.GOOGLE

    proj = ccrs.epsg(3857)

    ax = plt.axes(projection=ccrs.PlateCarree())

    deltax = 0.1
    deltay = 0.05
    ax.set_extent([24.8359483-deltax,24.835948+deltax,60.187860-deltay, 60.187860+deltay], crs=go)

	#fig_terrain, ax_terrain = plt.subplots(figsize=(10,10), subplot_kw=dict(projection=cartopy.crs.Mercator(), facecolor='#000000'))
	#ax_terrain.set_extent([60.160964,60.180603,24.818373,24.852362], crs=ccrs.Geodetic())
	#ax_terrain.add_image(terrain, 13)

	#plt.savefig("m.png", transparent=True, bbox_inches='tight', pad_inches=0, frameon=None)

    ax.add_image(terrain, 11)

    #ax.coastlines('10m')
    plt.show()


if __name__ == '__main__':
    main()