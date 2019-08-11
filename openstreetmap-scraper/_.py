import cv2
import re
import numpy as np

savename = "_test/EspooFinland9_t.png"
    
terrain_raster_261 = cv2.imread(savename)
terrain_raster_256_b = terrain_raster_261[3:259, 3:259, 0]
terrain_raster_256_g = terrain_raster_261[3:259, 3:259, 1]
terrain_raster_256_r = terrain_raster_261[3:259, 3:259, 2]

savename = "_test/EspooFinland9_2.png"

mask_261 = cv2.imread(savename, 0)
mask_256 = mask_261[3:259, 3:259]

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
mask_dilated_256 = cv2.dilate(mask_256, kernel, iterations=2)


savename = "_test/EspooFinland9_0.png"

roads_primary_secondary_raster_261 = cv2.imread(savename,0)
roads_primary_secondary_raster_256 = roads_primary_secondary_raster_261[3:259, 3:259]
roads_primary_secondary_raster_256 = cv2.threshold(roads_primary_secondary_raster_256, 127, 255, cv2.THRESH_BINARY_INV)[1]


savename = "_test/EspooFinland9_1.png"
roads_raster_261 = cv2.imread(savename,0)
roads_raster_256 = roads_raster_261[3:259, 3:259]
#b,g,r = cv2.split(masked_roads)''
roads_raster_256 = cv2.threshold(roads_raster_256, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
masked_roads_256 = cv2.bitwise_and(roads_raster_256,roads_raster_256,mask = mask_dilated_256)

   

__width = mask_256.shape[1]
__height = mask_256.shape[0]
empty = np.zeros((__height,__width), dtype=np.uint8)

#print(str(roads_raster.shape))
#print(str(mask.shape))
#print(str(empty.shape))

out=cv2.merge((roads_raster_256,empty,empty))

#savename = outdir + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_5.png"
#cv2.imwrite(savename, out)

combined_r = np.zeros((256,512), dtype=np.uint8)
combined_r[0:256, 0:256] = out[:,:,0]
combined_r[0:256, 256:512] = masked_roads_256

combined_g = np.zeros((256,512), dtype=np.uint8)
#combined_g[0:256, 0:256] = terrain_combined_256
#combined_g[0:256, 256:512] = masked_roads

combined_b = np.zeros((256,512), dtype=np.uint8)
combined_b[0:256, 0:256] = out[:,:,2]
#combined_b[0:256, 256:512] = masked_roads

#b,g,r = cv2.split(terrain_raster)

terrain_combined_256 = -10000 + ((terrain_raster_256_r*256*256 + terrain_raster_256_g*256 + terrain_raster_256_b) * 0.1)
#water = cv2.inRange(terrain_combined_256, 0, 24467)

smallest = terrain_combined_256.min() #terrain_combined_256.min()
biggest = smallest + 100 # terrain_combined_256.max()
#biggest = terrain_combined_256.max()

_range = biggest - smallest
#print ("biggest " + str(biggest))

#print ("range " + str(_range))

terrain_combined_256 = terrain_combined_256 - smallest
print ("smallest " + str(terrain_combined_256.min()))
print ("biggest " + str(terrain_combined_256.max()))

terrain_combined_256 = (terrain_combined_256 / _range) * 255
#ret,terrain_combined_256 = cv2.threshold(terrain_combined_256,2,255,cv2.THRESH_BINARY)

#terrain_combined_256 = cv2.bitwise_not(terrain_combined_256,terrain_combined_256, mask = water)

combined_g[0:256, 0:256] = terrain_combined_256
combined_g[0:256, 0:256] = np.clip(terrain_combined_256, a_min = 0, a_max = 255) 

#terrain_debug_savename = outdir + "/" + clean_name + "_" + str(counter) + "_" + str(int(density)) + "_tdbg.png"
#cv2.imwrite(terrain_debug_savename,terrain_new)

out=cv2.merge((combined_r,combined_g,combined_b))
print( "writing out file " + savename)
cv2.imwrite("t.png", out)



