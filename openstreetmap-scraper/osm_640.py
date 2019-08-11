from __future__ import division

from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass
from pprint import pprint

import shapely.geometry as sgeom
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cartopy
import numpy as np 
import matplotlib as mpl        
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import re
import weakref
import cv2
import sys
import os
import math
import requests
import requests_cache
import json
import glob

outdir = "out9"

_dpi = 67

featured_areas = {}


bbb = [60.149687,24.793653,60.187431,24.870901]

requests_cache.install_cache('scraping_cache', expire_after=(3600 * 24 * 90))

def maparea(area_name, offset=None):

  clean_name = str(re.sub(r'[^a-zA-Z\d]', '', area_name))
  nominatim = Nominatim()
  if area_name.isdigit():
    areaId = int(area_name)
  else:
    areaId = nominatim.query(area_name).areaId()


  overpass = Overpass()

  query = overpassQueryBuilder(bbox=bbb, elementType='way', selector='"landuse"="residential"', out='geom')
  residential_areas = [] #overpass.query(query, timeout=60).ways() or [] # 







  # 8
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="footway"', out='geom')
  roads_footway = overpass.query(query, timeout=60).ways() or []

  # 8
  query = overpassQueryBuilder(bbox=bbb, elementType='way', selector='"highway"="cycleway"', out='geom')
  roads_cycleway = overpass.query(query).ways() or []

  # 8
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector=['"highway"="service"', '"service"!="parking_aisle"'], out='geom')
  roads_service = overpass.query(query, timeout=60).ways() or []

  # 16
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="residential"', out='geom')
  roads_residential = overpass.query(query).ways() or []

  # 20
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="tertiary"', out='geom')
  roads_tertiary = overpass.query(query).ways() or []

  # 20
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="secondary"', out='geom')
  roads_secondary = overpass.query(query).ways() or []

  # 24
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="primary"', out='geom')
  roads_primary = overpass.query(query).ways() or []

  # 32
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="motorway"', out='geom')
  roads_motorway = overpass.query(query).ways() or [] 

  # 40
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="trunk"', out='geom')
  roads_trunk = overpass.query(query).ways() or [] 

  # 24
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="trunk_link"', out='geom')
  roads_trunk_link = overpass.query(query).ways() or []

  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="motorway_link"', out='geom')
  motorway_link = overpass.query(query).ways() or []

  # 20
  query = overpassQueryBuilder(bbox=bbb,  elementType='way', selector='"highway"="unclassified"', out='geom')
  roads_unclassified = overpass.query(query).ways() or []
  
  #print (result.countElements())
  #ax = plt.axes(projection=ccrs.PlateCarree(globe=ccrs.Globe(datum='WGS84',
  #                                              ellipse='WGS84')))
  #print(bbx)

  request = cimgt.OSM()

  terrain = cimgt.MapboxTiles('pk.eyJ1IjoiZWVzYWhlIiwiYSI6ImNqdGFjMTI1ODA1ZGc0NHRmN28wcG5ybnEifQ.ZBSMcdbWkjfjRgDVQLjfnw', 'terrain-rgb')

  plt.style.use('dark_background')


  # 10 inches, 512 pixels, 4000 meters, 1 in = 400 m, 1 pt = 1/72 in = 6
  fig, ax = plt.subplots(num=1, figsize=(10,10), subplot_kw=dict(projection=request.crs, facecolor='#000000'))

  #print("trying to save debug terrain img")
  #plt.savefig("terrain.png")
  #print("saved")

  #ax.set_extent([26.6561, 22.6589, 59.611, 60.8409])
  #ax.add_image(request, 10, zorder=0)

  roads = motorway_link + roads_footway + roads_cycleway + roads_service + roads_residential + roads_tertiary + roads_secondary + roads_primary + roads_unclassified + roads_trunk + roads_motorway + roads_trunk_link
  print(roads)

  def roads_in_bbox(bbox):
    ret = []
    for road in roads:
      if road._json['bounds']['minlat'] < bbox['maxlat'] and road._json['bounds']['maxlat'] > bbox['minlat'] and road._json['bounds']['minlon'] < bbox['maxlon'] and road._json['bounds']['maxlon'] > bbox['minlon']:
        ret.append(road)
    return ret

  # find areas completely enclosed inside a bounding box (not partially)
  def residential_areas_enclosed_in_bbox(bbox):
    ret = []
    for area_id, area in enumerate(residential_areas):
      if area._json['bounds']['maxlat'] < bbox['maxlat'] and area._json['bounds']['minlat'] > bbox['minlat'] and area._json['bounds']['maxlon'] < bbox['maxlon'] and area._json['bounds']['minlon'] > bbox['minlon']:
        featured_areas[area_id] = True
        ret.append(area)
    return ret

  cll = []

  bbx = None
  '''

  processed = 0
  roads = roads_residential + roads_tertiary + roads_secondary + roads_primary + roads_unclassified + roads_trunk + roads_trunk_link
  print("processing total " + str(len(roads)) + " roads")
  for area in (roads_residential + roads_tertiary + roads_secondary + roads_primary + roads_unclassified + roads_trunk + roads_trunk_link):
    if(processed % 500 == 0): 
      print("processing number " + str(processed))

    geometry = area._json['geometry']
    lons = []
    lats = []
    for point in geometry:
        lons.append(point['lon'])
        lats.append(point['lat'])
        #mycoords.append( (point['lat'], ) )

    xs = np.array(lons)
    ys = np.array(lats)
    xynps=ax.projection.transform_points(ccrs.Geodetic(), xs, ys)
    #print(xynps)
    #break
    ax.plot(xynps[:,0], xynps[:,1], "r", zorder=11, color='green', marker='o',linewidth=0.2, markersize=0, antialiased=False)
    processed+=1

  '''
  
  #print ("Found " + str(len(residential_areas)) + " residential areas in " + area_name)
  
  # What are we exactly doing here??

  '''
  for area in residential_areas:
    #print(vars(area))
    geometry = area._json['geometry']
    bbx = area._json['bounds']
    lons = []
    lats = []
    for point in geometry:
        lons.append(point['lon'])
        lats.append(point['lat'])
        #mycoords.append( (point['lat'], ) )

    xs = np.array(lons)
    ys = np.array(lats)
    xynps=ax.projection.transform_points(ccrs.Geodetic(), xs, ys)
    #print(xynps)
    #break
    plt.figure(1)
    ax.fill(xynps[:,0], xynps[:,1], "b", zorder=10, antialiased=False)

    cll.append( (lons,lats) )
  #print(len(cll))

  #ax.axis('off')
  '''


  counter = 0
  for area_id, area in enumerate(['a']):
    if(offset != None and area_id < offset):
      print("off")
      continue

    if(area_id in featured_areas):
      print("already featured")
      continue
    #if(lat is not None):
    #      bbx['minlon'] = lon
    #      bbx['maxlon'] = lon
    #      bbx['minlat'] = lat
    #      bbx['maxlat'] = lat
    #      width = 0
    #      height = 0
    #else:
    #bbx = area._json['bounds']
    bbx = {
      'minlat': 60.171382,
      'maxlat': 60.171382,
      'minlon': 24.833822,
      'maxlon': 24.833627,
    }


    
    width = bbx['maxlon'] - bbx['minlon']
    height = bbx['maxlat'] - bbx['minlat']
    #if(width < 0.0008 or height < 0.0008):
    #  #print("area too small, skipping")
    #  continue
    #print(width)
    #print(height)
    
    zoom = 0.7

    conv_factor = (2.0 * math.pi)/360.0;
    lat = bbx['minlat']
    lat = lat * conv_factor

    m1 = 111132.92;     # latitude calculation term 1
    m2 = -559.82;       # latitude calculation term 2
    m3 = 1.175;         # latitude calculation term 3
    m4 = -0.0023;       # latitude calculation term 4
    p1 = 111412.84;     # longitude calculation term 1
    p2 = -93.5;         # longitude calculation term 2
    p3 = 0.118;         # longitude calculation term 3

    # Calculate the length of a degree of latitude and longitude in meters
    latlen = m1 + (m2 * math.cos(2 * lat)) + (m3 * math.cos(4 * lat)) + \
            (m4 * math.cos(6 * lat))
    longlen = (p1 * math.cos(lat)) + (p2 * math.cos(3 * lat)) + \
                (p3 * math.cos(5 * lat))

    #print("lat len " + str(latlen))
    #print("lon len " + str(longlen))
    
    targetWidth = 1536
    targetHeight = 1536

    currentWidth = longlen * width
    currentHeight = latlen * height

    print("currentWidth " + str(currentWidth))
    print("currentHeight " + str(currentHeight))

    offset_w_meters = (targetWidth - currentWidth) / 2
    offset_h_meters = (targetHeight - currentHeight) / 2

    print("we want to add " + str(offset_w_meters*2) + " on width ")

    offset_w_angles = offset_w_meters / longlen
    offset_h_angles = offset_h_meters / latlen

    #print("currentWidth " + str(currentWidth))
    #print("currentHeight " + str(currentHeight))
    #print("offsetAngleW " + str(offset_w_angles))
    #print("offsetAngleH " + str(offset_h_angles))

    offsetW = offset_w_angles
    offsetH = offset_h_angles

    #my_cos=math.cos(rad)
    #print("my cos " + str(my_cos))
    #test = 0.1 * abs(math.cos(abs(bbx['minlat'])))
    #print("test " + str(test))
    

    #print("trying to make it " + str(zoom*test) + " degrees wide")

    #print("testOffsetW" + str(testOffsetW))

    #offsetW = ((zoom*0.051122172576223) - width) / 2
    #print("realoffsetW" + str(offsetW))


    #offsetH = ((zoom*0.038) - height) / 2
    #print("offsetH" + str(offsetH))
    
    if offsetW < 0 or offsetH < 0:
      print("area too big, skipping")
      continue

    test_savename = outdir + "/" + clean_name + "__" + str(area_id) + ".png"
    
    # continue if we already created this file
    if False and len(glob.glob(test_savename)) > 0:

      counter += 1
      continue


    #print(bbx)

    new_bbx = bbx.copy()
    try:
      new_bbx['minlon'] = bbx['minlon'] - offsetW
      new_bbx['maxlon'] = bbx['maxlon'] + offsetW
      new_bbx['minlat'] = bbx['minlat'] - offsetH
      new_bbx['maxlat'] = bbx['maxlat'] + offsetH
    except:
      print("FAILED, BBX: " + str(bbx))
      pprint(area._json)
      exit()

    print(new_bbx)

    # get population density

    ring = [
      [new_bbx['minlon'], new_bbx['minlat']],
      [new_bbx['minlon'], new_bbx['maxlat']],
      [new_bbx['maxlon'], new_bbx['maxlat']],
      [new_bbx['maxlon'], new_bbx['minlat']],
      [new_bbx['minlon'], new_bbx['minlat']]
    ]



    ring_string = json.dumps(ring)

    if(False):
      r = requests.post("http://sedac.ciesin.columbia.edu/arcgis/rest/services/sedac/geoprocessing/GPServer/pes-v2/execute", \
        data={'Input_Feature_Set': '{ "geometryType": "esriGeometryPolygon","fields": [{"name": "id","type": "esriFieldTypeInteger"}],"spatialReference": {"wkid": 4326},"features": [{"geometry": {"rings": [  \
        ' + ring_string + ' \
        ],"spatialReference": {"wkid": 4326}},"attributes": {"id": 1}}]}', 'f': 'json'})
      json_result = json.loads(r.text)

      #print(json_result)
      attributes = json_result['results'][0]['value']['features'][0]['attributes']
      pop = attributes['POPULATION']
      area = attributes['LANDAREA']
      density = pop / area


      if(density < 1000):
        print("density too small")
        continue




    #print(new_bbx)
    #exit()

    #print("bbx height " + str(new_bbx['maxlat'] - new_bbx['minlat']))

    #red_fill = ax.fill(xynps[:,0], xynps[:,1], "r", zorder=10, antialiased=False)

    plt.figure("terrain")
    

    #ax_terrain = plt.axes(projection=mycrs)
    fig_terrain, ax_terrain = plt.subplots(figsize=(10,10), subplot_kw=dict(projection=terrain.crs, facecolor='#000000'))
    
    #try:
      #ax_terrain.set_extent([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']], crs=ccrs.Geodetic())
    #except Exception:
    #  print(traceback.format_exc())
    #  print(sys.exc_info()[0])
    #  continue
    print("hello")

    #ax_terrain.set_extent([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']])
    #ax_terrain.add_image(terrain, 13)

    #print([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']])
    #ax_terrain.set_extent([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']], crs=ccrs.Geodetic())

    savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_t.png"
    
    plt.savefig(savename, dpi=_dpi, transparent=True, bbox_inches='tight', pad_inches=0, frameon=None)
    terrain_raster_515 = cv2.imread(savename)
    terrain_raster_512_b = terrain_raster_515[3:515, 3:515, 0]
    terrain_raster_512_g = terrain_raster_515[3:515, 3:515, 1]
    terrain_raster_512_r = terrain_raster_515[3:515, 3:515, 2]

    plt.figure(2)
    fig2, ax2 = plt.subplots(figsize=(10,10), subplot_kw=dict(projection=request.crs, facecolor='#000000'))
    

    xynps=ax.projection.transform_points(ccrs.Geodetic(), np.asarray([new_bbx['minlon'],new_bbx['maxlon']]), np.asarray([new_bbx['minlat'],new_bbx['maxlat']]))
    #print(xynps) 
    #exit()
    #max_lonlat=ax.projection.transform_points(ccrs.Geodetic(), new_bbx['maxlon'], new_bbx['maxlat'])
    

    #ax2.set_extent([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']])
    ax2.set_extent([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']], crs=ccrs.Geodetic())

    roads_current = roads_in_bbox(new_bbx)
    print("num roads " + str(roads))
    
    all_areas = residential_areas_enclosed_in_bbox(new_bbx)
    #print(str(len(all_areas)) + " areas in bbox")
    red_fills = []
    for current_residential_area in (all_areas):
        bbx = current_residential_area._json['bounds']
        width = bbx['maxlon'] - bbx['minlon']
        height = bbx['maxlat'] - bbx['minlat']
        if(width < 0.001 or height < 0.001):
          #print("area too small, skipping")
          continue

        geometry = current_residential_area._json['geometry']
        lons = []
        lats = []
        for point in geometry:
          lons.append(point['lon'])
          lats.append(point['lat'])
          #mycoords.append( (point['lat'], ) )

        xs = np.array(lons)
        ys = np.array(lats)
        xynps=ax.projection.transform_points(ccrs.Geodetic(), xs, ys)

        #print("area " + str(current_residential_area))
        red_fill2 = ax2.fill(xynps[:,0], xynps[:,1], "k", zorder=11, antialiased=False)
        red_fills.append(red_fill2)

    road_refs = []
    savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_2.png"
    #print("trying to save file called " +savename)
    plt.savefig(savename, dpi=_dpi, facecolor='0.0', edgecolor='0.0', transparent=True, bbox_inches='tight', pad_inches=0, frameon=None)

    #exit()

    mask_515 = cv2.imread(savename, 0)
    mask_512 = mask_515[3:515, 3:515]
    #print("shape:" + str(mask.shape))
    #exit()

    mask_512 = cv2.threshold(mask_512, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
    #kernel = np.ones((3,3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))


    savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_3.png"
    cv2.imwrite(savename, mask_512)
    
    #for road_ref in (road_refs):
    #  l = road_ref.pop(0)
    #  wl = weakref.ref(l)
    #  l.remove()
    #  del l

    plt.figure(1)


    ax.set_extent([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']], crs=ccrs.Geodetic())
    

    #print("fetch roads")
    #print("got " + str(len(roads))+ " roads ")
    #exit()

    #query = overpassQueryBuilder(bbox=[new_bbx['minlat'], new_bbx['minlon'], new_bbx['maxlat'], new_bbx['maxlon']], elementType='way', selector=['"highway"="residential"', '"highway"="tertiary"', '"highway"="secondary"', '"highway"="primary"', '"highway"="unclassified"', '"highway"="trunk"', '"highway"="trunk_link"'], out='geom')
    #roads_current = overpass.query(query).ways()
    #print("got num roads:")

    def road_width_meters(type):
      return {
          'footway': 0.01,
          'cycleway': 0.01,
          'service': 0.01,
          'residential': 1,
          'primary': 3.1,
          'secondary': 2.1,
          'tertiary': 1.1,
          'unclassified': 2.1,
          'trunk_link': 2.1,
          'motorway_link': 2.1,
          'trunk': 4.1,
          'motorway': 4.1
      }.get(type, 2)    # 9 is default if x not found


    if len(roads_current) > 0:
      print(len(roads_current))
      processed = 0


      for area in (roads_current):
        _type = area._json['tags']['highway']
        #print(_type)
        if _type not in ['primary', 'secondary', 'tertiary', 'highway', 'motorway', 'trunk']:
            continue

        geometry = area._json['geometry']
        lons = []
        lats = []
        for point in geometry:
          lons.append(point['lon'])
          lats.append(point['lat'])
          #mycoords.append( (point['lat'], ) )

        xs = np.array(lons)
        ys = np.array(lats)
        xynps=ax.projection.transform_points(ccrs.Geodetic(), xs, ys)
        #print(xynps)
        #break
        plt.figure(1)

        linewidth = road_width_meters(_type)

        road_ref = ax.plot(xynps[:,0], xynps[:,1], snap=False, zorder=11, color='black', marker='o',linewidth=linewidth, markersize=0, antialiased=False)
        road_refs.append(road_ref)
        processed+=1


    savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_0.png"

    plt.savefig(savename, dpi=_dpi, facecolor='0.0', edgecolor='1.0', transparent=True, bbox_inches='tight', pad_inches=0, frameon=None)

    roads_primary_secondary_raster_515 = cv2.imread(savename,0)
    roads_primary_secondary_raster_512 = roads_primary_secondary_raster_515[3:515, 3:515]
    roads_primary_secondary_raster_512 = cv2.threshold(roads_primary_secondary_raster_512, 127, 255, cv2.THRESH_BINARY_INV)[1]

    if len(roads_current) > 0:
      #print(len(roads_current))
      processed = 0
      for area in (roads_current):
        #if(processed % 500 == 0): 
          #print("processing number " + str(processed))

        geometry = area._json['geometry']
        _type = area._json['tags']['highway']

        lons = []
        lats = []
        for point in geometry:
          lons.append(point['lon'])
          lats.append(point['lat'])
          #mycoords.append( (point['lat'], ) )

        xs = np.array(lons)
        ys = np.array(lats)
        xynps=ax.projection.transform_points(ccrs.Geodetic(), xs, ys)
        #print(xynps)
        #break
        plt.figure(1)
        linewidth = road_width_meters(_type)

        print(str(_type) + ": " + str(linewidth))

        road_ref = ax.plot(xynps[:,0], xynps[:,1], snap=False, zorder=11, color='#000000', marker='o',linewidth=linewidth, markersize=0, antialiased=False)
        road_refs.append(road_ref)
        processed+=1

    savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_1.png"
    plt.savefig(savename, dpi=_dpi, facecolor='0.0', edgecolor='1.0', transparent=True, bbox_inches='tight', pad_inches=0, frameon=None)

    # FINISHED ALL MATPLOTLIB EXPORTS AT THIS POINT


    roads_raster_515 = cv2.imread(savename,0)
    roads_raster_512 = roads_raster_515[3:515, 3:515]
    #b,g,r = cv2.split(masked_roads)''
    roads_raster_512 = cv2.threshold(roads_raster_512, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
    

    cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    ones = np.ones((3,3))

    r1_dilated = cv2.dilate(mask_512,ones,iterations = 4)
    r1_hacked = cv2.erode(r1_dilated,ones,iterations = 4)

    # SECOND IMAGE
    # This one will only display the roads that are behind residential areas
    masked_roads_512 = cv2.bitwise_and(roads_raster_512,roads_raster_512, mask=r1_dilated)

    #masked_roads = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary

    #savename = outdir + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_4.png"
    #cv2.imwrite(savename, masked_roads)
    
    # This one will hide roads
    roads_raster_512 = cv2.bitwise_and(roads_raster_512,roads_raster_512,mask = (255-r1_hacked))
    roads_raster_512 = cv2.bitwise_or(roads_raster_512, roads_primary_secondary_raster_512)
    
    __width = mask_512.shape[1]
    __height = mask_512.shape[0]
    empty = np.zeros((__height,__width), dtype=np.uint8)

    #print(str(roads_raster.shape))
    #print(str(mask.shape))
    #print(str(empty.shape))

    out=cv2.merge((roads_raster_512,empty,r1_hacked))

    height = out.shape[0]
    width = out.shape[1]
    #print("width")
    #print(width)
    if width > 512:
        out = out[0:height, 0:512]
        masked_roads = masked_roads[0:height, 0:512]
        width = 512
    if height > 512:
        out = out[0:512, 0:width]
        masked_roads = masked_roads[0:512, 0:width]
        height = 512
    if width < 512 or height < 512:
        width_diff = 512 - width
        height_diff = 512 - height

        out = cv2.copyMakeBorder(out,height_diff,0,width_diff,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        masked_roads = cv2.copyMakeBorder(masked_roads,height_diff,0,width_diff,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        height = out.shape[0]
        width = out.shape[1]

    #savename = outdir + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_5.png"
    #cv2.imwrite(savename, out)

    combined_r = np.zeros((512,1024), dtype=np.uint8)
    combined_r[0:512, 0:512] = out[:,:,0]
    combined_r[0:512, 512:1024] = masked_roads_512

    combined_g = np.zeros((512,1024), dtype=np.uint8)
    #combined_g[0:256, 0:256] = terrain_combined_256
    #combined_g[0:256, 256:512] = masked_roads

    combined_b = np.zeros((512,1024), dtype=np.uint8)
    combined_b[0:512, 0:512] = out[:,:,2]
    #combined_b[0:256, 256:512] = masked_roads

    #b,g,r = cv2.split(terrain_raster)

    terrain_combined_512 = -10000 + ((terrain_raster_512_r*256*256 + terrain_raster_512_g*256 + terrain_raster_512_b) * 0.1)
    #water = cv2.inRange(terrain_combined_256, 0, 24467)
    
    smallest = terrain_combined_512.min() #terrain_combined_256.min()
    biggest = smallest + 100 # terrain_combined_256.max()
    #biggest = terrain_combined_256.max()

    _range = biggest - smallest
    #print ("biggest " + str(biggest))
    
    #print ("range " + str(_range))

    terrain_combined_512 = terrain_combined_512 - smallest
    print ("smallest " + str(terrain_combined_512.min()))
    print ("biggest " + str(terrain_combined_512.max()))

    terrain_combined_512 = (terrain_combined_512 / _range) * 255
    #ret,terrain_combined_256 = cv2.threshold(terrain_combined_256,2,255,cv2.THRESH_BINARY)
    
    #terrain_combined_256 = cv2.bitwise_not(terrain_combined_256,terrain_combined_256, mask = water)

    combined_g[0:512, 0:512] = terrain_combined_512
    combined_g[0:512, 0:512] = np.clip(terrain_combined_512, a_min = 0, a_max = 255) 

    #terrain_debug_savename = outdir + "/" + clean_name + "_" + str(counter) + "_" + str(int(density)) + "_tdbg.png"
    #cv2.imwrite(terrain_debug_savename,terrain_new)

    savename = outdir + "/" + clean_name + "__" + str(area_id) + ".png"
    out=cv2.merge((combined_r,combined_g,combined_b))
    print( "writing out file " + savename)
    cv2.imwrite(savename, out)
    
    for cur_red_fill in (red_fills):
      l = cur_red_fill.pop(0)
      wl = weakref.ref(l)
      l.remove()
      del l

    #red_fill2 = ax.fill(xynps[:,0], xynps[:,1], "r", zorder=15, antialiased=False)
  
    #savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_2.png"
    #plt.savefig(savename, dpi=128, facecolor='0.0', edgecolor='0.0', transparent=False, bbox_inches='tight', pad_inches=0, frameon=None)


    counter +=1
    #l = red_fill.pop(0)
    #wl = weakref.ref(l)
    #l.remove()
    #del l

    #counter +=1
    #l = red_fill2.pop(0)
    #wl = weakref.ref(l)
    #l.remove()
    #del l

    for road_ref in (road_refs):
      l = road_ref.pop(0)
      wl = weakref.ref(l)
      l.remove()
      del l

    #ax.fill(xynps[:,0], xynps[:,1], "b", zorder=10, antialiased=False)

  #ax.set_extent([bbx['minlat'], bbx['maxlat'], bbx['minlon'], bbx['maxlon']])
  '''
  #for _area in cll:
    #print("trying to fill area with")
    #print(_area[0])
    #print(_area[1])
    xs = np.array(_area[0])
    ys = np.array(_area[1])

    xynps=ax.projection.transform_points(ccrs.Geodetic(), xs, ys)
    #print(xynps)
    #break
    ax.fill(xynps[:,0], xynps[:,1], "b", zorder=10)
    #break
    #plt.plot(_area[1], _area[0], color='blue', linewidth=0.2, marker=',', transform=ccrs.Geodetic() )

  #ax.add_geometries([track], ccrs.Mercator(),
  #                      facecolor='red')
  '''
  #print (mycoords)
  #ax.add_geometries([mycoords],
  #                  ccrs.Mercator(),
  #                  facecolor='blue', hatch='xxxx')
  #ax.add_feature(cartopy.feature.ShapelyFeature(mycoords), facecolor='blue')

  #savename = str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + ".png"
  #plt.savefig(savename, dpi=128, facecolor='w', edgecolor='w',
  #        orientation='portrait', format=None,
  #        transparent=False, bbox_inches=0, pad_inches=0,
  #        frameon=None)
          
#plt.show()

area = sys.argv[1]
if(len(sys.argv) > 2):
  offset = float(sys.argv[2])
  maparea(area, offset)
else:
  if not area:
      print("Specify an area like 'Kyoto, Japan'")
      exit()
  maparea(area)