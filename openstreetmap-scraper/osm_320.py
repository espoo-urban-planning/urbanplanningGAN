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

outdir = "out8"

_dpi = 91

requests_cache.install_cache('scraping_cache', expire_after=(3600 * 24 * 90))

def maparea(area_name, lat=None, lon=None):
  clean_name = str(re.sub(r'[^a-zA-Z\d]', '', area_name))
  nominatim = Nominatim()
  if area_name.isdigit():
    areaId = int(area_name)
  else:
    areaId = nominatim.query(area_name).areaId()


  overpass = Overpass()

  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"landuse"="residential"', out='geom')
  residential_areas = overpass.query(query).ways() or [] # 







  # 8
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="footway"', out='geom')
  roads_footway = overpass.query(query).ways() or []

  # 8
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="cycleway"', out='geom')
  roads_cycleway = overpass.query(query).ways() or []

  # 8
  query = overpassQueryBuilder(area=areaId, elementType='way', selector=['"highway"="service"', '"service"!="parking_aisle"'], out='geom')
  roads_service = overpass.query(query).ways() or []

  # 16
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="residential"', out='geom')
  roads_residential = overpass.query(query).ways() or []

  # 20
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="tertiary"', out='geom')
  roads_tertiary = overpass.query(query).ways() or []

  # 20
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="secondary"', out='geom')
  roads_secondary = overpass.query(query).ways() or []

  # 24
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="primary"', out='geom')
  roads_primary = overpass.query(query).ways() or []

  # 32
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="motorway"', out='geom')
  roads_motorway = overpass.query(query).ways() or [] 

  # 40
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="trunk"', out='geom')
  roads_trunk = overpass.query(query).ways() or [] 

  # 24
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="trunk_link"', out='geom')
  roads_trunk_link = overpass.query(query).ways() or []

  # 20
  query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="unclassified"', out='geom')
  roads_unclassified = overpass.query(query).ways() or []
  
  #print (result.countElements())
  #ax = plt.axes(projection=ccrs.PlateCarree(globe=ccrs.Globe(datum='WGS84',
  #                                              ellipse='WGS84')))
  #print(bbx)

  request = cimgt.OSM()

  terrain = cimgt.MapboxTiles('pk.eyJ1IjoiZWVzYWhlIiwiYSI6ImNqdGFjMTI1ODA1ZGc0NHRmN28wcG5ybnEifQ.ZBSMcdbWkjfjRgDVQLjfnw', 'terrain-rgb')

  plt.style.use('dark_background')


  # 10 inches, 640 pixels, 5038 meters, 1 in = 503.8 m, 1 pt = 1/72 in = 7 m
  fig, ax = plt.subplots(num=1, figsize=(10,10), subplot_kw=dict(projection=request.crs, facecolor='#000000'))

  #print("trying to save debug terrain img")
  #plt.savefig("terrain.png")
  #print("saved")

  #ax.set_extent([26.6561, 22.6589, 59.611, 60.8409])
  #ax.add_image(request, 10, zorder=0)

  roads = roads_footway + roads_cycleway + roads_service + roads_residential + roads_tertiary + roads_secondary + roads_primary + roads_unclassified + roads_trunk + roads_trunk_link

  def roads_in_bbox(bbox):
    ret = []
    for road in roads:
      if road._json['bounds']['minlat'] < bbox['maxlat'] and road._json['bounds']['maxlat'] > bbox['minlat'] and road._json['bounds']['minlon'] < bbox['maxlon'] and road._json['bounds']['maxlon'] > bbox['minlon']:
        ret.append(road)
    return ret

  # find areas completely enclosed inside a bounding box (not partially)
  def residential_areas_enclosed_in_bbox(bbox):
    ret = []
    for area in residential_areas:
      if area._json['bounds']['maxlat'] < bbox['maxlat'] and area._json['bounds']['minlat'] > bbox['minlat'] and area._json['bounds']['maxlon'] < bbox['maxlon'] and area._json['bounds']['minlon'] > bbox['minlon']:
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
  
  print ("Found " + str(len(residential_areas)) + " residential areas in " + area_name)
  
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
  for area in residential_areas:
    #if(lat is not None):
    #      bbx['minlon'] = lon
    #      bbx['maxlon'] = lon
    #      bbx['minlat'] = lat
    #      bbx['maxlat'] = lat
    #      width = 0
    #      height = 0
    #else:
    bbx = area._json['bounds']
    
    width = bbx['maxlon'] - bbx['minlon']
    height = bbx['maxlat'] - bbx['minlat']
    if(width < 0.0008 or height < 0.0008):
      print("area too small, skipping")
      continue
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
    
    targetWidth = 2500
    targetHeight = 2500

    currentWidth = longlen * width
    currentHeight = latlen * height

    offset_w_meters = (targetWidth - currentWidth) / 2
    offset_h_meters = (targetHeight - currentHeight) / 2

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

    test_savename = outdir + "/" + clean_name + "_" + str(counter) + ".png"
    
    # continue if we already created this file
    if os.path.isfile(test_savename):
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

    # get population density

    ring = [
      [new_bbx['minlon'], new_bbx['minlat']],
      [new_bbx['minlon'], new_bbx['maxlat']],
      [new_bbx['maxlon'], new_bbx['maxlat']],
      [new_bbx['maxlon'], new_bbx['minlat']],
      [new_bbx['minlon'], new_bbx['minlat']]
    ]



    ring_string = json.dumps(ring)

    if(lon is None):
      r = requests.post("http://sedac.ciesin.columbia.edu/arcgis/rest/services/sedac/geoprocessing/GPServer/pes-v1/execute", \
        data={'Input_Feature_Set': '{ "geometryType": "esriGeometryPolygon","fields": [{"name": "id","type": "esriFieldTypeInteger"}],"spatialReference": {"wkid": 4326},"features": [{"geometry": {"rings": [  \
        ' + ring_string + ' \
        ],"spatialReference": {"wkid": 4326}},"attributes": {"id": 1}}]}', 'f': 'json'})
      json_result = json.loads(r.text)
      attributes = json_result['results'][0]['value']['features'][0]['attributes']
      pop = attributes['POPULATION05']
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

    fig_terrain, ax_terrain = plt.subplots(figsize=(10,10), subplot_kw=dict(projection=request.crs, facecolor='#000000'))
    
    try:
      ax_terrain.set_extent([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']], crs=ccrs.Geodetic())
    except Exception:
      print(traceback.format_exc())
      print(sys.exc_info()[0])
      continue

    ax_terrain.add_image(terrain, 13)
    savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_t.png"
    
    plt.savefig(savename, dpi=_dpi, transparent=True, bbox_inches='tight', pad_inches=0, frameon=None)
    terrain_raster_646 = cv2.imread(savename)
    terrain_raster_640_b = terrain_raster_646[3:643, 3:643, 0]
    terrain_raster_640_g = terrain_raster_646[3:643, 3:643, 1]
    terrain_raster_640_r = terrain_raster_646[3:643, 3:643, 2]

    plt.figure(2)
    fig2, ax2 = plt.subplots(figsize=(10,10), subplot_kw=dict(projection=request.crs, facecolor='#000000'))
    

    xynps=ax.projection.transform_points(ccrs.Geodetic(), np.asarray([new_bbx['minlon'],new_bbx['maxlon']]), np.asarray([new_bbx['minlat'],new_bbx['maxlat']]))
    #max_lonlat=ax.projection.transform_points(ccrs.Geodetic(), new_bbx['maxlon'], new_bbx['maxlat'])
    

    #ax2.set_extent([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']])
    ax2.set_extent([new_bbx['minlon'], new_bbx['maxlon'], new_bbx['minlat'], new_bbx['maxlat']], crs=ccrs.Geodetic())

    roads_current = roads_in_bbox(new_bbx)
    
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

    mask_646 = cv2.imread(savename, 0)
    mask_640 = mask_646[3:643, 3:643]
    #print("shape:" + str(mask.shape))
    #exit()

    mask_640 = cv2.threshold(mask_640, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
    #kernel = np.ones((3,3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    mask_dilated_640 = cv2.dilate(mask_640, kernel, iterations=2)

    savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_3.png"
    cv2.imwrite(savename, mask_640)
    
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
          'footway': 8,
          'cycleway': 8,
          'service': 12,
          'residential': 12,
          'primary': 24,
          'secondary': 20,
          'tertiary': 16,
          'unclassified': 16,
          'trunk_link': 24,
          'trunk': 40,
          'motorway': 32
      }.get(type, 20)    # 9 is default if x not found


    if len(roads_current) > 0:
      print(len(roads_current))
      processed = 0


      for area in (roads_current):
        _type = area._json['tags']['highway']
        #print(_type)
        if _type not in ['primary', 'secondary', 'tertiary', 'highway', 'trunk']:
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
        linewidth = road_width_meters(_type) / 70

        road_ref = ax.plot(xynps[:,0], xynps[:,1], zorder=11, color='black', marker='o',linewidth=linewidth, markersize=0, antialiased=False)
        road_refs.append(road_ref)
        processed+=1


    savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_0.png"

    plt.savefig(savename, dpi=_dpi, facecolor='0.0', edgecolor='1.0', transparent=True, bbox_inches='tight', pad_inches=0, frameon=None)

    roads_primary_secondary_raster_646 = cv2.imread(savename,0)
    roads_primary_secondary_raster_640 = roads_primary_secondary_raster_646[3:643, 3:643]
    roads_primary_secondary_raster_640 = cv2.threshold(roads_primary_secondary_raster_640, 127, 255, cv2.THRESH_BINARY_INV)[1]

    if len(roads_current) > 0:
      #print(len(roads_current))
      processed = 0
      for area in (roads_current):
        #if(processed % 500 == 0): 
          #print("processing number " + str(processed))

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
        linewidth = road_width_meters(_type) / 70

        road_ref = ax.plot(xynps[:,0], xynps[:,1], zorder=11, color='#000000', marker='o',linewidth=linewidth, markersize=0, antialiased=False)
        road_refs.append(road_ref)
        processed+=1

    
    savename = "osm/" + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_1.png"
    plt.savefig(savename, dpi=_dpi, facecolor='0.0', edgecolor='1.0', transparent=True, bbox_inches='tight', pad_inches=0, frameon=None)

    
    # FINISHED ALL MATPLOTLIB EXPORTS AT THIS POINT


    roads_raster_646 = cv2.imread(savename,0)
    roads_raster_640 = roads_raster_646[3:643, 3:643]
    #b,g,r = cv2.split(masked_roads)''
    roads_raster_640 = cv2.threshold(roads_raster_640, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
    

    # This one will only display the roads that are behind residential areas for the SECOND image

    masked_roads_640 = cv2.bitwise_and(roads_raster_640,roads_raster_640,mask = mask_dilated_640)

    #masked_roads = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary

    #savename = outdir + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_4.png"
    #cv2.imwrite(savename, masked_roads)
    
    # This one will only show the roads behind residential areas

    roads_raster_640 = cv2.bitwise_and(roads_raster_640,roads_raster_640,mask = (255-mask_640))
    roads_raster_640 = cv2.bitwise_or(roads_raster_640, roads_primary_secondary_raster_640)
    
    __width = mask_640.shape[1]
    __height = mask_640.shape[0]
    empty = np.zeros((__height,__width), dtype=np.uint8)

    #print(str(roads_raster.shape))
    #print(str(mask.shape))
    #print(str(empty.shape))

    out=cv2.merge((roads_raster_640,empty,mask_640))

    height = out.shape[0]
    width = out.shape[1]
    #print("width")
    #print(width)
    if width > 640:
        out = out[0:height, 0:640]
        masked_roads = masked_roads[0:height, 0:640]
        width = 640
    if height > 640:
        out = out[0:640, 0:width]
        masked_roads = masked_roads[0:640, 0:width]
        height = 640
    if width < 640 or height < 640:
        width_diff = 640 - width
        height_diff = 640 - height

        out = cv2.copyMakeBorder(out,height_diff,0,width_diff,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        masked_roads = cv2.copyMakeBorder(masked_roads,height_diff,0,width_diff,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        height = out.shape[0]
        width = out.shape[1]

    #savename = outdir + str(re.sub(r'[^a-zA-Z\d]', '', area_name)) + str(counter) + "_5.png"
    #cv2.imwrite(savename, out)

    combined_r = np.zeros((640,640), dtype=np.uint8)
    combined_r[0:640, 0:640] = out[:,:,0]
    combined_r[0:640, 640:640] = masked_roads_640

    combined_g = np.zeros((640,640), dtype=np.uint8)
    #combined_g[0:256, 0:256] = terrain_combined_256
    #combined_g[0:256, 256:512] = masked_roads

    combined_b = np.zeros((640,640), dtype=np.uint8)
    combined_b[0:640, 0:640] = out[:,:,2]
    #combined_b[0:256, 256:512] = masked_roads

    #b,g,r = cv2.split(terrain_raster)

    terrain_combined_640 = -10000 + ((terrain_raster_640_r*256*256 + terrain_raster_640_g*256 + terrain_raster_640_b) * 0.1)
    #water = cv2.inRange(terrain_combined_256, 0, 24467)
    
    smallest = terrain_combined_640.min() #terrain_combined_256.min()
    biggest = smallest + 100 # terrain_combined_256.max()
    #biggest = terrain_combined_256.max()

    _range = biggest - smallest
    #print ("biggest " + str(biggest))
    
    #print ("range " + str(_range))

    terrain_combined_640 = terrain_combined_640 - smallest
    print ("smallest " + str(terrain_combined_640.min()))
    print ("biggest " + str(terrain_combined_640.max()))

    terrain_combined_640 = (terrain_combined_640 / _range) * 255
    #ret,terrain_combined_256 = cv2.threshold(terrain_combined_256,2,255,cv2.THRESH_BINARY)
    
    #terrain_combined_256 = cv2.bitwise_not(terrain_combined_256,terrain_combined_256, mask = water)

    combined_g[0:640, 0:640] = terrain_combined_640
    combined_g[0:640, 0:640] = np.clip(terrain_combined_640, a_min = 0, a_max = 255) 

    #terrain_debug_savename = outdir + "/" + clean_name + "_" + str(counter) + "_" + str(int(density)) + "_tdbg.png"
    #cv2.imwrite(terrain_debug_savename,terrain_new)

    savename = outdir + "/" + clean_name + "_" + str(counter) + "_" + str(int(density)) + ".png"
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
  lat = float(sys.argv[2])
  lon = float(sys.argv[3])
  maparea(area, lat, lon)
else:
  if not area:
      print("Specify an area like 'Kyoto, Japan'")
      exit()
  maparea(area)