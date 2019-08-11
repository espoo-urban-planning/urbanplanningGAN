import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy
import math

img = cv2.imread('thinning.png',0)

nodes = []
nodeMap = np.empty([512, 512], dtype=int)
edgesOfNodes = []
edges = []



#I need list of all edges with duplicates
#[start id] [end id]

#each edge can have connections to multiple other edges...
# ok


def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1

    return [ 
        (x_1, y), 
        (x_1, y1), 
        (x, y1), 
        (x1, y1),     # P2,P3,P4,P5
        (x1, y), 
        (x1, y_1), 
        (x, y_1), 
        (x_1, y_1)
    ]    # P6,P7,P8,P9


for y in range(0, 512):
    for x in range(0, 512):
        # threshold the pixel
        pixel = img[y,x]
        if pixel == 255:
            curNodeId = len(nodes)
            nodeMap[x,y] = curNodeId
            nodes.append( (x,y) )
            edgesOfNodes.append( [] )

for curNodeId, node in enumerate(nodes):
    x = node[0]
    y = node[1]
    #print("NODE NUMBER " + str(curNodeId))

    for nb in neighbours(x,y,img):
        if(img[nb[1], nb[0]] == 255):
            #print("found neigbour of " + str(x) + " " + str(y) + " : " + str(nb))
            theOtherNode = nodeMap[nb[0], nb[1]]
            #print("coords of theOtherNode:" + str(nodes[theOtherNode]))

            edge = [curNodeId,theOtherNode];
            #print("adding edge" + str(edge))
            edges.append(edge)
            #print("added edge count of edges: " + str(len(edges) - 1))

            #if not curNodeId in edgesOfNodes:
            #    edgesOfNodes[curNodeId] = []
            edgesOfNodes[curNodeId].append(len(edges) - 1)

            #if(curNodeId > 200):
                #print(edgesOfNodes)
                #exit()
            
            #print("appending edge " + str(nodes[theOtherNode]) + " - " + str(node) )
            #smarterEdges.append( [curNodeId,theOtherNode] )
            #print("ids are " + str([curNodeId,theOtherNode]) )

foundFaces = []
currentFaceBeingBuilt = []

visitedEdges = {}
for curEdgeId, curEdge in enumerate(edges):
    startingNodeId = curEdge[0]
    startingPoint = nodes[startingNodeId]

    #print("starting point: " + str(startingPoint))

    if(curEdgeId in visitedEdges): continue

    print("first node coordinates " + str(nodes[curEdge[0]]))
    currentFaceBeingBuilt.append(curEdge[0])


    visitedEdges[curEdgeId] = True

    #print(curEdge)
    nextNode = curEdge[1]
    #print("next node has id " + str(curEdge[1]))

    nextNodeCoords = nodes[curEdge[1]]
    #print("looks like " + str(nextNodeCoords))
    #print(nextNode)

    prevNode = startingPoint
    #print("new cycle")
    for k in range(1000):
        #print("K " + str(k))
        if(nextNode == startingNodeId):
            #print("found start")
            foundFaces.append(currentFaceBeingBuilt)
            currentFaceBeingBuilt = []
            break
        print("next node coordinates " + str(nodes[nextNode]))
        currentFaceBeingBuilt.append(nextNode)
        node1 = nodes[nextNode]
        #print(node1)
        nextNodeIds = edgesOfNodes[nextNode]
        

        #print("that is connected to nodes with ids " + str(nextNodeIds))
        
        

        largestAngle = None
        largestEdgeId = None

        for potentialNextEdgeId in nextNodeIds:
            if(potentialNextEdgeId in visitedEdges): continue

            v1 = numpy.subtract(node1, prevNode)
            edge2 = edges[potentialNextEdgeId]
            node2 = nodes[edge2[1]]
            v2 = numpy.subtract(node2, node1)

            dot = v1[0]*v2[0] + v1[1]*v2[1] # dot product
            det = v1[0]*v2[1] - v1[1]*v2[0] # determinant
            

            angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

            #print("round angle " + str(round(angle,5)))
            if(round(angle,5) == round(math.pi,5)): 
                continue
                #print("wtf")
                #print("s1 " + str(startingPoint))
                #print("dot " + str(dot))
                #print("det " + str(det))
                #print("v1 " + str(v1))
                #print("n1 " + str(node1))
                #print("n2 " + str(node2))
                #print("v2 " + str(v2))
                #print("angle " + str(angle))
            if(angle > largestAngle): 
                #print(prevNode)
                #print(node1)
                #print(node2)
                #print("largest " + str(largestAngle))
                #print("edge we are checking out " + str((nodes[edge2[0]], nodes[edge2[1]])))
                #print("s1 " + str(startingPoint))
                #print("v1 " + str(v1))
                #print("n1 " + str(node1))
                #print("n2 " + str(node2))
                #print("v2 " + str(v2))
                #print("angle " + str(angle))
                largestAngle = angle
                largestEdgeId = potentialNextEdgeId
                nextNode = edge2[1]

            #print("s1 " + str(startingPoint))
            #print("v1 " + str(v1))
            #print("n1 " + str(node1))
            #print("n2 " + str(node2))
            #print("v2 " + str(v2))
            #print("angle " + str(angle))

        visitedEdges[largestEdgeId] = True
        prevNode = node1
        #print("k is " + str(k))

        #print("started from " + str(startingNodeId))
        #print("next node: " + str(nextNode))


        
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

fig, ax = plt.subplots()

import array 
for face in foundFaces:
    #print("face iteration")
    #print(numpy.asarray(nodes[node]))
    l = []
    for nodeId in face:
        node = nodes[nodeId]
        #print("coords of node " + str(node))
        l.append( [ node[0] , node[1]] )
        #print(nodes[node])

    if(len(face) == 3):
        area = 0.5 * ((l[1][0]-l[0][0])*(l[2][1]-l[0][1])-(l[2][0]-l[0][0])*(l[1][1]-l[0][1]))   
        #print("triangle area " + str(area))
        if area == 0.5: continue
    
    xs, ys = zip(*l)

    ax.fill(xs,ys,edgecolor='black') 

    #print(__list)
    #polygon = Polygon(__list, True)
    #patches.append(polygon)
    #print face

plt.gca().invert_yaxis()
#plt.show()

import osgeo.ogr, osgeo.osr #we will need some packages
from osgeo import ogr #and one more for the creation of a new field
spatialReference = osgeo.osr.SpatialReference() #will create a spatial reference locally to tell the system what the reference will be
spatialReference.ImportFromEPSG(4326) #here we define this reference to be wgs84..
driver = osgeo.ogr.GetDriverByName('ESRI Shapefile') # will select the driver foir our shp-file creation.
shapeData = driver.CreateDataSource('/usr/local/Cellar/nginx/1.15.2/html/n/result.shp') #so there we will store our data
layer = shapeData.CreateLayer('Example', spatialReference, osgeo.ogr.wkbPoint) #this will create a corresponding layer for our data with given spatial information.
layer_defn = layer.GetLayerDefn()

layer = shapeData.CreateLayer('layer', spatialReference, osgeo.ogr.wkbPoint) #this will create a corresponding layer for our data with given spatial information.
layer_defn = layer.GetLayerDefn() # gets parameters of the current shapefile
index = 0
 

readerDict = csv.DictReader(csvfile, delimiter=delimiter)
for field in ['foo', 'bar']:
    new_field = ogr.FieldDefn(field, ogr.OFTString) #we will create a new field with the content of our header
    layer.CreateField(new_field)

row = {'LAT': 20, 'LON': 30, 'foo': 'lol', 'bar': 'hah' }

print(row['LAT'], row['LON'])
point = osgeo.ogr.Geometry(osgeo.ogr.wkbPoint)
point.AddPoint(float(row['LON']), float(row['LAT'])) #we do have LATs and LONs as Strings, so we convert them
feature = osgeo.ogr.Feature(layer_defn)
feature.SetGeometry(point) #set the coordinates
feature.SetFID(index)
for field in readerDict.fieldnames:
    i = feature.GetFieldIndex(field)
    feature.SetField(i, row[field])
layer.CreateFeature(feature)
#index += 1
shapeData.Destroy() #lets close the shapefile

#for i in range(num_polygons):
#    polygon = Polygon(np.random.rand(num_sides ,2), True)
#    patches.append(polygon)

#p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

#colors = 100*np.random.rand(len(patches))
#p.set_array(np.array(colors))

#ax.add_collection(p)

#plt.show()
