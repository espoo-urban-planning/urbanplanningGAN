import cv2
import numpy 
import matplotlib
import matplotlib.pyplot as plt
import math
from math import *
import sys


def create_dxf(filename, outname):
    
    img = cv2.imread(filename,0)

    nodes = []
    nodeMap = numpy.empty([512, 512], dtype=int)
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

    def calcFaces():
        

        currentFaceBeingBuilt = []

        visitedEdges = {}
        for curEdgeId, curEdge in enumerate(edges):
            if(curEdge == None): continue
            debug = False
            startingNodeId = curEdge[0]
            startingPoint = nodes[startingNodeId]
            if(startingPoint == (120,126)
                or startingPoint == (119,125)
                or startingPoint == (120,124)
                or startingPoint == (120,125)
                or startingPoint == (121,125)):
                debug = False
                #print("whoa")
                #exit()

            if debug == True: print("starting edge: " + str(startingPoint) + " to " + str(nodes[curEdge[1]]))

            if(curEdgeId in visitedEdges): 
                if debug == True: print("already visited this edge")
                continue

            #print("first node coordinates " + str(nodes[curEdge[0]]))
            currentFaceBeingBuilt.append(curEdge[0])


            visitedEdges[curEdgeId] = True

            #print(curEdge)
            nextNode = curEdge[1]
            #print("next node has id " + str(curEdge[1]))

            nextNodeCoords = nodes[curEdge[1]]
            #if debug == True: print("looks like " + str(nextNodeCoords))
            #print(nextNode)

            prevNode = startingPoint
            #print("new cycle")
            for k in range(1000):
                #print("K " + str(k))
                if(nextNode == startingNodeId):
                    foundFaces.append(currentFaceBeingBuilt)
                    if debug == True: 
                        print("found start")   
                        print(currentFaceBeingBuilt)
                    
                    currentFaceBeingBuilt = []
                    break
                if debug == True: print("next node coordinates " + str(nodes[nextNode]))
                currentFaceBeingBuilt.append(nextNode)
                node1 = nodes[nextNode]
                #print(node1)
                nextNodeIds = edgesOfNodes[nextNode]
                
                #print("that is connected to nodes with ids " + str(nextNodeIds))

                largestAngle = -math.pi
                largestEdgeId = None

                for potentialNextEdgeId in nextNodeIds:
                    if(potentialNextEdgeId in visitedEdges): continue

                    v1 = numpy.subtract(node1, prevNode)
                    edge2 = edges[potentialNextEdgeId]
                    if(edge2 is None): continue

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

    calcFaces()
    
    print("hehee")

    #import shapely
    #from shapely.geometry import mapping, Polygon
    #from shapely import affinity

    #fig, ax = plt.subplots()

    import array 

    largest = 0
    largestId = 0
    for faceId, face in enumerate(foundFaces):
        smallestX = 255
        largestX = 0
        for nodeId in face:
            node = nodes[nodeId]
            if(node[0] < smallestX):
                smallestX = node[0]
            if(node[0] > largestX):
                largestX = node[0]
        width = largestX - smallestX

        if(width > largest):
            largestId = faceId
            largest = width

    foundFaces[largestId] = None
    #print("largest width is " + str(largest))
    #for nodeId in foundFaces[largestId]:
    #    allEdges = edgesOfNodes[nodeId]
    #    for edgeId in allEdges:
    #        edges[edgeId] = None
    #foundFaces[largestId] = None
    #print("widest face was " + str(largestId))



    for faceId, face in enumerate(foundFaces):
        if(face == None): 
            #print("skipping largest")
            continue
        l = []
        for nodeId in face:
            node = nodes[nodeId]
            #print("coords of node " + str(node))
            l.append( ( node[0], node[1]) )
            #print(nodes[node])

        if(len(face) <= 3):
            #print("face check")
            area = 0.5 * ((l[1][0]-l[0][0])*(l[2][1]-l[0][1])-(l[2][0]-l[0][0])*(l[1][1]-l[0][1]))   
            #print("area " + str(area))
            if area == 0.5:  
                #print("check again")      
                dst1 = abs(numpy.subtract( nodes[face[0]],  nodes[face[1]] ))
                dst2 = abs(numpy.subtract( nodes[face[1]],  nodes[face[2]] ))
                dst3 = abs(numpy.subtract( nodes[face[2]],  nodes[face[0]] ))
                ones = numpy.array([1, 1])
                if(numpy.array_equal(dst1,ones)):
                    face1 = face[0]
                    face2 = face[1]
                    #print("1")
                if(numpy.array_equal(dst2,ones)):
                    face1 = face[1]
                    face2 = face[2]
                    #print("2")
                if(numpy.array_equal(dst3,ones)):
                    face1 = face[2]
                    face2 = face[0]
                    #print("3")

                #print(face)
                #print(face1)
                #print(face2)
                node1edges = edgesOfNodes[face1]
                node2edges = edgesOfNodes[face2]

                #print("edges in first something " + str(node1edges))
                #print("edges in second something " + str(node2edges))

                countEdgesDeleted = 0

                for curEdgeId in node1edges:
                    #if(curEdgeId not in edges): continue
                    curEdge = edges[curEdgeId]
                    if(curEdge == None): continue
                    if(curEdge[1] != face2): continue
                    #if(curEdge[1] == face2):
                    #print("deleting edge " + str(curEdgeId))
                    edges[curEdgeId] = None
                    node1edges.remove(curEdgeId)
                    edgesOfNodes[face1]=node1edges
                    countEdgesDeleted += 1
                        
                for curEdgeId in node2edges:
                    #if(curEdgeId not in edges): continue
                    curEdge = edges[curEdgeId]
                    if(curEdge == None): continue
                    if(curEdge[1] != face1): continue

                    #if(curEdge[1] == face1):
                    #print("deleting edge " + str(curEdgeId))
                    edges[curEdgeId] = None
                    node2edges.remove(curEdgeId)
                    edgesOfNodes[face2]=node2edges
                    countEdgesDeleted += 1
                foundFaces[faceId] = None
                #print("deleted total " + str(countEdgesDeleted) + " edges")      

    print("jeejee")
    foundFaces = []
    calcFaces()

    largest = 0
    largestId = 0
    for faceId, face in enumerate(foundFaces):
        smallestX = 255
        largestX = 0
        for nodeId in face:
            node = nodes[nodeId]
            if(node[0] < smallestX):
                smallestX = node[0]
            if(node[0] > largestX):
                largestX = node[0]
        width = largestX - smallestX

        if(width > largest):
            largestId = faceId
            largest = width

    foundFaces[largestId] = None




    from fiona.crs import from_epsg

    # Define a polygon feature geometry with one attribute
    #schema = {
    #    'geometry': 'Polygon',
    #    'properties': {'id': 'int'},
    #}
    

    #print("len edges " + str(edges[0]))
    #with fiona.open('out2.shp', 'w', crs=from_epsg(2393), driver='ESRI Shapefile', schema=schema) as c:    
    for faceId, face in enumerate(foundFaces):
        if(face == None): 
            #print("skipping largest")
            continue

        l = []
        for nodeId in face:
            node = nodes[nodeId]
            #print(node)
            l.append( ( node[0], node[1]) )
        
        xs, ys = zip(*l)

        col = numpy.random.rand(3,)
        #ax.fill(xs,ys,color=col,edgecolor='black') 
        #ax.text(xs[0] + numpy.random.rand(1,), ys[0] + numpy.random.rand(1,), len(face))

        divided = numpy.array(l, dtype='f')
        #print(divided)
        #poly = shapely.geometry.Polygon(divided)

        #c.write({
        #    'geometry': mapping(poly),
        #    'properties': {'id': faceId},
        #})
        
        #print(__list)
        #polygon = Polygon(__list, True)
        #patches.append(polygon)
        #print face

    

    #from shapely.geometry import Point, LineString, MultiPoint
    #from shapely.wkt import dumps
    #from shapely import geometry, ops

    #schema = {
    #    'geometry': 'LineString',
    #    'properties': {'id': 'int'},
    #}

    from dxfwrite import DXFEngine as dxf

    drawing = dxf.drawing(outname)

    counter = 0
    #print("len edges " + str(edges[0]))

    # if loop through all edges, only select those with deg!=2
    # will that work?



    currentStreetChain = []
    streetChains = []
    for curEdgeId, curEdge in enumerate(edges):
        debug = False
        if(len(streetChains) == 109):
            print("running debug")
            debug = True

        if curEdge is None:
                #print("skipping")
                continue

        startNodeId = curEdge[0]
        endNodeId = curEdge[1]

        deg = len(edgesOfNodes[startNodeId])
        #print(deg)

        if(deg == 2): continue

        currentStreetChain.append(startNodeId)
        

        prevNodeId = startNodeId
        nextNodeId = endNodeId

        for k in range(1000):
            if(debug is True): print(nodes[nextNodeId])
            currentStreetChain.append(nextNodeId)

            nextNodeEdges = edgesOfNodes[nextNodeId]
            deg = len(nextNodeEdges)
            if(deg != 2):
                if(len(currentStreetChain) > 0):
                    streetChains.append(currentStreetChain)
                #if debug == True: 
                #print("formed a street chain")   
                #print(currentStreetChain)
                
                currentStreetChain = []
                break

            if(debug == True):
                print("current" + str(nextNodeId))
                print("previous" + str(prevNodeId))
            edge1 = edges[nextNodeEdges[0]]
            edge2 = edges[nextNodeEdges[1]]
            if(edge1[1] != prevNodeId):
                prevNodeId = nextNodeId
                nextNodeId = edge1[1]
            elif(edge2[1] != prevNodeId):
                prevNodeId = nextNodeId
                nextNodeId = edge2[1]
            else:
                nextNodeId = None

            if(nextNodeId == None):
                print("guru meditation")
                print(edge1)
                print(edge2)
                print(prevNodeId)
                exit()

            if(debug == True):
                #print(len(nextNodeEdges))
                print("chose as next " + str(nextNodeId))
                #print(edge1)
                #print(edge2)
            #exit()


        #curEdge1 = numpy.array(nodes[ curEdge[0] ], dtype='f' ) 
        #curEdge2 = numpy.array(nodes[ curEdge[1] ], dtype='f' ) 

        #s = [ curEdge1[0], curEdge2[0] ]
        #ys = [ curEdge1[1], curEdge2[1] ]

        #ax.text(xs[0], ys[0], curEdgeId)

    def smoothlist(l):
        smoothie = []
        smoothie.append(l[0])
        # skip first and last element
        for i, v in enumerate(l[1:-1]):
            # we are offsetting with 1 by slicing 
            # this just makes it more readable
            i = i+1
            v0 = 0.25 * l[i-1][0] + 0.5 * l[i][0] + 0.25 * l[i+1][0]
            v1 = 0.25 * l[i-1][1] + 0.5 * l[i][1] + 0.25 * l[i+1][1]
            smoothie.append( (v0, v1) )
        smoothie.append(l[-1])

        return smoothie

    class DPAlgorithm():

        def distance(self,  a, b):
            return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        def point_line_distance(self,  point, start, end):
            if (start == end):
                return self.distance(point, start)
            else:
                n = abs(
                    (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
                )
                d = sqrt(
                    (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                )
                return n / d

        def rdp(self, points, epsilon):
            """
            Reduces a series of points to a simplified version that loses detail, but
            maintains the general shape of the series.
            """
            dmax = 0.0
            index = 0
            i=1
            for i in range(1, len(points) - 1):
                d = self.point_line_distance(points[i], points[0], points[-1])
                if d > dmax :
                    index = i
                    dmax = d

            if dmax >= epsilon :
                results = self.rdp(points[:index+1], epsilon)[:-1] + self.rdp(points[index:], epsilon)
            else:
                results = [points[0], points[-1]]
            return results

    dpa = DPAlgorithm()

    #with fiona.open('out_linestring.shp', 'w', crs=from_epsg(2393), driver='ESRI Shapefile', schema=schema) as c:    
    line_strings = []
    for curStreetChainId, curStreetChain in enumerate(streetChains):
        #if(curStreetChainId != 1): continue

        l = []
        for nodeId in curStreetChain:
            node = nodes[nodeId]
            #print(node)
            l.append( ( 512-node[0], node[1] ) )
        
        #print("before smoothie")
        #print(l)
        smoothie = smoothlist(l)
        smoothie2 = smoothlist(smoothie)
        smoothie3 = smoothlist(smoothie2)
        smoothie4 = smoothlist(smoothie3)
        smoothie5 = smoothlist(smoothie4)

        smoothie5 = dpa.rdp(smoothie5, 1)

        xs, ys = zip(*smoothie5)
        #print(smoothie)
        #xs = smoothie[0]
        #ys = smoothie[1]
        #print(xs)

        #curEdge1 = numpy.array(nodes[ curEdge[0] ], dtype='f' ) * 3
        #curEdge2 = numpy.array(nodes[ curEdge[1] ], dtype='f' ) * 3
        #xs = [ curEdge1[0], curEdge2[0] ]
        #ys = [ curEdge1[1], curEdge2[1] ]


        #ls = LineString([Point(*curEdge1), Point(*curEdge1)])
        #line_strings.append(ls)

        #print(multi_line)
        #c.write({
        #    'geometry': mapping(ls),
        #    'properties': {'id': counter },
        #})

        scaled = numpy.array( smoothie5 , dtype='f' ) * 3

        polyline= dxf.polyline(linetype='DOT')
        polyline.add_vertices( scaled )
        drawing.add(polyline)

        #drawing.add(dxf.line(curEdge1, curEdge2, color=7))
    
        #ax.text(xs[0], ys[0], curEdgeId)
        #print(curEdge1)
        #print(curEdge2)
        col = numpy.random.rand(3,)
        #plt.plot(xs, ys, linestyle='-', linewidth=2, color=col)
        #ax.text(xs[0] + numpy.random.rand(1,) * 5, ys[0] + numpy.random.rand(1,) * 5, curStreetChainId)

    #multi_line = geometry.MultiLineString(line_strings)

    #print(multi_line)
    #c.write({
    #    'geometry': mapping(multi_line),
    #    'properties': {'id': counter },
    #})

    print("Exporting file " + outname)
    #plt.gca().invert_yaxis()
    drawing.save()
    #plt.show()

if __name__ == "__main__":
    create_dxf('thinned.png', 'out.dxf')