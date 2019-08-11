import cv2


img = cv2.imread('osm/HelsinkiFinland2_t.png')

b,g,r = cv2.split(img)

new = g*256 + b
biggest = new.max()
smallest = 134*256
_range = biggest - smallest
print ("biggest " + str(biggest))
print ("smallest " + str(smallest))
print ("range " + str(_range))
new = new - smallest
new = (new / _range) * 255

cv2.imwrite('n.png',new)
#cv2.imwrite('g.png',g)
#cv2.imwrite('r.png',r)

