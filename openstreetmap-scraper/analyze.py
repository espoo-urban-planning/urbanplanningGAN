import cv2
import numpy as np
import imutils

c = 1

res = cv2.imread('res.png', 0)
#res = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
#connectivity = 8 
#output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
#kadut_mask = cv2.erode(kadut_mask, kernel, iterations=1)
kernel = np.ones((2,2), np.uint8)
res = cv2.erode(res, kernel, iterations=1)
res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imwrite("processed.jpg", res)

'''


korttelit = cv2.imread('_2.jpg', 0)
korttelit = cv2.threshold(korttelit, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary

kadut = cv2.imread('_0.jpg', 0)
kadut_mask = cv2.threshold(kadut, 127, 255, cv2.THRESH_BINARY_INV)[1]
kernel = np.ones((5,5), np.uint8)
kadut_mask = cv2.erode(kadut_mask, kernel, iterations=1)

ret = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]


identifier = 0

for label in range(1,ret):
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]

    if(w < 10 or h < 10): continue

    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[labels == label] = 255    
    crop = mask[y:y+h,x:x+w]
    crop = cv2.resize(crop, (0,0), fx=c, fy=c)
    newH,newW = crop.shape 
    if(newH > 512 or newW > 512):
        continue
        #padding = (newH - 512) // 2
        #crop = crop[padding:padding+512,:]

    black = np.zeros((512,1024), dtype=np.uint8)
    offsetX = (512 - newW) // 2
    offsetY = (512 - newH) // 2
    #print("x " + str(x) + " width " + str(newW) + " offset X " + str(offsetX))
    #print("y " + str(y) + " height " + str(newH) + " offset Y " + str(offsetY))
    black[offsetY:offsetY+newH, offsetX:offsetX+newW] = crop

    korttelit_slice = korttelit[y:y+(512/c), x:x+(512/c)]
    korttelit_slice = cv2.resize(korttelit_slice, (0,0), fx=c, fy=c)
    kH, kW = korttelit_slice.shape
    korttelit_slice = cv2.bitwise_and(korttelit_slice,korttelit_slice,mask = black[0:kH, 0:kW])
    black[0:kH, 512:(512 + kW)] = korttelit_slice

    kadut_mask_slice = kadut_mask[y:y+(512/c), x:x+(512/c)]
    kadut_mask_slice = cv2.resize(kadut_mask_slice, (0,0), fx=c, fy=c)
    kmsH, kmsW = kadut_mask_slice.shape
    if(kmsW != 512 or kmsH != 512): continue

    kadut_slice = kadut[y:y+(512/c), x:x+(512/c)]
    kadut_slice = cv2.resize(kadut_slice, (0,0), fx=c, fy=c)
    ksH, ksW = kadut_slice.shape

    #print("kmsH " + str(kmsH) + " kmsW " + str(kmsW))
    #print("ksH " + str(ksH) + " ksW " + str(ksW))

    bigmask = np.zeros((512,1024), dtype=np.uint8)
    bigmask[0:512, 0:512] = kadut_mask_slice
    black[bigmask != 0] = 128

    #black = cv2.multiply(1.0 - bigmask, black)

    #kadut_slice = cv2.multiply(kadut_mask_slice)
    
    #kadut_slice = cv2.bitwise_and(kadut_slice,kadut_slice,mask = kadut_mask_slice)
    #black[0:kH, 0:kW] = kadut_slice

    white = np.sum(korttelit_slice == 255)
    if(white < 1): continue

    cv2.imwrite("out/" + str(identifier) + ".jpg", black)
    print(identifier)
    #cv2.imwrite("out/" + str(identifier) + "_kms.jpg", kadut_mask_slice)
    #cv2.imwrite("out/" + str(identifier) + "_bm.jpg", bigmask)
    identifier += 1
    #cv2.waitKey(0)


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    #cv2.imshow('2.png', labeled_img)
    cv2.imwrite("labeled2.jpg", labeled_img)
    #cv2.waitKey()
'''

#imshow_components(labels)