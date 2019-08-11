import cv2
import numpy as np
import imutils

c = 1

folder = "input4/"

alueet = cv2.imread(folder + 'Yleiskaava.png', 0)
alueet = cv2.threshold(alueet, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
connectivity = 8 
output = cv2.connectedComponentsWithStats(alueet, connectivity, cv2.CV_32S)

kernel = np.ones((5,5), np.uint8)

korttelit = cv2.imread(folder + 'Korttelit.png', 0)
korttelit = cv2.threshold(korttelit, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary

#korttelit_mask = cv2.dilate(korttelit, kernel, iterations=5)
#korttelit_mask = cv2.erode(korttelit_mask, kernel, iterations=3)

kadut = cv2.imread(folder + 'Tavoiteverkko.png', 0)
kadut_mask = cv2.threshold(kadut, 127, 255, cv2.THRESH_BINARY_INV)[1]

detailed_kadut = cv2.imread(folder + 'Katualueet.png', 0)
detailed_kadut = cv2.threshold(detailed_kadut, 127, 255, cv2.THRESH_BINARY_INV)[1]

#kadut_mask = cv2.erode(kadut_mask, kernel, iterations=1)

tehokkuus = cv2.imread(folder + 'Tehokkuus.png', 0)

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

    mask = np.zeros(alueet.shape, dtype=np.uint8)
    mask[labels == label] = 255    
    crop = mask[y:y+h,x:x+w]
    crop = cv2.resize(crop, (0,0), fx=c, fy=c)
    newH,newW = crop.shape 
    if(newH > 512 or newW > 512):
        continue
        #padding = (newH - 512) // 2
        #crop = crop[padding:padding+512,:]

    green = np.zeros((512,1024), dtype=np.uint8)
    blue = np.zeros((512,1024), dtype=np.uint8)
    red = np.zeros((512,1024), dtype=np.uint8)
    offsetX = (512 - newW) // 2
    offsetY = (512 - newH) // 2
    #print("x " + str(x) + " width " + str(newW) + " offset X " + str(offsetX))
    #print("y " + str(y) + " height " + str(newH) + " offset Y " + str(offsetY))
    green[offsetY:offsetY+newH, offsetX:offsetX+newW] = crop

    yos = y-offsetY
    xos = x-offsetX

    korttelit_slice = korttelit[yos:yos+(512/c), xos:xos+(512/c)]
    korttelit_slice = cv2.resize(korttelit_slice, (0,0), fx=c, fy=c)
    kH, kW = korttelit_slice.shape
    korttelit_slice = cv2.bitwise_and(korttelit_slice,korttelit_slice,mask = green[0:kH, 0:kW])
    #blue[0:kH, 512:(512 + kW)] = korttelit_slice
    #red[0:kH, 512:(512 + kW)] = korttelit_slice

    korttelit_mask_slice = korttelit[yos:yos+(512/c), xos:xos+(512/c)]
    korttelit_mask_slice = cv2.resize(korttelit_mask_slice, (0,0), fx=c, fy=c)
    korttelit_mask_slice = cv2.bitwise_and(korttelit_mask_slice,korttelit_mask_slice,mask = green[0:kH, 0:kW])
    korttelit_mask_slice = cv2.dilate(korttelit_mask_slice, kernel, iterations=5)
    komsH, komsW = korttelit_mask_slice.shape
    if(komsW != 512 or komsH != 512): continue

    korttelit_mask_slice_2x = np.zeros((512,1024), dtype=np.uint8)
    korttelit_mask_slice_2x[0:512, 0:512] = korttelit_mask_slice

    # also add green channel to output so we have RGB 
    #(before added just blue and red, and have to do it here)
    green = cv2.bitwise_and(green,green,mask = korttelit_mask_slice_2x)
    green[0:kH, 512:(512 + kW)] = korttelit_slice
    red[0:kH, 512:(512 + kW)] = korttelit_slice
    blue[0:kH, 512:(512 + kW)] = korttelit_slice


    kadut_mask_slice = kadut_mask[yos:yos+(512/c), xos:xos+(512/c)]
    kadut_mask_slice = cv2.resize(kadut_mask_slice, (0,0), fx=c, fy=c)
    kmsH, kmsW = kadut_mask_slice.shape
    if(kmsW != 512 or kmsH != 512): continue

    kadut_slice = kadut[yos:yos+(512/c), xos:xos+(512/c)]
    kadut_slice = cv2.resize(kadut_slice, (0,0), fx=c, fy=c)
    ksH, ksW = kadut_slice.shape

    detailed_kadut_slice = detailed_kadut[yos:yos+(512/c), xos:xos+(512/c)]
    detailed_kadut_slice = cv2.resize(detailed_kadut_slice, (0,0), fx=c, fy=c)
    detailed_kadut_slice_masked = cv2.bitwise_and(detailed_kadut_slice, detailed_kadut_slice, mask=(255-green[0:512, 0:512]))
    dksH, dksW = detailed_kadut_slice_masked.shape

    #print("kmsH " + str(kmsH) + " kmsW " + str(kmsW))
    #print("ksH " + str(ksH) + " ksW " + str(ksW))

    combined = cv2.add(kadut_mask_slice,detailed_kadut_slice_masked)
    blue[0:512, 0:512] = detailed_kadut_slice_masked
    red[0:512, 0:512] = kadut_mask_slice
    #blue[0:512, 512:1024] = cv2.bitwise_and(detailed_kadut_slice, detailed_kadut_slice, mask=(green[0:512, 0:512]))
    #blue[0:512, 512:1024] = cv2.add(kadut_mask_slice,detailed_kadut_slice) 
    #cv2.bitwise_and(detailed_kadut_slice, green[0:512, 0:512])

    #tehokkuus_slice = tehokkuus[yos:yos+(512/c), xos:xos+(512/c)]
    #tehokkuus_slice = cv2.resize(tehokkuus_slice, (0,0), fx=c, fy=c)
    #tsH, tsW = tehokkuus_slice.shape
    #cv2.imwrite("out/" + str(identifier) + "_t.jpg", tehokkuus_slice)

    #red[0:512, 0:512] = tehokkuus_slice

    #green = cv2.multiply(1.0 - bigmask, green)

    #kadut_slice = cv2.multiply(kadut_mask_slice)
    
    #kadut_slice = cv2.bitwise_and(kadut_slice,kadut_slice,mask = kadut_mask_slice)
    #green[0:kH, 0:kW] = kadut_slice

    white = np.sum(korttelit_slice == 255)
    if(white < 1): continue

    out=cv2.merge((blue,green,red))

    cv2.imwrite("out/" + str(identifier) + ".png", out)
    print(identifier)
    cv2.imwrite("out/" + str(identifier) + "_dk.png", detailed_kadut[yos:yos+(512/c), xos:xos+(512/c)])
    cv2.imwrite("out/" + str(identifier) + "_dks.png", detailed_kadut_slice)
    cv2.imwrite("out/" + str(identifier) + "_dksm.png", detailed_kadut_slice_masked)
    #cv2.imwrite("out/" + str(identifier) + "_kms.jpg", kadut_mask_slice)
    #cv2.imwrite("out/" + str(identifier) + "_bm.jpg", bigmask)
    #cv2.imwrite("out/" + str(identifier) + "_koms.jpg", korttelit_mask_slice)
    identifier += 1
    #cv2.waitKey(0)

'''
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to green
    labeled_img[label_hue==0] = 0

    #cv2.imshow('2.png', labeled_img)
    cv2.imwrite("labeled2.jpg", labeled_img)
    #cv2.waitKey()
'''

#imshow_components(labels)