import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Read the image
image = cv2.imread('building.jpg')

cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use Nearest Neighbor Interpolation to turn the image by 30 degrees 
def Nearest_Neighbor_Turn(image, angle):
    angle = np.deg2rad(angle)
    height, width = image.shape[:2]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # Rotate by applying the inverse rotation
            x = ((i - height / 2) * np.cos(angle) + (j - width / 2) * np.sin(angle) + height / 2)
            y = (-(i - height / 2) * np.sin(angle) + (j - width / 2) * np.cos(angle) + width / 2)
            
            
            
            if x >= 0 and x < height and y >= 0 and y < width:
                new_image[i, j] = image[int(x), int(y)]
    return new_image

rotated_image = Nearest_Neighbor_Turn(image, -30)

cv2.imshow('Rotated Image', rotated_image)
cv2.imwrite('output_0.png', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use Bilinear Interpolation to turn the image by 30 degrees

def Bilinear_Turn(image, angle):
    angle = np.deg2rad(angle)
    height, width = image.shape[:2]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # Rotate by applying the inverse rotation
            x = ((i - height / 2) * np.cos(angle) + (j - width / 2) * np.sin(angle) + height / 2)
            y = (-(i - height / 2) * np.sin(angle) + (j - width / 2) * np.cos(angle) + width / 2)
            
            # Bilinear Interpolation
            if x >= 0 and x < height and y >= 0 and y < width:
                x1, y1 = int(x), int(y)
                x2, y2 = x1 + 1, y1 + 1
                if x2 < height and y2 < width:
                    x_diff, y_diff = x - x1, y - y1
                    new_image[i, j] = (1 - x_diff) * (1 - y_diff) * image[x1, y1] + (1 - x_diff) * y_diff * image[x1, y2] + x_diff * (1 - y_diff) * image[x2, y1] + x_diff * y_diff * image[x2, y2]
    return new_image

rotated_image = Bilinear_Turn(image, -30)

cv2.imshow('Bilinear Rotated Image', rotated_image)
cv2.imwrite('output_1.png', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use Bicubic Interpolation to turn the image by 30 degrees

def Bicubic_Turn(image, angle):
    angle = np.deg2rad(angle)
    height, width = image.shape[:2]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            
            # Rotate by applying the inverse rotation
            x = ((i - height / 2) * np.cos(angle) + (j - width / 2) * np.sin(angle) + height / 2)
            y = (-(i - height / 2) * np.sin(angle) + (j - width / 2) * np.cos(angle) + width / 2)
            
            if x >= 1 and x < height-2 and y >= 1 and y < width-2:
                for channel in range(3):
                    
                    xt0 = int(x) - 1
                    xt1 = int(x)
                    xt2 = int(x) + 1
                    xt3 = int(x) + 2
                    yt0 = int(y) - 1
                    yt1 = int(y)
                    yt2 = int(y) + 1
                    yt3 = int(y) + 2
                    p00 = image[xt0, yt0, channel].astype(np.int32)
                    p01 = image[xt0, yt1, channel].astype(np.int32)
                    p02 = image[xt0, yt2, channel].astype(np.int32)
                    p03 = image[xt0, yt3, channel].astype(np.int32)
                    p10 = image[xt1, yt0, channel].astype(np.int32)
                    p11 = image[xt1, yt1, channel].astype(np.int32)
                    p12 = image[xt1, yt2, channel].astype(np.int32)
                    p13 = image[xt1, yt3, channel].astype(np.int32)
                    p20 = image[xt2, yt0, channel].astype(np.int32)
                    p21 = image[xt2, yt1, channel].astype(np.int32)
                    p22 = image[xt2, yt2, channel].astype(np.int32)
                    p23 = image[xt2, yt3, channel].astype(np.int32)
                    p30 = image[xt3, yt0, channel].astype(np.int32)
                    p31 = image[xt3, yt1, channel].astype(np.int32)
                    p32 = image[xt3, yt2, channel].astype(np.int32)
                    p33 = image[xt3, yt3, channel].astype(np.int32)
                    
                    a = -1/2*p00 + 3/2*p01 - 3/2*p02 + 1/2*p03
                    b = p00 - 5/2*p01 + 2*p02 - 1/2*p03
                    c = -1/2*p00 + 1/2*p02
                    d = p01
                    
                    x0 = a*(y-np.floor(y))**3 + b*(y-np.floor(y))**2 + c*(y-np.floor(y)) + d
                    
                    a = -1/2*p10 + 3/2*p11 - 3/2*p12 + 1/2*p13
                    b = p10 - 5/2*p11 + 2*p12 - 1/2*p13
                    c = -1/2*p10 + 1/2*p12
                    d = p11
                    
                    x1 = a*(y-np.floor(y))**3 + b*(y-np.floor(y))**2 + c*(y-np.floor(y)) + d
                    
                    a = -1/2*p20 + 3/2*p21 - 3/2*p22 + 1/2*p23
                    b = p20 - 5/2*p21 + 2*p22 - 1/2*p23
                    c = -1/2*p20 + 1/2*p22
                    d = p21
                    
                    x2 = a*(y-np.floor(y))**3 + b*(y-np.floor(y))**2 + c*(y-np.floor(y)) + d
                    
                    a = -1/2*p30 + 3/2*p31 - 3/2*p32 + 1/2*p33
                    b = p30 - 5/2*p31 + 2*p32 - 1/2*p33
                    c = -1/2*p30 + 1/2*p32
                    d = p31
                    
                    x3 = a*(y-np.floor(y))**3 + b*(y-np.floor(y))**2 + c*(y-np.floor(y)) + d
                    
                    a = -1/2*x0 + 3/2*x1 - 3/2*x2 + 1/2*x3
                    b = x0 - 5/2*x1 + 2*x2 - 1/2*x3
                    c = -1/2*x0 + 1/2*x2
                    d = x1
                    
                    
                    fin = a*(x-np.floor(x))**3 + b*(x-np.floor(x))**2 + c*(x-np.floor(x)) + d
                    if(fin < 0):
                        fin = 0
                    if(fin > 255):
                        fin = 255
                    new_image[i, j, channel] = fin
                    
                    
                    
                    print(i, j, channel, x0, x1, x2, x3, new_image[i, j, channel])
            
            
            
    return new_image

rotated_image = Bicubic_Turn(image, -30)

cv2.imshow('Bicubic Rotated Image', rotated_image)
cv2.imwrite('output_bicubic.png', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use Nearest Neighbor Interpolation to scale the image by 2

def Nearest_Neighbor_Scale(image, scale):
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            x, y = int(i / scale), int(j / scale)
            if x >= 0 and x < height and y >= 0 and y < width:
                new_image[i, j] = image[x, y]
    return new_image

scaled_image = Nearest_Neighbor_Scale(image, 2)

cv2.imshow('Nearest Neighbor Scaled Image', scaled_image)
cv2.imwrite('output0.png', scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use Bilinear Interpolation to scale the image by 2

def Bilinear_Scale(image, scale):
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            x, y = i / scale, j / scale
            x1, y1 = int(x), int(y)
            x2, y2 = x1 + 1, y1 + 1
            if x2 < height and y2 < width:
                x_diff, y_diff = x - x1, y - y1
                new_image[i, j] = (1 - x_diff) * (1 - y_diff) * image[x1, y1] + (1 - x_diff) * y_diff * image[x1, y2] + x_diff * (1 - y_diff) * image[x2, y1] + x_diff * y_diff * image[x2, y2]
    return new_image

scaled_image = Bilinear_Scale(image, 2)

cv2.imshow('Bilinear Scaled Image', scaled_image)
cv2.imwrite('output1.png', scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use Bicubic Interpolation to scale the image by 2

def Bicubic_Scale(image, scale):
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            x, y = i / scale, j / scale
            if x >= 1 and x < height-2 and y >= 1 and y < width-2:
                for channel in range(3):
                    xt0 = int(x) - 1
                    xt1 = int(x)
                    xt2 = int(x) + 1
                    xt3 = int(x) + 2
                    yt0 = int(y) - 1
                    yt1 = int(y)
                    yt2 = int(y) + 1
                    yt3 = int(y) + 2
                    p00 = image[xt0, yt0, channel].astype(np.int32)
                    p01 = image[xt0, yt1, channel].astype(np.int32)
                    p02 = image[xt0, yt2, channel].astype(np.int32)
                    p03 = image[xt0, yt3, channel].astype(np.int32)
                    p10 = image[xt1, yt0, channel].astype(np.int32)
                    p11 = image[xt1, yt1, channel].astype(np.int32)
                    p12 = image[xt1, yt2, channel].astype(np.int32)
                    p13 = image[xt1, yt3, channel].astype(np.int32)
                    p20 = image[xt2, yt0, channel].astype(np.int32)
                    p21 = image[xt2, yt1, channel].astype(np.int32)
                    p22 = image[xt2, yt2, channel].astype(np.int32)
                    p23 = image[xt2, yt3, channel].astype(np.int32)
                    p30 = image[xt3, yt0, channel].astype(np.int32)
                    p31 = image[xt3, yt1, channel].astype(np.int32)
                    p32 = image[xt3, yt2, channel].astype(np.int32)
                    p33 = image[xt3, yt3, channel].astype(np.int32)
                    
                    a = -1/2*p00 + 3/2*p01 - 3/2*p02 + 1/2*p03
                    b = p00 - 5/2*p01 + 2*p02 - 1/2*p03
                    c = -1/2*p00 + 1/2*p02
                    d = p01
                    
                    x0 = a*(y-np.floor(y))**3 + b*(y-np.floor(y))**2 + c*(y-np.floor(y)) + d
                    
                    a = -1/2*p10 + 3/2*p11 - 3/2*p12 + 1/2*p13
                    b = p10 - 5/2*p11 + 2*p12 - 1/2*p13
                    c = -1/2*p10 + 1/2*p12
                    d = p11
                    
                    x1 = a*(y-np.floor(y))**3 + b*(y-np.floor(y))**2 + c*(y-np.floor(y)) + d
                    
                    a = -1/2*p20 + 3/2*p21 - 3/2*p22 + 1/2*p23
                    b = p20 - 5/2*p21 + 2*p22 - 1/2*p23
                    c = -1/2*p20 + 1/2*p22
                    d = p21
                    
                    x2 = a*(y-np.floor(y))**3 + b*(y-np.floor(y))**2 + c*(y-np.floor(y)) + d
                    
                    a = -1/2*p30 + 3/2*p31 - 3/2*p32 + 1/2*p33
                    b = p30 - 5/2*p31 + 2*p32 - 1/2*p33
                    c = -1/2*p30 + 1/2*p32
                    d = p31
                    
                    x3 = a*(y-np.floor(y))**3 + b*(y-np.floor(y))**2 + c*(y-np.floor(y)) + d
                    
                    a = -1/2*x0 + 3/2*x1 - 3/2*x2 + 1/2*x3
                    b = x0 - 5/2*x1 + 2*x2 - 1/2*x3
                    c = -1/2*x0 + 1/2*x2
                    d = x1
                    
                    
                    fin = a*(x-np.floor(x))**3 + b*(x-np.floor(x))**2 + c*(x-np.floor(x)) + d
                    if(fin < 0):
                        fin = 0
                    if(fin > 255):
                        fin = 255
                    new_image[i, j, channel] = fin
                    
                    print(i, j, channel, x0, x1, x2, x3, new_image[i, j, channel])
    return new_image

scaled_image = Bicubic_Scale(image, 2)

cv2.imshow('Bicubic Scaled Image', scaled_image)
cv2.imwrite('output2.png', scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
                    
