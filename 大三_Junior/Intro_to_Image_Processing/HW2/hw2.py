import numpy as np
import cv2




img = cv2.imread('Q1.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Do histogram equalization to the given image

def histogram_equalization(img):
    h, w = img.shape
    img2 = np.zeros((h, w), dtype=np.uint8)
    hist = np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
            
    pdf = hist / hist.sum()
    cdf = np.zeros(256)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
    cdf = np.round(cdf * 255)
    
    for i in range(h):
        for j in range(w):
            img2[i, j] = cdf[img[i, j]]
            
    return img2

img_q1 = histogram_equalization(img)

cv2.imshow('image',img_q1)
cv2.imwrite('Q1_result.jpg', img_q1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_q2_src = cv2.imread('Q2_source.jpg', cv2.IMREAD_GRAYSCALE)
img_q2_ref = cv2.imread('Q2_reference.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('image',img_q2_src)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image',img_q2_ref)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Do histogram specification to the given image

def histogram_specification(img_src, img_ref):
    src_h, src_w = img_src.shape
    ref_h, ref_w = img_ref.shape
    img2 = np.zeros((src_h, src_w), dtype=np.uint8)
    hist_src = np.zeros(256)
    hist_ref = np.zeros(256)
    
    for i in range(src_h):
        for j in range(src_w):
            hist_src[img_src[i, j]] += 1
    for i in range(ref_h):
        for j in range(ref_w):
            hist_ref[img_ref[i, j]] += 1
    
    pdf_src = hist_src / hist_src.sum()
    pdf_ref = hist_ref / hist_ref.sum()
    cdf_src = np.zeros(256)
    cdf_ref = np.zeros(256)
    cdf_src[0] = pdf_src[0]
    cdf_ref[0] = pdf_ref[0]
    
    for i in range(1, 256):
        cdf_src[i] = cdf_src[i-1] + pdf_src[i]
        cdf_ref[i] = cdf_ref[i-1] + pdf_ref[i]
    cdf_src = np.round(cdf_src * 255)
    cdf_ref = np.round(cdf_ref * 255)
    
    mapping = np.zeros(256)
    for i in range(256):
        diff = np.abs(cdf_src[i] - cdf_ref)
        mapping[i] = np.argmin(diff)
        
    for i in range(src_h):
        for j in range(src_w):
            img2[i, j] = mapping[img_src[i, j]]
            
    return img2
    

img_q2 = histogram_specification(img_q2_src, img_q2_ref)

cv2.imshow('image',img_q2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Q2_result.jpg', img_q2)







