import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('new_soviet.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Apply Laplacian filter to the image in Spatial Domain
# Create a new image with the same size as the original image


new_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        x = img[i][j]
        x = (int)(x)
        x -= img[i-1][j]
        x -= img[i+1][j]
        x -= img[i][j-1]
        x -= img[i][j+1]
        x += 4 * img[i][j]
        if(x < 0):
            x = 0
        elif(x > 255):
            x = 255
        new_image[i][j] = x
        
# Show the image
cv2.imshow('new_image', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite('new_image_spatial.jpg', new_image)

# Apply another Laplacian filter to the original image in Spatial Domain
# Create a new image with the same size as the original image

new_image_2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        x = img[i][j]
        x = (int)(x)
        x -= img[i-1][j]
        x -= img[i+1][j]
        x -= img[i][j-1]
        x -= img[i][j+1]
        x -= img[i-1][j-1]
        x -= img[i-1][j+1]
        x -= img[i+1][j-1]
        x -= img[i+1][j+1]
        x += 8 * img[i][j]
        if(x < 0):
            x = 0
        elif(x > 255):
            x = 255
        new_image_2[i][j] = x
        
# Show the image
cv2.imshow('new_image_2', new_image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite('new_image_2_spatial.jpg', new_image_2)

# Apply Laplacian filter to the image in Frequency Domain
# Create a new image with the same size as the original image

#new_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)



# Normalize the image to 0-1

img = img / 255.0

# Apply the Fourier Transform to the image

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

p, q = img.shape

# Get the Laplacian filter
H = np.zeros((p, q), dtype=np.float32)
for i in range(p):
    for j in range(q):
        H[i][j] = -4 * np.pi * np.pi * ((i - p / 2) * (i - p / 2) + (j - q / 2) * (j - q / 2))

# Get the Laplacian image
lap = H*fshift

# Apply the Inverse Fourier Transform to the image

lap_ishift = np.fft.ifftshift(lap)
new_lap = np.fft.ifft2(lap_ishift)
new_lap_real = np.real(new_lap)

# Convert the image to -1 to 1
r = np.max(new_lap_real) - np.min(new_lap_real)
new_r = 2
new_lap_real_converted = new_r * (new_lap_real - np.min(new_lap_real)) / r - 1

#img enhancement

g = img - new_lap_real_converted
g = np.clip(g, 0, 1)

# Show the image
cv2.imshow('g', g)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Normalize the image to 0-255
g = g * 255

# Save the image
cv2.imwrite('new_soviet_frequency.jpg', g)







