from scipy import ndimage
from dtcwt import Transform2d, Pyramid
from skimage.filters import sobel
import matplotlib.pyplot as plt
import numpy as np
import cv2


path = "images/lena_large.png"
img1 = cv2.imread(path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.resize(img1, 
                            (512, 512), 
                            interpolation=cv2.INTER_LINEAR)

transform = Transform2d()
coeffs1 = transform.forward(img1, nlevels=1)
[h,w] = img1.shape

# Calculate intensity gradient
grad_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)  # ∂I/∂x
grad_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)  # ∂I/∂y
gradient1 = np.sqrt(grad_x**2 + grad_y**2)

texture_gradient = np.zeros((512, 512))
E_tex = np.zeros((512, 512))
for i in range(6):
    # Compute texture information within each band
    subband = np.abs(coeffs1.highpasses[0][:, :, i]).astype(np.float32)
    subband_unfiltered_gradient = ndimage.gaussian_gradient_magnitude(subband, sigma=5)
    med_filter_size = 13
    subband_filtered = ndimage.median_filter(subband, 
                                             size=(1,med_filter_size)) # Horizontal median filtering
    subband_filtered = ndimage.median_filter(subband_filtered, 
                                             size=(med_filter_size,1)) # Vertical median filtering
    
    # Calculate texture gradient using a Gaussian filter
    sigma_val = 5
    subband_dx = ndimage.gaussian_filter(subband_filtered, 
                                         sigma= sigma_val, 
                                         order = (0,1))
    subband_dy = ndimage.gaussian_filter(subband_filtered, 
                                         sigma= sigma_val, 
                                         order = (1,0))
    
    subband_gradient = np.sqrt(subband_dx**2 + subband_dy**2)

    subband1_eroded = cv2.erode(subband_filtered/1.1, 
                                (3,3), 
                                iterations=1)
    
    # Calculate texture gradient
    subband_gradient_normalized = (subband_gradient / np.max(subband_gradient))
    weight_factor = ((subband.shape[0]*subband.shape[1])) / np.sum(subband_gradient_normalized **2)
    subband1_weighted_sum = weight_factor * subband_gradient_normalized
    
    # Upscale matrix
    subband1_weighted_sum = cv2.resize(subband1_weighted_sum, 
                            (h, w), 
                            interpolation=cv2.INTER_LANCZOS4)
    subband1_eroded = cv2.resize(subband1_eroded, 
                            (h, w), 
                            interpolation=cv2.INTER_LANCZOS4)
    
    
    #texture_gradient += np.log1p(subband1_weighted_sum)
    texture_gradient += (subband1_weighted_sum)
    E_tex += subband1_eroded

    plt.figure(1, figsize=(9,6))
    plt.subplot(2,3,1)
    plt.title(f"subband{i}")
    plt.imshow(np.abs(subband), cmap='gray')

    plt.subplot(2,3,2)
    plt.title(f"subband{i} filtered")
    plt.imshow(np.abs(subband_filtered), cmap='gray')

    plt.subplot(2,3,5)
    plt.title(f"subband{i} grad")
    plt.imshow(np.abs(subband_gradient), cmap='gray')

    plt.subplot(2,3,4)
    plt.title(f"subband{i} unfiltered grad")
    plt.imshow(np.abs(subband_unfiltered_gradient), cmap='gray')

    plt.subplot(2,3,3)
    plt.title(f"subband{i} weighted sum")
    plt.imshow(np.abs(subband1_weighted_sum), cmap='gray')

    plt.subplot(2,3,6)
    plt.title(f"eroded")
    plt.imshow(np.abs(subband1_eroded), cmap='gray')
    plt.tight_layout()
    #plt.show()

    

alpha = 2
beta = 7
activity = np.exp(np.maximum(0,(E_tex/alpha) - beta))

gradient1_median = np.median(gradient1)
texture_gradient_median = np.median(texture_gradient)

adjusted_intensity = (np.abs(gradient1) / (activity*(4*gradient1_median)))
adjusted_texture_gradient = (texture_gradient / texture_gradient_median)

combined_gradient = adjusted_intensity + adjusted_texture_gradient

img_min = combined_gradient.min()
img_max = combined_gradient.max()
img_norm = (combined_gradient - img_min) / (img_max - img_min) 
combined_gradient = (img_norm * 255).astype(np.uint8)
cv2.imwrite("combined.jpg", combined_gradient)

plt.figure(2, figsize=(9,6))
plt.subplot(2,3,1)
plt.title("Intensity Gradient")
plt.imshow(gradient1, cmap='gray')

plt.subplot(2,3,4)
plt.title("Texture Gradient ")
plt.imshow(texture_gradient, cmap='gray')

plt.subplot(2,3,3)
plt.title("Combined Gradient")
plt.imshow(combined_gradient, cmap='gray')

plt.subplot(2,3,2)
plt.title("Activity")
plt.imshow(np.log(activity), cmap='gray')

plt.subplot(2,3,5)
plt.title("Modulated Gradient")
plt.imshow(adjusted_intensity, cmap='gray')


plt.tight_layout()
#plt.show()


### Watershed
from skimage.morphology import h_minima
from skimage.morphology import reconstruction

# Assuming `image` is a 2D NumPy array
plt.figure(3)
plt.hist(combined_gradient.ravel(), bins=256, range=(np.min(combined_gradient), np.max(combined_gradient)), color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Image Values')
plt.grid(True)


activity_suppressed = h_minima(combined_gradient, h=10)
activity_removed = combined_gradient - activity_suppressed

ret, thresh = cv2.threshold(activity_removed,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
 
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
 
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
 
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
 
# Now, mark the region of unknown with zero
markers[unknown==255] = 0


img_color = cv2.cvtColor(activity_removed, cv2.COLOR_GRAY2BGR)
img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

markers = cv2.watershed(img_color,markers)
img1_color[markers == -1] = [255,0,0]

plt.figure(10)
plt.imshow(img1_color)
plt.show()