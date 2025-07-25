import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os

# 1. Load your clean track image
image_path = "../trackImages/Silverstone_track_extracted.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. (Optional) Clean up the image a little if needed
_, image_bin = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))  # Uncomment if you see speckles

# 3. Find the contours of the boundaries (outer and inner)
contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
main_contours = []
if hierarchy is not None:
    hierarchy = hierarchy[0]
    for i, h in enumerate(hierarchy):
        # h[3] == -1 means outermost, h[3] != -1 means it's a hole (inner edge)
        if h[3] == -1 or h[3] >= 0:
            main_contours.append(contours[i])
# Sort by area and take the two largest
main_contours = sorted(main_contours, key=cv2.contourArea, reverse=True)[:2]

splines = []
for contour in main_contours:
    points = contour.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    tck, u = splprep([x, y], s=0, per=True)
    splines.append(tck)

# 5. Save the spline tck data to trackModels as .npz files
os.makedirs("../trackModels", exist_ok=True)
for i, tck in enumerate(splines):
    np.savez(f"../trackModels/track_spline_{i}.npz", tck=np.array(tck, dtype=object))

# 6. Visualize the results to verify
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')
u_new = np.linspace(0, 1, 1000)
colors = ['cyan', 'magenta']
for i, tck in enumerate(splines):
    x_new, y_new = splev(u_new, tck)
    plt.plot(x_new, y_new, colors[i], linewidth=2)
plt.title("Digitized Spline Boundaries (Both Edges, No Over-Smoothing)")
plt.savefig("../trackModels/track_splines_visualization.png")
plt.show()