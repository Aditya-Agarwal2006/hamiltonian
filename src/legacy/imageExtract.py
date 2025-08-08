import cv2
import numpy as np

# Load image with alpha channel
img = cv2.imread("../trackImages/yas_marina_circuit.png", cv2.IMREAD_UNCHANGED)

bgr = img[..., :3]
alpha = img[..., 3]

# Create a mask for the opaque (non-transparent) region
# Mask where alpha > 0 (opaque)
mask = alpha > 0

# Convert the BGR to grayscale
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Invert grayscale so black track becomes white (255), background is black (0)
inverted_gray = 255 - gray

# Apply threshold only where mask is True
track_mask = np.zeros_like(gray)
track_mask[mask] = inverted_gray[mask] > 200  # 200 can be tuned; higher = only very dark pixels

# Convert boolean mask to uint8 (0 or 255)
track_mask = (track_mask * 255).astype(np.uint8)


# 1. Find all the contours in the thresholded image
contours, _ = cv2.findContours(track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # 2. Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # 3. Create a new, black image with the same dimensions
    clean_mask = np.zeros_like(track_mask)

    # 4. Draw ONLY the largest contour onto the new black image
    # The -1 argument fills the contour
    cv2.drawContours(clean_mask, [largest_contour], -1, (255), thickness=14)

    # (Optional) Save or display the final clean result
    cv2.imwrite("../trackImages/YasMarina_track_extracted.png", clean_mask)

else:
    print("No contours found.")
