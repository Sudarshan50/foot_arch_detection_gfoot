import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def preprocess_image(image_path):
    """Load image, convert to grayscale, remove background, and extract contours"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Background removal
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask and extract foot
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(img_rgb), plt.title('Original Image')
    plt.subplot(132), plt.imshow(thresh, cmap='gray'), plt.title('Thresholded Image')
    plt.subplot(133), plt.imshow(masked_img), plt.title('Extracted Foot')
    plt.tight_layout()
    plt.show()
    
    return masked_img, largest_contour

def detect_keypoints(contour, img):
    """Detect key points and extract sole boundary"""
    points = contour.squeeze()
    #plot all points with label..
    plt.scatter(points[:, 0], points[:, 1], color='black', s=5)
    plt.show()
    #save in a file..
    # np.savetxt('points.txt', points, fmt='%d')
    
    # Detect ball (leftmost point with highest y in column)
    ball_idx = np.argmin(points[:, 0])
    ball = tuple(points[ball_idx])
    
    # Detect heel (rightmost point with highest y in column)
    heel_idx = np.argmax(points[:, 0])
    heel = tuple(points[heel_idx])
        
    # Ensure ball is left of heel
    if ball[0] > heel[0]:
        ball, heel = heel, ball


    
    
    # Extract sole boundary points (lowest points between ball and heel)
    sole_points = []
    for x in range(ball[0], heel[0] + 1):
        # Get all points in current column
        col_points = points[points[:, 0] == x]
        if len(col_points) > 0:
            # Filter points with y greater than both ball and heel y-coordinates
            col_points = col_points[(col_points[:, 1] >= ball[1]) | (col_points[:, 1] >= heel[1])]
            if len(col_points) > 0:
                # Select point with highest y (lowest in image)
                sole_point = col_points[np.argmax(col_points[:, 1])]
                sole_points.append(sole_point)
    
    sole_points = np.array(sole_points)


    # Divide sole points into left and right halves
    mid_index = len(sole_points) // 2
    left_half = sole_points[:mid_index]
    right_half = sole_points[mid_index:]

    # Ensure both halves are not empty
    if len(left_half) == 0 or len(right_half) == 0:
        print("Error: One of the halves is empty")
        return None, None

    # Find maximum y in both halves
    max_y_from_left = np.argmax(left_half[:, 1])
    x_from_left = left_half[max_y_from_left][0]
    max_y_from_right = np.argmax(right_half[:, 1])
    x_from_right = right_half[max_y_from_right][0]

    new_ball = left_half[max_y_from_left]
    new_heel = right_half[max_y_from_right]

    # Extract new sole points between x_from_left and x_from_right
    new_sole_points = [point for point in sole_points if x_from_left <= point[0] <= x_from_right]
    

    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.scatter(ball[0], ball[1], color='red', s=50, label='Leftmost Point')
    plt.scatter(heel[0], heel[1], color='blue', s=50, label='Rightmost Point')
    plt.scatter(new_ball[0], new_ball[1], color='yellow', s=50, label='New Ball')
    plt.scatter(new_heel[0], new_heel[1], color='green', s=50, label='New Heel')
    new_sole_points = np.array(new_sole_points)
    plt.plot(new_sole_points[:, 0], new_sole_points[:, 1], 'g.', markersize=5, label='Sole Boundary',color='purple')
    plt.legend()
    plt.title('Key Points and Sole Boundary Detection')
    plt.show()
    
    return new_ball, new_heel, new_sole_points

def fit_polynomial(sole_points, degree=11):
    """Fit polynomial to sole boundary points"""
    x, y = sole_points[:, 0], sole_points[:, 1]
    
    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    
    # Generate points for plotting
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly(x_fit)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='black', s=10, label='Sole Points')
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Fitted Curve')
    plt.title('Sole Boundary Polynomial Fit')
    plt.legend()
    
    # Label the polynomial equation
    equation_text = "y = " + " + ".join([f"{coeff:.2e}x^{i}" for i, coeff in enumerate(coeffs[::-1])])
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=5, verticalalignment='top')
    plt.show()
    
    return poly

def find_arch_peak(poly, ball, heel):
    """Find arch peak using derivative analysis"""
    # Create base line between ball and heel
    base_line = np.linspace(ball[0], heel[0], 100)
    
    # Find peak by evaluating polynomial over dense grid
    x_dense = np.linspace(ball[0], heel[0], 1000)
    y_dense = poly(x_dense)
    peak_idx = np.argmin(y_dense)
    peak_x = x_dense[peak_idx]
    peak_y = y_dense[peak_idx]
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(base_line, poly(base_line), 'b--', label='Base Line')
    plt.scatter(peak_x, peak_y, color='green', s=100, label='Arch Peak')
    plt.title('Arch Peak Detection')
    plt.legend()
    plt.show()
    
    return (int(peak_x), int(peak_y))

def classify_foot(ball, peak, heel, img):
    """Classify foot type based on arch characteristics"""
    # Calculate measurements
    foot_length = abs(heel[0] - ball[0])
    arch_height = abs(ball[1] - peak[1])
    ratio = arch_height / foot_length
    
    # Classification logic
    if ratio < 0.15:
        foot_type = "Flat Foot"
    elif 0.15 <= ratio <= 0.30:
        foot_type = "Medium Arch"
    else:
        foot_type = "High Arch"
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.scatter([ball[0], heel[0]], [ball[1], heel[1]], 
                color=['red', 'blue'], s=50, label=['Ball', 'Heel'])
    plt.scatter(peak[0], peak[1], color='green', s=100, label='Arch Peak')
    plt.plot([ball[0], heel[0]], [ball[1], heel[1]], 'y--', label='Base Line')
    plt.vlines(peak[0], peak[1], ball[1], colors='purple', 
               linestyles='dashed', label='Arch Height')
    
    plt.legend()
    plt.title(f'Arch Height: {arch_height}px\n Bump Length: {foot_length}px\n')
    plt.show()
    
    return foot_type

def main(image_path):
    """Main processing pipeline"""
    # Process image through pipeline
    img, contour = preprocess_image(image_path)
    ball, heel, sole_points = detect_keypoints(contour, img)
    
    if ball is None or heel is None:
        print("Error: Key points not detected")
        return
    
    poly = fit_polynomial(sole_points)
    peak = find_arch_peak(poly, ball, heel)
    classification = classify_foot(ball, peak, heel, img)
    
    # Print results
    print(f"Arch Height Ratio: {abs(peak[1] - ball[1]) / abs(heel[0] - ball[0]):.2f}")
    print(f"Foot Classification: {classification}")

if __name__ == "__main__":
    image_path = "./f_images/ab.png" 
    main(image_path)