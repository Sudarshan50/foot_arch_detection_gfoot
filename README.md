# Foot Arch Detection

This repository contains a Python-based solution for detecting foot arch characteristics using computer vision techniques. The provided script `gfoot_class.py` includes all necessary functionalities to analyze foot arch data.

## Features
- Detects foot arch structure from given input images.
- Utilizes OpenCV and other essential Python libraries for image processing.
- Easy-to-follow steps for setup and usage.

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Required libraries (specified in `requirements.txt`).

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Sudarshan50/foot_arch_detection_gfoot.git
   cd foot_arch_detection_gfoot
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   Install the required libraries from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare the Input**
   - Place the foot image(s) in the `input/` folder.

2. **Run the Detection Script**
   ```bash
   python gfoot_class.py --input input/foot_image.jpg --output output/
   ```

3. **Output**
   - The processed images and corresponding analysis results will be stored in the `output/` folder.

## Example
```bash
python gfoot_class.py --input input/sample_foot.jpg --output output/
```
Expected output includes:
- Annotated image with marked foot arch
- Textual data describing arch height and structure

## How It Works
The foot arch detection process follows these steps:
1. **Background Removal:** The input image undergoes background removal to isolate the foot structure.
2. **Temporary Ball and Heel Points Identification:** The leftmost and rightmost points are considered temporary ball and heel points.
3. **Boundary Extraction:** Using OpenCV, the boundary points of the isolated foot are extracted.
4. **Boundary Filtering:** Points with higher `y` values (since the foot image is inverted) than the temporary ball and heel points are selected.
5. **Dividing the Foot:** The extracted boundary is divided into two halves:
   - The left half is iterated to identify the point with the maximum `y` value — this is the **Ball Point**.
   - The right half is iterated similarly to identify the **Heel Point**.
6. **Sole Point Extraction:** Points lying between the Ball Point's `x` coordinate and the Heel Point's `x` coordinate are identified as sole points.
7. **Polynomial Fitting:** A higher-order polynomial is fitted to the extracted sole points.
8. **Arch Height Calculation:** The maximum point of the polynomial curve is determined by minimizing its derivative. This point represents the **Arch Height** and **Bump Length**.

## Libraries Used
The following libraries are used in this project:
- `opencv-python` for image processing
- `numpy` for numerical operations
- `matplotlib` for visualization

To install these libraries manually if needed:
```bash
pip install opencv-python numpy matplotlib
```


