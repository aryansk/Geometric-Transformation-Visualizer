# 🔷 Geometric Transformation Visualizer 🔷

A comprehensive toolkit for visualizing various geometric transformations on both 2D shapes and images using Python. This project demonstrates fundamental geometric operations through interactive visualizations.

## 🎯 Project Overview

This project provides implementations of various geometric transformations in two parts:
1. Triangle Transformations: Basic 2D geometric transformations on a triangle
2. Image Transformations: Advanced transformations on images using OpenCV

## ⚙️ Features

### Triangle Transformations 📐
- Translation (moving the shape)
- Scaling (resizing)
- Rotation (around origin)
- Reflection (across x and y axes)
- Shearing (deformation)

### Image Transformations 🖼️
- Translation
- Reflection (horizontal, vertical, both)
- Rotation (with angle specification)
- Scaling (with custom factors)
- Cropping
- Shearing (X-axis and Y-axis)

## 🛠️ Requirements

```
numpy
matplotlib
opencv-python
```

## 📂 Project Structure

```
├── geometric_transformations/
│   ├── triangle_transform.py
│   └── image_transform.py
├── examples/
│   └── sample_images/
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/geometric-transformation-visualizer.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Triangle Transformations
```python
from triangle_transform import *

# Create and transform a triangle
triangle = np.array([[0, 0], [1, 0], [0.5, 1]])

# Apply transformations
translated = translate(triangle, 2, 3)
scaled = scale(triangle, 2, 2)
rotated = rotate(triangle, 45)
```

### Image Transformations
```python
from image_transform import *

# Load and transform an image
image = cv2.imread('path_to_image')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply transformations
translated = translate_image(image, 50, 50)
rotated = rotate_image(image, 45)
scaled = scale_image(image, 0.5, 0.5)
```

## 📊 Example Transformations

### Triangle Transformations:
- Translation: (tx=2, ty=3)
- Scaling: (sx=2, sy=2)
- Rotation: 45 degrees
- Reflection: x-axis
- Shearing: (shx=1)

### Image Transformations:
- Translation: 50 pixels in x and y
- Rotation: 45 degrees
- Scaling: 50% reduction
- Reflection: horizontal and vertical
- Shearing: 0.2 factor in x and y directions

## 🔍 Visualization

The project includes built-in visualization functions:
- `plot_triangle()` for 2D shape transformations
- `display_image()` for image transformations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Tech Stack

- Python 🐍
- NumPy 🔢
- OpenCV 📸
- Matplotlib 📊

## ✨ Author

Your Name
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- OpenCV documentation
- NumPy documentation
- Matplotlib plotting guide
