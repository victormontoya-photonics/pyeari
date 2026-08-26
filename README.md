```markdown
# Custom Image Demosaicking (MEARI & CEARI)

A Python package providing custom-built algorithms for advanced image demosaicking. This package introduces two specific methods, MEARI and CEARI, designed to reconstruct high-quality full-color and polarization images from raw filter array data using guided filtering and residual interpolation. 

You can view the package on [PyPI](https://pypi.org/project/eari-demosaic/).

## Installation

You can install the package directly from PyPI:

```bash
pip install eari-demosaic

```

## Dependencies

This package requires the following libraries:

* `numpy`
* `scipy`
* `opencv-python` (cv2)

## Available Functions

### Input & Output Data Formats

* **Input:** Raw image arrays provided to the functions must be either `uint8` (values 0-255) or `float32`.
* **Output:** All demosaicking functions return `float32` arrays strictly clipped to the range of `0` to `1`.

The package exposes the following main functions:

* **`CEARI(CPFA, pattern)`**: A custom demosaicking function for a Color Polarization Filter Array (CPFA).
* **`MEARI(MPFA)`**: A custom demosaicking function for a Monochrome Polarization Filter Array (MPFA).
* **`ri(cfa, pattern)`**: Standard color demosaicking using Residual Interpolation.
* **`make_cfa(img_path, pattern)`**: A utility function to generate a simulated Color Filter Array (CFA) from a standard image file.

### Supported Bayer & Polarization Patterns

Functions requiring a `pattern` argument accept the following standard Bayer array strings for color:

* `'RGGB'`
* `'BGGR'`
* `'GRBG'`
* `'GBRG'`

**Supported Polarization Pattern:**
Currently, the algorithms support only one polarization arrangement—the standard 2x2 macro-pixel. When combined with a color filter array, it forms a Color Polarization Filter Array (CPFA).

Here is an example of the supported CPFA layout (showing a **BGGR** color pattern):

```text
+-------+-------+-------+-------+
|  90°  |  45°  |  90°  |  45°  |
|   B   |   B   |   G   |   G   |
+-------+-------+-------+-------+
| 135°  |   0°  | 135°  |   0°  |
|   B   |   B   |   G   |   G   |
+-------+-------+-------+-------+
|  90°  |  45°  |  90°  |  45°  |
|   G   |   G   |   R   |   R   |
+-------+-------+-------+-------+
| 135°  |   0°  | 135°  |   0°  |
|   G   |   G   |   R   |   R   |
+-------+-------+-------+-------+

```

This specific polarimetric arrangement is the industry standard and is used by the most common polarimetric cameras on the market, such as those based on the **Sony IMX250MYR** (color) and **Sony IMX250MZR** (monochrome) sensors.

---

## Usage Examples

### 1. Color Polarization Demosaicking (CEARI)

The `CEARI` function takes a raw CPFA image and a Bayer pattern string. It standardizes the input automatically and returns four separate RGB images corresponding to the 90°, 0°, 45°, and 135° polarization angles.

```python
import cv2
from eari import CEARI

# Load your raw Color Polarization Filter Array image
raw_cpfa = cv2.imread("path_to_raw_image.tif", cv2.IMREAD_UNCHANGED)

# Process the image (assuming an RGGB pattern)
RGB90, RGB00, RGB45, RGB135 = CEARI(raw_cpfa, 'RGGB')

# Save or display the resulting images (converting back to uint8)
cv2.imwrite("output_90.png", RGB90 * 255)

```

### 2. Monochrome Polarization Demosaicking (MEARI)

If you are working with monochrome polarization data, use `MEARI`. It takes an MPFA image and returns the four directional intensity images.

```python
from eari import MEARI

# Process the monochrome array
I90, I00, I45, I135 = MEARI(raw_mpfa)

```

### 3. Standard Color Demosaicking (RI)

To perform standard residual interpolation on a normal Color Filter Array, you can use the `ri` function.

```python
from eari import ri, make_cfa

# Generate a simulated CFA from a ground-truth image for testing
cfa_image = make_cfa("test_image.png", "RGGB")

# Reconstruct the full RGB image
demosaicked_rgb = ri(cfa_image, "RGGB")

```

```

```
