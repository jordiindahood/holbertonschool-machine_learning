# Convolutions and Pooling

This project focuses on implementing foundational operations of Convolutional Neural Networks (CNNs) using NumPy.

## 🛠️ Features

### 1. Convolution with Padding and Strides

Implements a single 2D convolutional layer with flexible padding and stride.

**Function:** `convolve()` in [convolve.py](convolve.py)

**Features:**
- **Padding**: Supports `"same"` (automatic calculation) and tuple-based padding.
- **Strides**: Custom stride lengths `(sh, sw)`.
- **Multi-channel**: Handles input images with multiple channels (e.g., RGB).

### 2. Multi-Channel Convolution

Implements a specialized convolution for images with multiple channels, utilizing efficient NumPy broadcasting.

**Function:** `convolve_channels()` in [4-convolve_channels.py](4-convolve_channels.py)

**Key Optimization:**
- Instead of nested loops over channels, it performs a single broadcast multiplication followed by a sum, significantly speeding up the computation.

### 3. Max Pooling

Implements the max pooling operation, essential for reducing spatial dimensions and introducing translation invariance.

**Function:** `max_pool()` in [6-pool.py](6-pool.py)

**Features:**
- **Window Size**: Custom kernel/window size `(kh, kw)`.
- **Strides**: Custom stride lengths `(sh, sw)`.
- **Padding**: Supports `"same"` padding to maintain spatial dimensions.

### 4. Conv + Max Pool (ConvNet Layer)

Combines the convolution and max pooling operations into a standard ConvNet layer.

**Function:** `conv_pool()` in [7-conv_pool.py](7-conv_pool.py)

**Features:**
- **Sequence**: Applies convolution first, followed by max pooling.
- **Multiple Filters**: Supports multiple filters (kernels) for the convolution step.
- **Activation**: Applies the `tanh` activation function after convolution and before pooling.

## 🚀 Getting Started

### Prerequisites

- **Python 3.x**
- **NumPy 1.19.x or higher**

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd convolutions_and_pooling
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

## 📖 Usage

### Example: Convolution
```python
import numpy as np
from convolve import convolve

# Example usage
images = np.random.randn(10, 28, 28, 1)  # Batch of 10 grayscale images
kernels = np.random.randn(5, 5, 1, 16)  # 16 filters of size 5x5

output = convolve(images, kernels, padding="same", stride=(1, 1))
print(output.shape)  # (10, 28, 28, 16)
```

### Example: Pooling
```python
from pool import max_pool

pooled_output = max_pool(images, kernel_shape=(3, 3), stride=(2, 2))
print(pooled_output.shape)  # (10, 14, 14, 1)
```

### Example: Full ConvNet Layer
```python
from conv_pool import conv_pool

output = conv_pool(images, kernels, padding="same", stride=(1, 1))
print(output.shape)  # (10, 28, 28, 16)
```

## 📂 Project Structure

```
convolutions_and_pooling/
├── 2-convolve.py           # Basic convolution (no padding/stride)
├── 4-convolve_channels.py  # Multi-channel convolution
├── 5-convolve.py           # Full convolution with padding & stride
├── 6-pool.py               # Max pooling
├── 7-conv_pool.py          # Complete ConvNet layer (Conv + Pool + Tanh)
└── README.md
```

## 🧪 Testing

Run the scripts directly to test the implementations:

```bash
python 2-convolve.py
python 4-convolve_channels.py
python 5-convolve.py
python 6-pool.py
python 7-conv_pool.py
```

## 📚 Detailed Function Specifications

### `convolve(images, kernels, padding="same", stride=(1, 1))`

**Parameters:**

- **`images`**: NumPy array `(m, h, w, c)` representing a batch of images.
- **`kernels`**: NumPy array `(kh, kw, c, nc)` for the convolution kernels.
- **`padding`**: String (`"same"` or `"valid"`) or tuple `(ph, pw)` for padding.
- **`stride`**: Tuple `(sh, sw)` for stride lengths.

**Returns:**

- NumPy array `(m, h_out, w_out, nc)` representing the convoluted output.

### `max_pool(images, kernel_shape=(2, 2), stride=(2, 2))`

**Parameters:**

- **`images`**: NumPy array `(m, h, w, c)`.
- **`kernel_shape`**: Tuple `(kh, kw)` for the pooling window size.
- **`stride`**: Tuple `(sh, sw)` for stride lengths.

**Returns:**

- NumPy array `(m, h_out, w_out, c)` representing the pooled output.

### `conv_pool(images, kernels, padding="same", stride=(1, 1))`

**Parameters:**

- **`images`**: NumPy array `(m, h, w, c)`.
- **`kernels`**: NumPy array `(kh, kw, c, nc)`.
- **`padding`**: String (`"same"` or `"valid"`) or tuple `(ph, pw)`.
- **`stride`**: Tuple `(sh, sw)`.

**Returns:**

- NumPy array `(m, h_out, w_out, nc)` with `tanh` activation applied.

## 📄 License

This project is created for educational purposes as part of the Holberton School curriculum.
