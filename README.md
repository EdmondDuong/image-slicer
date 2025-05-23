# Image Slicer Tool

A high-performance Python tool for slicing large images into smaller segments with various options and features.

## Features

- Multiple slicing modes:
  - Pixel-overlap mode: Create slices of exact dimensions with optional overlap
  - Image-count mode: Create a specific number of evenly-spaced slices
- Multi-threaded processing for improved performance
- Memory optimization for large images
- Detailed statistics and progress tracking
- Batch processing capabilities
- Comprehensive error handling

## Installation

Required Python packages:
```
pip install Pillow tqdm psutil
```

## Usage

### Command Line Interface

1. Single Image Processing:
```
python image_slicer_cli.py single input.jpg -o output_dir -s 1000,100
```

2. Batch Processing:
```
python image_slicer_cli.py batch "*.jpg" -o batch_output --overlap 50,50
```

### Python API

```python
from image_slicer import slice_image
from batch_process import batch_process

# Process single image
result = slice_image(
    "input.jpg",
    "output_dir",
    slice_size=(1000, 100),
    overlap=(50, 50)
)

# Batch process multiple images
results = batch_process(
    "*.jpg",
    "output_dir",
    slice_size=(1000, 100)
)
```

## Testing

Run the test suites:
```
python test_image_slicer.py
python test_batch_processing.py
```

## Documentation

For detailed documentation of all features and options, see:
- `image_slicer_cli.py --help` for command line options
- Function docstrings in `image_slicer.py` and `batch_process.py`

## License

MIT License
