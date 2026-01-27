# 2D CAD Drawing generation from 3D CAD models

This tool is used to generate 2D CAD drawings from 3D CAD models.

## Installation

Install the package from PyPI:

```bash
pip install cad-3dto2d
```

Or install the package from the source code:

```bash
git clone https://github.com/neka-nat/cad-3dto2d.git
cd cad-3dto2d
uv sync
```

## Usage

```
python scripts/gen2d.py --step_file </path/to/step_file> --template A4_LandscapeParam --add_dimensions
```

## Demo

### Original 3D model (Flange)
![3d_flange](assets/simple_flange_3d.png)

### Generated 2D drawing (Flange)
![2d_flange](assets/simple_flange_2d.png)

### Original 3D model (Lego Block)
![3d_lego_block](assets/lego_block_3d.png)

### Generated 2D drawing (Lego Block)
![2d_lego_block](assets/lego_block_2d.png)
