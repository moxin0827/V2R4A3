#!/bin/bash
# ============================================================================
# PromptHMR Installation Script (Blackwell GPU sm_120 only)
# ============================================================================
#
# Environment: NVIDIA RTX PRO 6000 Blackwell (sm_120)
# PyTorch: 2.9.1 + CUDA 12.8
# Python: 3.11
#
# Usage:
#   conda create -n phmr python=3.11 -y
#   conda activate phmr
#   bash scripts/install_blackwell.sh
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "============================================"
echo "PromptHMR Blackwell GPU Installation Start"
echo "============================================"

# GCC-11 setup (applied to all builds)
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

# ----------------------------------------------------------------------------
# Step 1: PyTorch 2.9.1 + CUDA 12.8 + xformers installation
# - Blackwell sm_120 requires PyTorch 2.7+
# - xformers requires PyTorch 2.9.1, so install 2.9.1 from the start
# ----------------------------------------------------------------------------
echo ""
echo "[1/9] Installing PyTorch 2.9.1 + xformers..."
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128

# ----------------------------------------------------------------------------
# Step 2: Basic dependencies installation
# ----------------------------------------------------------------------------
echo ""
echo "[2/9] Installing basic dependencies..."
pip install -r requirements.txt

# ----------------------------------------------------------------------------
# Step 3: Submodule and Eigen library download
# - lietorch: DROID-SLAM Lie Group operations
# - eigen: Required for lietorch and droid_backends_intr build
# ----------------------------------------------------------------------------
echo ""
echo "[3/9] Downloading Submodule and Eigen..."

# git submodule init (lietorch etc.)
cd "$ROOT_DIR"
if [ -f ".gitmodules" ]; then
    git submodule update --init --recursive 2>/dev/null || true
fi

# Clone lietorch if not present
mkdir -p "$ROOT_DIR/pipeline/droidcalib/thirdparty"
cd "$ROOT_DIR/pipeline/droidcalib/thirdparty"
if [ ! -d "lietorch" ] || [ ! -f "lietorch/lietorch/lietorch.py" ]; then
    rm -rf lietorch
    git clone --depth 1 https://github.com/princeton-vl/lietorch.git
fi

# Download Eigen
if [ ! -d "eigen" ] || [ ! -f "eigen/Eigen/Dense" ]; then
    rm -rf eigen
    git clone --depth 1 https://gitlab.com/libeigen/eigen.git
fi

# ----------------------------------------------------------------------------
# Step 4: lietorch source build
# - DROID-SLAM Lie Group operations library
# ----------------------------------------------------------------------------
echo ""
echo "[4/9] Building lietorch..."
cd "$ROOT_DIR/pipeline/droidcalib"

# Create setup.py for lietorch
cat > setup.py << 'EOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='lietorch',
    version='0.3',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension('lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp',
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_90,code=sm_90',
                    '-gencode=arch=compute_100,code=sm_100',
                    '-gencode=arch=compute_120,code=sm_120',
                ]}),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
EOF

rm -rf build/
python setup.py install

# ----------------------------------------------------------------------------
# Step 5: droid_backends_intr source build
# - DROID-SLAM CUDA backend
# ----------------------------------------------------------------------------
echo ""
echo "[5/9] Building droid_backends_intr..."

# PyTorch 2.9 API compatibility: .type() -> .scalar_type()
sed -i 's/fmap1\.type()/fmap1.scalar_type()/g' src/altcorr_kernel.cu 2>/dev/null || true
sed -i 's/volume\.type()/volume.scalar_type()/g' src/correlation_kernels.cu 2>/dev/null || true

# Create setup.py for droid_backends_intr
cat > setup.py << 'EOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='droid_backends_intr',
    version='0.3',
    ext_modules=[
        CUDAExtension('droid_backends_intr',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_90,code=sm_90',
                    '-gencode=arch=compute_100,code=sm_100',
                    '-gencode=arch=compute_120,code=sm_120',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
EOF

rm -rf build/
python setup.py install

# ----------------------------------------------------------------------------
# Step 6: Additional dependencies
# - torch_scatter: For BA (Bundle Adjustment) operations
# - chumpy: For SMPL model loading (using Arthur151 fork)
# - --no-build-isolation: Fixes build environment isolation issues
# ----------------------------------------------------------------------------
echo ""
echo "[6/9] Installing additional dependencies..."
cd "$ROOT_DIR"

pip install torch_scatter --no-build-isolation

# chumpy: Original is not NumPy 2.x compatible, using Arthur151 fork
mkdir -p python_libs
if [ ! -f "python_libs/chumpy/setup.py" ]; then
    rm -rf python_libs/chumpy
    git clone https://github.com/Arthur151/chumpy python_libs/chumpy
fi
pip install -e python_libs/chumpy --no-build-isolation

# ----------------------------------------------------------------------------
# Step 7: detectron2 + sam2 installation (for video pipeline)
# - detectron2: Required for person detection
# - sam2: Segment Anything Model 2, required for video tracking
# ----------------------------------------------------------------------------
echo ""
echo "[7/9] Installing detectron2 + sam2 + checkpoints..."

pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# Download sam2 wheel (if not present)
if [ ! -f "data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl" ]; then
    echo "Downloading sam2 wheel..."
    gdown --folder -O ./data/ https://drive.google.com/drive/folders/1IXyhVqL25ofI-tYqyUZCqF-h4V20795H 2>/dev/null || true
fi

# sam2: Using PromptHMR custom wheel (includes load_video_frames_from_np etc.)
if [ -f "data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl" ]; then
    pip install data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl
else
    echo "Warning: sam2 custom wheel not found. Installing PyPI version..."
    pip install sam2
    echo "    (Some features may be limited. Full features require data/wheels/sam2-*.whl)"
fi

# Download checkpoints
mkdir -p "$ROOT_DIR/data/pretrain/sam2_ckpts"

# SAM2 checkpoint
if [ ! -f "$ROOT_DIR/data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt" ]; then
    echo "Downloading SAM2 checkpoint..."
    if ! wget -q -P "$ROOT_DIR/data/pretrain/sam2_ckpts" https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt; then
        echo "Warning: SAM2 checkpoint download failed! Manual download required:"
        echo "    wget -P data/pretrain/sam2_ckpts https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    fi
fi

# keypoint_rcnn checkpoint (Detectron2 official model)
if [ ! -f "$ROOT_DIR/data/pretrain/sam2_ckpts/keypoint_rcnn_5ad38f.pkl" ]; then
    echo "Downloading keypoint_rcnn checkpoint..."
    if ! wget -q https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl \
        -O "$ROOT_DIR/data/pretrain/sam2_ckpts/keypoint_rcnn_5ad38f.pkl"; then
        echo "Warning: keypoint_rcnn download failed! Manual download required:"
        echo "    wget https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl -O data/pretrain/sam2_ckpts/keypoint_rcnn_5ad38f.pkl"
    fi
fi

# ----------------------------------------------------------------------------
# Step 8: PromptHMR checkpoint download
# - Auto-run fetch_data.sh
# - Download GVHMR missing files (coco_aug_dict.pth etc.)
# ----------------------------------------------------------------------------
echo ""
echo "[8/9] Downloading PromptHMR checkpoints..."
cd "$ROOT_DIR"
bash scripts/fetch_data.sh

# Download GVHMR missing files (fixes PromptHMR repo bug)
BODY_MODEL_DIR="$ROOT_DIR/pipeline/gvhmr/hmr4d/utils/body_model"
GVHMR_RAW_URL="https://github.com/zju3dv/GVHMR/raw/main/hmr4d/utils/body_model"

GVHMR_FILES=(
    "coco_aug_dict.pth"
    "smplx2smpl_sparse.pt"
    "smpl_coco17_J_regressor.pt"
    "smpl_neutral_J_regressor.pt"
    "smpl_3dpw14_J_regressor_sparse.pt"
    "smplx_verts437.pt"
)

for file in "${GVHMR_FILES[@]}"; do
    if [ ! -f "$BODY_MODEL_DIR/$file" ]; then
        echo "Downloading $file (GVHMR)..."
        wget -q -O "$BODY_MODEL_DIR/$file" "$GVHMR_RAW_URL/$file" \
            || echo "Warning: $file download failed! Manual download required"
    fi
done

# ----------------------------------------------------------------------------
# Step 9: SMPL-X / SMPL body model download guide
# - Registration required, manual download instructions provided
# ----------------------------------------------------------------------------
echo ""
echo "[9/9] Checking SMPL-X / SMPL body models..."
if [ ! -d "$ROOT_DIR/data/body_models/smplx" ] || [ ! -d "$ROOT_DIR/data/body_models/smpl" ]; then
    echo "Warning: SMPL-X / SMPL body models not found."
    echo "    Please download manually after registration:"
    echo "    - https://smpl-x.is.tue.mpg.de"
    echo "    - https://smpl.is.tue.mpg.de"
    echo "    - bash scripts/fetch_smplx.sh"
else
    echo "SMPL-X / SMPL body models: OK"
fi

# Create GMR body_models symlink (for video2robot project)
GMR_DIR="$(dirname "$ROOT_DIR")/GMR"
if [ -d "$GMR_DIR" ] && [ -d "$ROOT_DIR/data/body_models/smplx" ]; then
    mkdir -p "$GMR_DIR/assets/body_models"
    if [ ! -L "$GMR_DIR/assets/body_models/smplx" ] && [ ! -d "$GMR_DIR/assets/body_models/smplx" ]; then
        ln -s "$ROOT_DIR/data/body_models/smplx" "$GMR_DIR/assets/body_models/smplx"
        echo "GMR body_models symlink created: OK"
    fi
fi

# ----------------------------------------------------------------------------
# Installation verification
# ----------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')
print(f'sm_120 support: {\"sm_120\" in str(torch.cuda.get_arch_list())}')

import xformers
print(f'xformers: {xformers.__version__}')

import lietorch
print('lietorch: OK')

import droid_backends_intr
print('droid_backends_intr: OK')

import chumpy
print('chumpy: OK')

import detectron2
print(f'detectron2: {detectron2.__version__}')

import sam2
print('sam2: OK')

print()
print('All installations complete!')
"

echo ""
echo "============================================"
echo "Next step: Download SMPL body models (registration required)"
echo "============================================"
echo ""
echo "If SMPL-X / SMPL body models are missing:"
echo "   - Register at https://smpl-x.is.tue.mpg.de"
echo "   - Register at https://smpl.is.tue.mpg.de"
echo "   - bash scripts/fetch_smplx.sh"
echo ""
echo "============================================"
echo "Usage:"
echo "  Image: python scripts/demo_phmr.py --image data/examples/example_1.jpg --gravity_align"
echo "  Video: python scripts/demo_video.py --input-video data/examples/boxing_short.mp4"
echo "============================================"
