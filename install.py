"""Install SlabGCN."""

import argparse
import os
from shutil import unpack_archive

try:
    import requests
except ModuleNotFoundError:
    os.system("python -m pip install requests")
    import requests

# Create argument parser
parser = argparse.ArgumentParser(
    prog="SlabGCN Setup",
    description="Micro-utility to setup SlabGCN.",
)
parser.add_argument(
    "-m",
    "--mode",
    action="store",
    help="Specify either 'cpu' or 'gpu'. The latter installs the requisite CUDA versions of torch.",
    required=True,
)
parser.add_argument(
    "-cv",
    "--cuda-version",
    action="store",
    help="Specify a CUDA version (default is 12.6)",
    default="12.6",
)
parser.add_argument(
    "--skip-cuda-check",
    action="store_true",
    help="If this is specified, the CUDA version check is skipped (specify this if installing without internet).",
    default=False,
)
args = parser.parse_args()

# First, pip install the module
print("Installing python modules...")
if args.mode.strip().lower() == "gpu":
    ## Process cuda version
    cuda_version = str(round(float(args.cuda_version.strip()), 1))
    cuda_string = cuda_version.replace(".", "")

    ## Check validity of CUDA version
    cuda_url = f"https://download.pytorch.org/whl/cu{cuda_string}"
    if not args.skip_cuda_check:
        r = requests.head(cuda_url, allow_redirects=True, timeout=5)
        if r.status_code == 200:
            pass
        else:
            raise Exception("Invalid CUDA version specified.")

    ## Run pip in development mode
    os.system(f"python -m pip install -e . --extra-index-url {cuda_url}")

elif args.mode.strip().lower() == "cpu":
    ## Run pip in development mode
    os.system(
        f"python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu"
    )
else:
    raise Exception("--mode must either be 'cpu' or 'gpu'.")

# Second, unzip the structs archive
print("Testing...")
home = os.path.dirname(os.path.abspath(__file__))
archive_path = os.path.join(home, "data", "structs.tar.gz")
unpack_path = os.path.join(home, "data")
unpack_archive(archive_path, unpack_path)

# Test installation
dataset_path = os.path.join(unpack_path, "structs")
csv_path = os.path.join(dataset_path, "name_prop.csv")
save_path = os.path.join(home, "test_model")
os.system(f"slabgcn --dataset {dataset_path} --csv {csv_path} --epochs 1 --save {save_path}")
print("Test completed successfully.")

# Installation complete
print("Setup completed successfully.")
