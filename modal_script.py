# This script is used to set up a Modal application for training a YOLACT model.
# It mounts the necessary directories and installs the required dependencies.
# The script is designed to run in a Modal environment, which is a cloud-based platform for running Python code.
# The script uses the Modal library to define the application and its functions.

import os
import subprocess
import modal
from pathlib import Path


#Reference a pre-created volume holding your data
vol_data = modal.Volume.from_name("my-volume")
vol_model = modal.Volume.from_name("yolact-models")

app = modal.App(
    name="yolact-edge",
    image=(
        modal.Image.debian_slim("3.12")
        .run_commands(
            "apt-get update && apt-get install -y "
            "git libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 "
            "build-essential python3-dev && rm -rf /var/lib/apt/lists/*"
        )
        .pip_install(
            "torch==2.5.0",
            "torchvision==0.20.0",
            "torchaudio==2.5.0",
            "numpy",
            "Pillow",
            "opencv-python",
            "pycocotools",
            "tqdm",
            "requests",
            "matplotlib",
            "scikit-image",
            "cython",
            "tensorboard",
            "termcolor",
            "gitpython",
        )
        .run_commands("python3 -m pip install --upgrade pip setuptools")
        # mount code into /root/project
        .add_local_dir("yolact_edge", remote_path="/root/project_semantic_segmentation/yolact_edge")
        # mount data to the exact path train.py expects
        #.add_local_dir("data", remote_path=f"{defaulthome}/data")
    ),
    volumes={"/root/project_semantic_segmentation/data": vol_data, 
             "/root/project_semantic_segmentation/models": vol_model},
)

@app.function(gpu="H100", timeout=86400)
def train_mode():
    vol_data.reload()
    vol_model.reload()
    # change to code dir for train.py imports
    os.chdir("/root/project_semantic_segmentation/yolact_edge")
    # run training script; config inside train.py points at defaulthome/data
    subprocess.run(["python3", 
                    "train.py", 
                    "--config=my_custom_dataset", 
                    "--save_folder=/root/project_semantic_segmentation/models/", 
                    "--epochs=100"], check=True)
    vol_model.commit()

@app.local_entrypoint()
def main():
    '''if not os.path.exists("/root/project_semantic_segmentation/data"):
        print("Data directory does not exist. Please create it and add the necessary files.")
        return
    if not os.path.exists("/root/project_semantic_segmentation/models"):
        print("Model directory does not exist. Please create it.")
        return
    if not os.path.exists("/root/project_semantic_segmentation/yolact_edge"):
        print("YOLACT Edge directory does not exist. Please create it.")
        return'''
    train_mode.remote()

if __name__ == "__main__":
    main()
