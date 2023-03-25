import os
import shutil
import subprocess
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2 as cv
import glob

def clearDir():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/data")
    reconstructPath = os.path.join(path, "reconstruct")
    transferPath = os.path.join(path, "transfer")
    for transfer_file in os.scandir(transferPath):
        if os.path.basename(transfer_file) != ".ipynb_checkpoints":
            os.remove(transfer_file)
    for reconstruct_file in os.scandir(reconstructPath):
        os.remove(reconstruct_file)

def create_video_from_intermediate_results(config, results_path):
    """
        time = iterations/(fps*sav_freq)
    """
    frameSize = (config.size, config.size)
    out = cv.VideoWriter(os.path.dirname(results_path) + "/0out.avi", fourcc=0, fps=config.fps, frameSize=frameSize)
    files = list(os.scandir(results_path))
    filter_files = [filename for filename in files if os.path.basename(filename) not in [".ipynb_checkpoints", "0out.mp4"]]
    for filename in sorted(filter_files, key=lambda e: int(e.name.split(".")[0])):
        img = cv.imread(filename.path)
        out.write(img)
    out.release()
    print (f"Video Created at {os.path.dirname(results_path)}")

def image_loader(path, config, device):
    image = Image.open(path)
    loader = transforms.Compose([transforms.Resize((config.size, config.size)),
                                 transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)