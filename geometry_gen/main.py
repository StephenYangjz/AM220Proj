# vuer import
import yaml
import os
from asyncio import sleep
from typing import List
from vuer import Vuer, VuerSession
from vuer.schemas import CameraView, Sphere, Glb, AmbientLight, Scene, Box
import asyncio
from utils import *

# other imports
import json
import random
import trimesh
import numpy as np
import PIL.Image as PImage
from io import BytesIO
from tqdm import tqdm

global counter

FOV = 60
SCALE = 1
JSON_INPUT = 'transform_template.json'
YAML_PATH = 'layout.yaml'
OUTPUT_DIR = "gen_dataset"
SEED = 42
WIDTH = 800
HEIGHT = 800

# seeding and directory
np.random.seed(SEED)
random.seed(SEED)
CENTER = (0, 0, 0)

# fov etc calculation
aspect = WIDTH/HEIGHT
foh = np.deg2rad(FOV)
h = 2 * np.sin(foh/2)
w = aspect * h
fov = 2 * np.arcsin(w / 2)


# start vuer
app = Vuer(static_root="assets")
sphere = Sphere(
    args=[6, 32, 16],
    materialType="standard",
    material=dict(map="http://localhost:8012/static/background.jpg", side=1),
    position=[0, 0, 0],
)


with open(YAML_PATH, 'r') as file:
    layout_data = yaml.safe_load(file)


@app.spawn
async def session(sess: VuerSession):

    ############## Load Obj ####################
    objects = []
    # Process each object in the living room
    for item in layout_data['objects']:
        # mesh = trimesh.load_mesh(item["filename"])
        # bounding_box = mesh.bounding_box_oriented
        # original_size = bounding_box.extents
        position = [item['position'][0], item['position'][1], item['position'][2] if len(item['position'])>2 else 0.3]
        filename = os.path.join("http://localhost:8012/static", item["filename"].split("/")[-1])
        factor = item["size"]
        objects.append(Glb(src=filename,
                        scale=factor,
                        position=position,
                        rotation=[np.pi/2,0,0],
                        materialType="standard",
                        material=dict(color="white", side=0),
                        ))
    # add single sided walls
    objects.append(Box(material=dict(color= "grey", side=1),
            args=[2.5, 2.5, 1],
            position=[0, 0, 0.5],
            rotation=[0, 0, 0],                                            
            ),)
    # set scene up
    sess.set @ Scene(
        sphere,
        bgChildren=[
            AmbientLight(color="#ffffff", intensity=20, key="default-light", rotation=[np.pi / 2, 0, 0]),
        ],
        *objects,
        up=[0, 0, 1],
        collapseMenu=True,
    )


    while True:
        await asyncio.sleep(1.0)
app.run()
