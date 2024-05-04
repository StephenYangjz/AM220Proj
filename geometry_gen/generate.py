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
SCALE = 0.7
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

# Prepare output JSON file
create_dir_with_prompt(OUTPUT_DIR)
json_data_in = json.load(open(JSON_INPUT))

# start vuer
app = Vuer(static_root="assets")
sphere = Sphere(
    args=[4, 32, 16],
    materialType="standard",
    material=dict(map="http://localhost:8012/static/background.jpg", side=1),
    position=[0, 0, 0],
)

############## Cam Frustums ####################

cam_frustums: List[CameraView] = []
cam_extrinsics: List[CameraView] = []
visited_views = set()
# Iterate over all of the camera positions
for i, frame_in in tqdm(enumerate(json_data_in['frames'])):
    if frame_in["view_index"] in visited_views or any(frame_in["transform_matrix"][i][3] <= 0 for i in range(3)): # filter based on quadrant
        continue

    visited_views.add(frame_in["view_index"])
    relative_pos = np.array([
        frame_in['transform_matrix'][0][3],
        frame_in['transform_matrix'][2][3],
        frame_in['transform_matrix'][1][3],  # b/c different handedness
    ])

    # Scale the position outwards
    if SCALE != 1.0:
        origin = np.array([0, 0, 0])
        relative_pos = SCALE * (relative_pos - origin) + origin

    # Compute the pitch and yaw to point the camera towards the origin
    direction = pos_to_rot([relative_pos[0], relative_pos[1], relative_pos[2]], degrees=False)
    pitch = direction['pitch']
    yaw = direction['yaw']
    roll = 0
    angle = [pitch, yaw, roll]

    # Now shift everything to the point of interest
    shifted_pos = relative_pos + np.array(CENTER)

    # rotations and points returned assume pointinr towards and centered on (0,0,0)
    point = shifted_pos[0], shifted_pos[1], shifted_pos[2]
    scaled_matrix = scale_extrinsics(frame_in['transform_matrix'], SCALE)
    cam_extrinsics.append(scaled_matrix)
    flattened_transform = [scaled_matrix[col][row] for col in range(4) for row in range(4)]
    cam_frustums.append(
        CameraView(
            fov=FOV,
            width=WIDTH,
            height=HEIGHT,
            key="default-cam",
            stream="ondemand",
            fps=30,
            near=0.45,
            far=7,
            showFrustum=True,
            downsample=1,
            distanceToCamera=2,
            matrix=colmap2three(flattened_transform),
            monitor=True,
            renderDepth=True
        ),
    )


with open(YAML_PATH, 'r') as file:
    layout_data = yaml.safe_load(file)

json_data_out = {}
json_data_out["camera_angle_x"] = fov
json_data_out['frames'] = []

frame_temp = {}
@app.spawn
async def session(sess: VuerSession):
    await sleep(2)
    for object_i in range(len(layout_data['objects']) + 1):

        # per object paths
        mask_dir = os.path.join(OUTPUT_DIR, "mask"+str(object_i))
        empty_or_create_directory(mask_dir)

        # depth path
        if object_i == 0:
            depth_dir = os.path.join(OUTPUT_DIR, "depth")
            empty_or_create_directory(depth_dir)

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
                            material=dict(color="white" if item["key"] == object_i else "black", side=0),
                            ))
        # add single sided walls
        objects.append(Box(material=dict(color="white" if len(layout_data['objects']) == object_i else "black", side=1),
                args=[2.5, 2.5, 2],
                position=[0, 0, 1],
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

        for i, cam_frustum in enumerate(tqdm(cam_frustums)):
            sess.upsert @ cam_frustum
            await sleep(0.1)
            event = await sess.grab_render(key=cam_frustum.key, ttl=5.0)
            path = f"{i:04d}.jpg"
            save_render(event.value['frame'], mask_dir, filename=path)
            if i not in frame_temp.keys():
                frame_temp[i]= {
                    'transform_matrix': cam_extrinsics[i].tolist(),
                    'mask': [os.path.join(os.path.split(mask_dir)[1], path.split(".")[0])]  # blender dataset format for nerfstudio
                }
            else:
                frame_temp[i]['mask'].append(os.path.join(os.path.split(mask_dir)[1], path.split(".")[0]))
            # depth
            if object_i == 0:
                depth_img = event.value["depthFrame"]
                save_render(depth_img, depth_dir, filename=path)
                frame_temp[i]["depth"] = os.path.join(os.path.split(depth_dir)[1], path.split(".")[0])  # blender dataset format for nerfstudio
                
    # Generate the NeRF json file
    for frame_data in frame_temp.values():
        json_data_out["frames"].append(frame_data)

    for split in ["train", "val", "test"]:  # temp hack to use the same dataset for all 3
        with open(os.path.join(OUTPUT_DIR, f"transforms_{split}.json"), 'w') as out_file:
            json.dump(json_data_out, out_file, indent=4)
        # write depth data
    print("Done!")

    while True:
        await asyncio.sleep(1.0)
app.run()
