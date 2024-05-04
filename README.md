# AM220 Project DreamScenes: Text Conditioned 3D Scene Generation via Compositional Diffusion

By Stephen Yang

## Directory Instructions

Generated dataset for evaluation: gen_dataset_old.
Step 1: Base Geometry Generation: geometry_gen
Step 2: Masked Compositional Conditional Diffusion & Step 3: 3D Differentiable Rendering: code integrated into the NeRFStudio pipeline in instruct-gs2gs directory as well as nerfstudio directory.

Evaluation code for generating point cloud from depth: pcd_from_depth.py

## Instructions to run the experiments

Each folder should either contain its own requirement.txt or should only have a handful of dependencies. For Step 1, please refer to the HoloDeck paper and the included MeshGPT training script. Then run ```generate.py```. Example dataset is included as gen_dataset_old.

Please follow NeRFStudio instructions to compile Step 2 and Step 3 described as theh report, and run it using CLI commands using

```pip install -e .```

Example command can be:

```ns-train igs2gs --data gen_dataset_old --load-dir /home/stephen/Desktop/dreamscenes/outputs/unnamed/splatfacto/2024-04-20_223804/nerfstudio_models  --pipeline.guidance-scale 12.5 --pipeline.image-guidance-scale 1.5 blender-data```

If you were to run Step 2 separately, please navigate to ```instruct-gs2gs/igs2gs/conditioning``` folder and run ```main.py```.

Please note that pretrained checkpoints for Stable Diffusion and ControlNet need to be downloaded following instructions from ComfyUI.

