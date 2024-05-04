import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import json
import argparse
import shutil

module_path = os.path.abspath("instruct-gs2gs/igs2gs/conditioning")
if module_path not in sys.path:
    sys.path.append(module_path)

from igs2gs.conditioning.nodes import (
    EmptyLatentImage,
    VAEEncode,
    ControlNetLoader,
    LoadImage,
    CheckpointLoaderSimple,
    NODE_CLASS_MAPPINGS,
    ControlNetApply,
    ConditioningSetMask,
    SaveImage,
    CLIPTextEncode,
    ConditioningCombine,
    VAEDecode,
)

class CompositDiffusion():
    def __init__(self) -> None:
        self.add_comfyui_directory_to_sys_path()
        self.add_extra_model_paths()

        self.import_custom_nodes()

        # TODO wrap the following
        with torch.inference_mode():
            self.emptylatentimage = EmptyLatentImage()

            checkpointloadersimple = CheckpointLoaderSimple()
            self.sdxl_ckpt = checkpointloadersimple.load_checkpoint(
                ckpt_name="sd_xl_turbo_1.0_fp16.safetensors"
            )

            self.cliptextencode = CLIPTextEncode()

            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            self.ksamplerselect_14 = ksamplerselect.get_sampler(sampler_name="lcm")

            controlnetloader = ControlNetLoader()
            self.controlnetloader_31 = controlnetloader.load_controlnet(
                control_net_name="controlnet_depth_sdxl_1.0.safetensors"
            )

            self.loadimage = LoadImage()

            self.imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
            self.conditioningsetmask = ConditioningSetMask()
            self.conditioningcombine = ConditioningCombine()
            self.controlnetapply = ControlNetApply()
            self.sdturboscheduler = NODE_CLASS_MAPPINGS["SDTurboScheduler"]()
            self.samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
            self.vaeencode = VAEEncode()
            self.vaedecode = VAEDecode()
            self.saveimage = SaveImage()

    def get_value_at_index(self, obj: Union[Sequence, Mapping], index: int) -> Any:
        try:
            return obj[index]
        except KeyError:
            return obj["result"][index]


    def find_path(self, name: str, path: str = None) -> str:
        # If no path is given, use the current working directory
        if path is None:
            path = os.getcwd()

        # Check if the current directory contains the name
        if name in os.listdir(path):
            path_name = os.path.join(path, name)
            print(f"{name} found: {path_name}")
            return path_name

        # Get the parent directory
        parent_directory = os.path.dirname(path)

        # If the parent directory is the same as the current directory, we've reached the root and stop the search
        if parent_directory == path:
            return None

        # Recursively call the function with the parent directory
        return self.find_path(name, parent_directory)


    def add_comfyui_directory_to_sys_path(self) -> None:
        comfyui_path = self.find_path("ComfyUI")
        if comfyui_path is not None and os.path.isdir(comfyui_path):
            sys.path.append(comfyui_path)
            print(f"'{comfyui_path}' added to sys.path")


    def add_extra_model_paths(self) -> None:

        from comfyui import load_extra_path_config

        extra_model_paths = self.find_path("extra_model_paths.yaml")

        if extra_model_paths is not None:
            load_extra_path_config(extra_model_paths)
        else:
            print("Could not find the extra_model_paths config file.")

    def import_custom_nodes(self) -> None:

        import asyncio
        import execution
        from nodes import init_custom_nodes
        import server

        # Creating a new event loop and setting it as the default loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Creating an instance of PromptServer with the loop
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # Initializing custom nodes
        init_custom_nodes()
        
    def condition(self, current_render_path, new_path, depth, *masks):
        
        with torch.inference_mode():
            # VAE encode for update step
            if current_render_path == None:
                encoded_current_render = self.emptylatentimage.generate(width=1000, height=1000, batch_size=1)
            else:
                current_render = self.loadimage.load_image(image=current_render_path)
                encoded_current_render = self.vaeencode.encode(
                    pixels=self.get_value_at_index(current_render, 0),
                    vae=self.get_value_at_index(self.sdxl_ckpt, 2),
                )

            depth_img = self.loadimage.load_image(image=depth)
            loadimage_49 = self.loadimage.load_image(image=masks[0])
            loadimage_103 = self.loadimage.load_image(image=masks[2])
            loadimage_110 = self.loadimage.load_image(image=masks[3])
            loadimage_67 = self.loadimage.load_image(image=masks[1])
            loadimage_113 = self.loadimage.load_image(image=masks[4])
            loadimage_118 = self.loadimage.load_image(image=masks[5])

            self.cliptextencode_6 = self.cliptextencode.encode(
                text="a white textured office table, with a open book on it.",
                clip=self.get_value_at_index(self.sdxl_ckpt, 1),
            )
            self.cliptextencode_56 = self.cliptextencode.encode(
                text="nonrealistic, nontextured",
                clip=self.get_value_at_index(self.sdxl_ckpt, 1),
            )

            self.cliptextencode_76 = self.cliptextencode.encode(
                text="a black real leather made office chair that shows wear with time.",
                clip=self.get_value_at_index(self.sdxl_ckpt, 1),
            )

            self.cliptextencode_96 = self.cliptextencode.encode(
                text="A great antique style, walnet colored wooden night stand.",
                clip=self.get_value_at_index(self.sdxl_ckpt, 1),
            )
            self.cliptextencode_111 = self.cliptextencode.encode(
                text="a wooden bed with white colored textured duvet.",
                clip=self.get_value_at_index(self.sdxl_ckpt, 1),
            )
            self.cliptextencode_116 = self.cliptextencode.encode(
                text="A red wood closet that shows fine tuxtures",
                clip=self.get_value_at_index(self.sdxl_ckpt, 1),
            )
            self.cliptextencode_121 = self.cliptextencode.encode(
                text="A light yellow wooden floor with texture, and white walls all around",
                clip=self.get_value_at_index(self.sdxl_ckpt, 1),
            )

            imagetomask_69 = self.imagetomask.image_to_mask(
                channel="red", image=self.get_value_at_index(loadimage_49, 0)
            )

            conditioningsetmask_91 = self.conditioningsetmask.append(
                strength=0.8,
                set_cond_area="default",
                conditioning=self.get_value_at_index(self.cliptextencode_6, 0),
                mask=self.get_value_at_index(imagetomask_69, 0),
            )

            imagetomask_70 = self.imagetomask.image_to_mask(
                channel="red", image=self.get_value_at_index(loadimage_67, 0)
            )

            conditioningsetmask_90 = self.conditioningsetmask.append(
                strength=0.8,
                set_cond_area="default",
                conditioning=self.get_value_at_index(self.cliptextencode_76, 0),
                mask=self.get_value_at_index(imagetomask_70, 0),
            )

            conditioningcombine_92 = self.conditioningcombine.combine(
                conditioning_1=self.get_value_at_index(conditioningsetmask_91, 0),
                conditioning_2=self.get_value_at_index(conditioningsetmask_90, 0),
            )

            imagetomask_109 = self.imagetomask.image_to_mask(
                channel="red", image=self.get_value_at_index(loadimage_110, 0)
            )

            conditioningsetmask_108 = self.conditioningsetmask.append(
                strength=0.8,
                set_cond_area="default",
                conditioning=self.get_value_at_index(self.cliptextencode_111, 0),
                mask=self.get_value_at_index(imagetomask_109, 0),
            )

            conditioningcombine_107 = self.conditioningcombine.combine(
                conditioning_1=self.get_value_at_index(conditioningcombine_92, 0),
                conditioning_2=self.get_value_at_index(conditioningsetmask_108, 0),
            )

            imagetomask_104 = self.imagetomask.image_to_mask(
                channel="red", image=self.get_value_at_index(loadimage_103, 0)
            )

            conditioningsetmask_105 = self.conditioningsetmask.append(
                strength=0.8,
                set_cond_area="default",
                conditioning=self.get_value_at_index(self.cliptextencode_96, 0),
                mask=self.get_value_at_index(imagetomask_104, 0),
            )

            conditioningcombine_106 = self.conditioningcombine.combine(
                conditioning_1=self.get_value_at_index(conditioningcombine_107, 0),
                conditioning_2=self.get_value_at_index(conditioningsetmask_105, 0),
            )

            imagetomask_114 = self.imagetomask.image_to_mask(
                channel="red", image=self.get_value_at_index(loadimage_113, 0)
            )

            conditioningsetmask_115 = self.conditioningsetmask.append(
                strength=0.8,
                set_cond_area="default",
                conditioning=self.get_value_at_index(self.cliptextencode_116, 0),
                mask=self.get_value_at_index(imagetomask_114, 0),
            )

            conditioningcombine_117 = self.conditioningcombine.combine(
                conditioning_1=self.get_value_at_index(conditioningcombine_106, 0),
                conditioning_2=self.get_value_at_index(conditioningsetmask_115, 0),
            )

            imagetomask_119 = self.imagetomask.image_to_mask(
                channel="red", image=self.get_value_at_index(loadimage_118, 0)
            )

            conditioningsetmask_120 = self.conditioningsetmask.append(
                strength=0.8,
                set_cond_area="default",
                conditioning=self.get_value_at_index(self.cliptextencode_121, 0),
                mask=self.get_value_at_index(imagetomask_119, 0),
            )

            conditioningcombine_122 = self.conditioningcombine.combine(
                conditioning_1=self.get_value_at_index(conditioningcombine_117, 0),
                conditioning_2=self.get_value_at_index(conditioningsetmask_120, 0),
            )

            controlnetapply_59 = self.controlnetapply.apply_controlnet(
                strength=0.65,
                conditioning=self.get_value_at_index(conditioningcombine_122, 0),
                control_net=self.get_value_at_index(self.controlnetloader_31, 0),
                image=self.get_value_at_index(depth_img, 0),
            )

            sdturboscheduler_22 = self.sdturboscheduler.get_sigmas(
                steps=6,
                denoise=0.21,
                model=self.get_value_at_index(self.sdxl_ckpt, 0),
            )

            samplercustom_13 = self.samplercustom.sample(
                add_noise=True,
                noise_seed=random.randint(1, 2**64),
                cfg=1,
                model=self.get_value_at_index(self.sdxl_ckpt, 0),
                positive=self.get_value_at_index(controlnetapply_59, 0),
                negative=self.get_value_at_index(self.cliptextencode_56, 0),
                sampler=self.get_value_at_index(self.ksamplerselect_14, 0),
                sigmas=self.get_value_at_index(sdturboscheduler_22, 0),
                latent_image=self.get_value_at_index(encoded_current_render, 0),
                # latent_image=self.get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_8 = self.vaedecode.decode(
                samples=self.get_value_at_index(samplercustom_13, 0),
                vae=self.get_value_at_index(self.sdxl_ckpt, 2),
            )

            return self.saveimage.save_images(
                new_path, images=self.get_value_at_index(vaedecode_8, 0)
                )

    def empty_or_create_directory(self, path):
        if os.path.exists(path) and os.path.isdir(path):
            # Remove all contents of the directory
            for filename in os.listdir(path):
                file_path = os.path.join(path, os.path.basename(filename))
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            os.makedirs(path, exist_ok=True)

        # condition("0014 (6).jpg", "0014 (7).jpg", "0014 (8).jpg", "0014 (9).jpg", "0014 (10).jpg", "0014 (11).jpg", "0014 (12).jpg")

    def create_dir_with_prompt(self, directory):
        if os.path.exists(directory):
            input(f"The directory {directory} already exists. Press Enter to overwrite.")
        
        self.empty_or_create_directory(directory)   

def main():
    composite_diffusion = CompositDiffusion()
    parser = argparse.ArgumentParser(description='Process some JSON data.')
    parser.add_argument('--path', type=str, default='gen_dataset',
                        help='Path to the data (default: gen_dataset)')
    
    args = parser.parse_args()
    
    json_file = os.path.join(args.path, "transforms_train.json")
    composite_diffusion.create_dir_with_prompt(os.path.join(args.path, "images"))
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    i = 0
    for frame in frames:
        depth_path = os.path.join(args.path, frame['depth']+".jpg")
        mask_paths = [os.path.join(args.path, mask_path+".jpg") for mask_path in frame['mask']]
        new_path = os.path.join(args.path, os.path.join('images', os.path.basename(frame['depth'])+".png"))
        composite_diffusion.condition(None, new_path, depth_path, *mask_paths)
        # Update frame information
        frame['file_path'] = os.path.join("images", os.path.basename(frame['depth']))

    for split in ["train", "val", "test"]:  # temp hack to use the same dataset for all 3
        with open(os.path.join(args.path, f"transforms_{split}.json"), 'w') as out_file:
            json.dump(data, out_file, indent=4)


if __name__ == "__main__":
    main()

