import torch
from diffusers import FluxPipeline
from PIL import Image
import argparse
import random
import numpy as np
import yaml
import os
import time
import csv

from wssc import WSSC


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0,
                        help="GPU device number")
    parser.add_argument("--exp_yaml", type=str, default="wssc_config.yaml",
                        help="experiment yaml file")
    parser.add_argument("--model_path", type=str,
                        default="./models/FLUX.1-dev",
                        help="path to the FLUX.1-dev model")

    args = parser.parse_args()

    device_number = args.device_number
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    exp_yaml = args.exp_yaml
    with open(exp_yaml) as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    model_type = exp_configs[0]["model_type"]

    if model_type == 'FLUX':
        pipe = FluxPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    scheduler = pipe.scheduler
    print(pipe.scheduler.__class__.__name__)
    print(type(pipe.scheduler))
    pipe = pipe.to(device)

    for exp_dict in exp_configs:

        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]
        spectral_boost = exp_dict.get("spectral_boost", 3.0)
        use_mask = exp_dict.get("use_mask", True)
        n_levels = exp_dict.get("n_levels", 3)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        dataset_yaml = exp_dict["dataset_yaml"]
        with open(dataset_yaml) as file:
            dataset_configs = yaml.load(file, Loader=yaml.FullLoader)

        processing_times = []
        csv_dir = f"./outputs/timing/{exp_name}/{model_type}"
        csv_file_path = f"{csv_dir}/processing_times.csv"
        os.makedirs(csv_dir, exist_ok=True)

        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Image Path', 'Source Prompt',
                                 'Target Prompt', 'Processing Time (seconds)'])

        for data_dict in dataset_configs:

            src_prompt = data_dict["source_prompt"]
            tar_prompts = data_dict["target_prompts"]
            negative_prompt = ""
            image_src_path = data_dict["input_img"]
            mask_path = data_dict.get("mask_path", None)

            image = Image.open(image_src_path)
            print(image_src_path)

            # crop image to have both dimensions divisible by 16
            image = image.crop((0, 0, image.width - image.width % 16,
                                image.height - image.height % 16))
            print(image.width, image.height)
            image_src = pipe.image_processor.preprocess(image)
            image_src = image_src.to(device).half()
            with torch.autocast("cuda"), torch.inference_mode():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
            x0_src = ((x0_src_denorm - pipe.vae.config.shift_factor)
                      * pipe.vae.config.scaling_factor)
            x0_src = x0_src.to(device)

            for tar_num, tar_prompt in enumerate(tar_prompts):

                start_time = time.time()
                x0_tar = WSSC(
                    pipe=pipe,
                    scheduler=scheduler,
                    x_src=x0_src,
                    src_prompt=src_prompt,
                    tar_prompt=tar_prompt,
                    negative_prompt=negative_prompt,
                    T_steps=T_steps,
                    n_avg=n_avg,
                    src_guidance_scale=src_guidance_scale,
                    tar_guidance_scale=tar_guidance_scale,
                    n_min=n_min,
                    n_max=n_max,
                    mask_path=mask_path,
                    spectral_boost=spectral_boost,
                    use_mask=use_mask,
                    n_levels=n_levels,
                )

                end_time = time.time()
                x0_tar_denorm = ((x0_tar / pipe.vae.config.scaling_factor)
                                 + pipe.vae.config.shift_factor)
                with torch.autocast("cuda"), torch.inference_mode():
                    image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                image_tar = pipe.image_processor.postprocess(image_tar)

                # determine relative path for saving
                if "PIE-Bench" in image_src_path:
                    relative_path = image_src_path.split("PIE-Bench")[1].lstrip('/')
                    relative_path = os.path.dirname(relative_path)
                else:
                    path_parts = image_src_path.split('/')
                    relative_path = ('/'.join(path_parts[-3:-1])
                                     if len(path_parts) > 2 else path_parts[-2])

                image_filename = os.path.basename(image_src_path)

                save_dir = (f"./outputs/{exp_name}/{model_type}"
                            f"_sb{spectral_boost}_mask{int(use_mask)}"
                            f"/{relative_path}")
                os.makedirs(save_dir, exist_ok=True)

                output_path = f"{save_dir}/{image_filename}"
                print(output_path)

                image_tar[0].save(output_path)

                with open(f"{save_dir}/prompts.txt", "w") as f:
                    f.write(f"Source prompt: {src_prompt}\n")
                    f.write(f"Target prompt: {tar_prompt}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Sampler type: {model_type}\n")

                processing_time = end_time - start_time
                processing_times.append(processing_time)

                with open(csv_file_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([output_path, src_prompt,
                                         tar_prompt, processing_time])

    print("Done")
