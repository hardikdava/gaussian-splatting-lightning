import json
import os
from uuid import uuid4
os.environ["TQDM_DISABLE"] = "False"

import time
from pathlib import Path
from supervision.utils.file import read_json_file, save_json_file
import subprocess


def generate_appearance_group(workspace_dir: Path):
    camera_json_file = workspace_dir / "ns" / "transforms.json"
    cameras = read_json_file(file_path=camera_json_file)
    image_group = {}
    for cam in cameras["frames"]:
        image_file = Path(cam["file_path"])
        key = image_file.name
        if key not in image_group:
            image_group[key] = []
        image_group[key].append(image_file.name)

    for i in image_group:
        image_group[i].sort()

    save_path = workspace_dir / "appearance_groups.json"
    with open(save_path, "w") as f:
        json.dump(image_group, f, indent=4, ensure_ascii=False)

def run_train(workspace_dir: Path):
    dry_run = False
    config_path = "sh_view_dependent.yaml"
    iterations = 1000
    model_version = str(uuid4())
    training_args = [
                        "python", "main.py", "fit",
                        "--config", config_path,
                        "--data.parser", "NGP",
                        "--data.path", str(workspace_dir),
                        "--data.parser.appearance_groups", "appearance_groups",
                        "--model.renderer.init_args.optimization.max_steps", str(iterations),
                        "--trainer.limit_val_batches", "0",
                        "--save_iterations", f"[30_000, {iterations}]",
                        "--float32_matmul_precision", "highest",
                        "-v", model_version
                    ]
                    
    print("***************************")
    print(training_args)
    print("***************************")

    if dry_run is False:
        subprocess.run(training_args)
    else:
        print(" ".join(training_args))


def fuse_ckpt(model_path: Path, data_path: Path):
    from fuse_appearance_embeddings_into_shs_dc import main
    main(model_path=model_path.as_posix(), dataset_path=data_path.as_posix())


def convert_to_ply(ckpt_path: Path, ply_path: Path):

    import torch
    from internal.utils.gaussian_utils import GaussianPlyUtils
    from internal.utils.gaussian_model_loader import GaussianModelLoader

    print("Searching checkpoint file...")
    load_file = GaussianModelLoader.search_load_file(ckpt_path.as_posix())

    print(f"Loading checkpoint '{load_file}'...")
    ckpt = torch.load(load_file)
    print("Converting...")
    GaussianPlyUtils.load_from_state_dict(ckpt["state_dict"]).to_ply_format().save_to_ply(ply_path)
    print(f"Saved to '{ply_path}'")


def detect_last_ckpt(ckpt_dir: Path):
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    last_ckpt = None
    last_iteration = 0
    for ckpt_file in ckpt_files:
        name = ckpt_file.stem
        if name == "fused":
            continue
        iteration = name.split("step=")[-1]
        if int(iteration) > last_iteration:
            last_ckpt = ckpt_file
            last_iteration = int(iteration)
    return last_ckpt

projects =    ["cactus"]

for project in projects:
    start_time = time.time()
    workspace = Path(f"data/{project}")
    generate_appearance_group(workspace_dir=workspace)
    model_version = run_train(workspace_dir=workspace)
    model_dir = Path(f"outputs/data_{project}/{model_version}/checkpoints/")
    last_ckpt_file = detect_last_ckpt(ckpt_dir=model_dir)
    fuse_ckpt(model_path=last_ckpt_file, data_path=workspace)
    fused_ckpt = model_dir / "fused.ckpt"
    convert_to_ply(ckpt_path=fused_ckpt, ply_path=workspace / f"{project}.ply")
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
