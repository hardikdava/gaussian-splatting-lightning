import json
from pathlib import Path
from tqdm import tqdm
import subprocess
from internal.utils.colmap import read_images_binary


def generate_appearance_group(workspace_dir: Path):
    images_bin_path =  workspace_dir / "sparse"/ "0"/ "images.bin"

    images = read_images_binary(images_bin_path)
    image_group = {}
    for i in tqdm(images, desc="reading image information"):
        image = images[i]
        key = image.name
        if key not in image_group:
            image_group[key] = []
        image_group[key].append(image.name)

    for i in image_group:
        image_group[i].sort()

    save_path =  workspace_dir / "appearance_groups.json"
    with open(save_path, "w") as f:
        json.dump(image_group, f, indent=4, ensure_ascii=False)
    print(save_path)

def run_train(workspace_dir: Path):
    dry_run = False
    config_path = "sh_view_dependent.yaml"
    iterations = 30000
    training_args = [
                        "python", "main.py", "fit",
                        "--data.parser", "Colmap",
                        "--config", config_path,
                        "--data.path", str(workspace_dir),
                        "--data.parser.appearance_groups", "appearance_groups",
                        "--model.renderer.init_args.optimization.max_steps", str(iterations),
                        "--trainer.limit_val_batches", "0",
                        "--save_iterations", f"[30_000, {iterations}]",
                        "--float32_matmul_precision", "highest",
                        #"--model.density.cull_opacity_threshold", "0.001",
                    ]
                    
    print("***************************")
    print(training_args)
    print("***************************")

    _training_args = [
                        "python", "main.py", "fit",
                        "--data.parser", "Colmap",
                        "--config", config_path,
                        "--data.path", str(workspace_dir),
                        "--data.parser.appearance_groups", "appearance_groups",
                        "--model.renderer.init_args.optimization.max_steps", str(iterations),
                        "--trainer.limit_val_batches", "0",
                        #"--model.density.absgrad", "true",
                        "--model.density.cull_opacity_threshold", "0.0009",
                        "--model.density.densify_grad_threshold", "0.0002",
                        "-v", "high_app_low_opa"
                        
                    ]

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

projects =    ["/data set name here"]

for project in projects:
    workspace = Path(f"data/{project}")
    generate_appearance_group(workspace_dir=workspace)
    run_train(workspace_dir=workspace)
    model_dir = Path(f"outputs/data_{project}/checkpoints/")
    last_ckpt_file = detect_last_ckpt(ckpt_dir=model_dir)
    fuse_ckpt(model_path=last_ckpt_file, data_path=workspace)
    fused_ckpt = model_dir / "fused.ckpt"
    convert_to_ply(ckpt_path=fused_ckpt, ply_path=workspace / f"{project}.ply")
