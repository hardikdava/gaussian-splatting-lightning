from typing import Literal
import os
import argparse
import torch
from tqdm.auto import tqdm
from internal.renderers.gsplat_renderer import GSPlatRenderer
from internal.renderers.gsplat_hit_pixel_count_renderer import GSplatHitPixelCountRenderer
from internal.utils.sh_utils import RGB2SH
from internal.utils.gaussian_model_loader import GaussianModelLoader


@torch.no_grad()
def calculate_gaussian_scores(cameras, gaussian_model, device):
    hit_count_list = []
    opacity_score_list = []
    alpha_score_list = []
    all_visibility_score = torch.zeros((len(cameras), gaussian_model.get_xyz.shape[0]), dtype=torch.float, device=device)
    for idx, camera in tqdm(enumerate(cameras), total=len(cameras), leave=False, desc="Calculating gaussian visibilities"):
        hit_count, opacity_score, alpha_score, visibility_score = GSplatHitPixelCountRenderer.hit_pixel_count(
            means3D=gaussian_model.get_xyz,
            opacities=gaussian_model.get_opacity,
            scales=gaussian_model.get_scaling,
            rotations=gaussian_model.get_rotation,
            viewpoint_camera=camera.to_device("cuda"),
        )
        # hit_count_list.append(hit_count.cpu())
        # opacity_score_list.append(opacity_score.cpu())
        # alpha_score_list.append(alpha_score.cpu())
        all_visibility_score[idx] = visibility_score.to(device=device)
        # visibility_score_list.append(visibility_score.cpu())

    torch.cuda.empty_cache()

    return all_visibility_score


@torch.no_grad()
def prune_gaussian_model(gaussian_model, mask):
    gaussian_model.properties = {k: v[mask] for k, v in gaussian_model.properties.items()}


@torch.no_grad()
def prune_and_get_weights(gaussian_model, cameras, n_average_cameras, weight_device: torch.device):
    # calculate Gaussians' visibility score to each camera; the output may consume a lot of memory, so put it on CPU
    visibility_score = calculate_gaussian_scores(cameras, gaussian_model, "cpu").T  # [N_gaussians, N_cameras]
    # calculate total visibility score for each Gaussian
    visibility_score_acc = torch.sum(visibility_score, dim=-1)
    # find Gaussian whose total visibility is closed to zero
    visibility_score_acc_is_close_to_zero = torch.isclose(visibility_score_acc, torch.tensor(0., device=visibility_score_acc.device))
    gaussian_to_preserve = ~visibility_score_acc_is_close_to_zero
    # prune
    prune_gaussian_model(gaussian_model, gaussian_to_preserve.to(device=gaussian_model.means.device))
    visibility_score_pruned = visibility_score[~visibility_score_acc_is_close_to_zero]
    del visibility_score
    visibility_score_acc_is_close_to_zero.sum()

    # ===

    # get top `n_average_cameras` visibility cameras
    visibility_score_pruned_sorted = torch.topk(visibility_score_pruned, k=n_average_cameras, dim=-1)
    visibility_score_pruned_sorted_values = visibility_score_pruned_sorted.values.to(device=weight_device)
    visibility_score_pruned_sorted_indices = visibility_score_pruned_sorted.indices.to(device=weight_device)
    del visibility_score_pruned_sorted

    visibility_score_pruned_top_k_acc = torch.sum(visibility_score_pruned_sorted_values, dim=-1, keepdim=True)
    # calculate the weight of each camera
    visibility_score_pruned_top_k_pdf = visibility_score_pruned_sorted_values / visibility_score_pruned_top_k_acc

    assert torch.all(torch.isclose(visibility_score_pruned_top_k_pdf.sum(dim=-1), torch.tensor(1., device=visibility_score_pruned_top_k_pdf.device)))

    return visibility_score_pruned_sorted_values, visibility_score_pruned_sorted_indices, visibility_score_pruned_top_k_pdf


@torch.no_grad()
def average_color_fusing(
        gaussian_model,
        renderer,
        n_average_cameras: int,
        camera_chunk_size: int,
        cameras: list,
        device: torch.device,
        visibility_score_pruned_sorted_indices: torch.Tensor,  # [N_gaussians, N_cameras]
        visibility_score_pruned_top_k_pdf: torch.Tensor,  # [N_gaussians, N_cameras]
):
    cuda_device = torch.device("cuda")

    n_chunk = (n_average_cameras + camera_chunk_size - 1) // camera_chunk_size

    rgb_offset_for_each_camera = torch.ones((gaussian_model.n_gaussians, n_average_cameras, 3), dtype=torch.float, device=device) * -1024.
    for i in range(n_chunk):
        appearance_model_input_feature_list = []

        left = i * camera_chunk_size
        right = min(left + camera_chunk_size, n_average_cameras)
        real_chunk_size = right - left

        appearance_features = gaussian_model.get_appearance_features().unsqueeze(1).repeat(1, real_chunk_size, 1)
        appearance_model_input_feature_list.append(appearance_features)

        # pick appearance id
        camera_index_to_appearance_id = torch.tensor([i.appearance_id for i in cameras], dtype=torch.int, device=visibility_score_pruned_sorted_indices.device)
        appearance_ids = camera_index_to_appearance_id[visibility_score_pruned_sorted_indices[..., left:right].reshape(-1)]  # [N_gaussians * real_chunk_size]
        # pick appearance embeddings
        appearance_embeddings = renderer.model.embedding(appearance_ids).reshape((
            visibility_score_pruned_sorted_indices.shape[0],
            real_chunk_size,
            -1,
        ))  # [N_gaussians, real_chunk_size, N_embedding_dims]
        appearance_model_input_feature_list.append(appearance_embeddings)

        if renderer.model_config.is_view_dependent:
            camera_index_to_camera_center = torch.stack(
                [i.camera_center for i in cameras],
            ).to(device=cuda_device)  # [N_cameras, 3]

            camera_centers = camera_index_to_camera_center[visibility_score_pruned_sorted_indices[..., left:right]]  # [N_gaussians, real_chunk_size, 3]
            view_directions = torch.nn.functional.normalize(gaussian_model.get_means().unsqueeze(1) - camera_centers, dim=-1)  # [N_gaussians, n_average_cameras, 3]
            encoded_view_directions = renderer.model.view_direction_encoding(view_directions)
            appearance_model_input_feature_list.append(encoded_view_directions)

        input_features = torch.concat(appearance_model_input_feature_list, dim=-1).reshape((gaussian_model.n_gaussians * real_chunk_size, -1))

        rgb_offset_for_each_camera[:, left:right, :] = renderer.model.network(input_features).reshape((gaussian_model.n_gaussians, real_chunk_size, -1)).to(device=device)

    # make sure all the elements are filled
    assert not torch.any(torch.isclose(rgb_offset_for_each_camera, torch.tensor(-1024., device=device)))

    weighted_rgb_offset_for_each_camera = rgb_offset_for_each_camera * visibility_score_pruned_top_k_pdf.unsqueeze(-1).to(device=device)
    rgb_offset = torch.sum(weighted_rgb_offset_for_each_camera, dim=1)

    return rgb_offset


@torch.no_grad()
def average_embedding_fusing(
        gaussian_model,
        renderer,
        n_average_cameras: int,
        cameras: list,
        visibility_score_pruned_sorted_indices: torch.Tensor,  # [N_gaussians, N_cameras]
        visibility_score_pruned_top_k_pdf: torch.Tensor,  # [N_gaussians, N_cameras]
        view_dir_average_mode: str = Literal["camera_center", "view_direction"],
):
    cuda_device = torch.device("cuda")

    # pick appearance id
    camera_index_to_appearance_id = torch.tensor([i.appearance_id for i in cameras], dtype=torch.int, device=visibility_score_pruned_sorted_indices.device)
    appearance_ids = camera_index_to_appearance_id[visibility_score_pruned_sorted_indices.reshape(-1)]  # [N_gaussians * n_average_cameras]
    # pick appearance embeddings
    appearance_embeddings = renderer.model.embedding(appearance_ids).reshape((
        visibility_score_pruned_sorted_indices.shape[0],
        n_average_cameras,
        -1,
    ))  # [N_gaussians, n_average_cameras, N_embedding_dims]

    # multiply embeddings by camera weights
    weighted_appearance_embeddings = appearance_embeddings * visibility_score_pruned_top_k_pdf.unsqueeze(-1)
    # merge `n_average_cameras` embedding to a single embedding
    final_appearance_embeddings = torch.sum(weighted_appearance_embeddings, dim=1)

    # embedding network forward, output rgb_offset
    embedding_network = renderer.model.network
    input_tensor_list = [
        gaussian_model.get_appearance_features(),
        final_appearance_embeddings,
    ]

    # view dependent
    if renderer.model_config.is_view_dependent:
        camera_index_to_camera_center = torch.stack(
            [i.camera_center for i in cameras],
        ).to(device=cuda_device)  # [N_cameras, 3]
        camera_centers = camera_index_to_camera_center[visibility_score_pruned_sorted_indices]  # [N_gaussians, n_average_cameras, 3]

        # not sure which one below is better

        if view_dir_average_mode == "camera_center":
            # [OPTION 1] weighted camera centers
            weighted_camera_centers = camera_centers * visibility_score_pruned_top_k_pdf.unsqueeze(-1)
            final_camera_centers = torch.sum(weighted_camera_centers, dim=1)  # [N_gaussians, 3]
            view_directions = torch.nn.functional.normalize(gaussian_model.get_means() - final_camera_centers, dim=-1)
        else:
            # [OPTION 2] weighted view directions
            unweighted_view_directions = torch.nn.functional.normalize(gaussian_model.get_means().unsqueeze(1) - camera_centers, dim=-1)  # [N_gaussians, n_average_cameras, 3]
            weighted_view_directions = unweighted_view_directions * visibility_score_pruned_top_k_pdf.unsqueeze(-1)
            view_directions = torch.nn.functional.normalize(torch.sum(weighted_view_directions, dim=1), dim=-1)  # [N_gaussians, 3]

        encoded_view_directions = renderer.model.view_direction_encoding(view_directions)

        input_tensor_list.append(encoded_view_directions)

    input_tensor = torch.concat(input_tensor_list, dim=-1).to(cuda_device)
    rgb_offset = embedding_network(input_tensor).clamp(min=0., max=1.)

    return rgb_offset


@torch.no_grad()
def fuse(
        ckpt,
        device: torch.device,
        n_average_cameras: int,
        camera_chunk_size: int,
        mode ="color",
        embedding_view_dir_mode = "view_direction",
        dataset_path_override: str = None
):
    dataparser_outputs = ckpt["datamodule_hyper_parameters"]["parser"].instantiate(
        path=ckpt["datamodule_hyper_parameters"]["path"] if dataset_path_override is None else dataset_path_override,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()

    cuda_device = torch.device("cuda")

    # get cameras
    cameras = dataparser_outputs.train_set.cameras

    # ===

    gaussian_model = GaussianModelLoader.initialize_model_from_checkpoint(
        ckpt,
        cuda_device,
    )
    # prune those outside bounding box
    gaussian_model.to(device=cuda_device)
    # get renderer
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(ckpt, stage="validation", device=cuda_device)

    # ===

    _, visibility_score_pruned_sorted_indices, visibility_score_pruned_top_k_pdf = prune_and_get_weights(
        gaussian_model=gaussian_model,
        cameras=cameras,
        n_average_cameras=n_average_cameras,
        weight_device=cuda_device,
    )

    # ===

    if mode == "color":
        rgb_offset = average_color_fusing(
            gaussian_model,
            renderer,
            n_average_cameras=n_average_cameras,
            camera_chunk_size=camera_chunk_size,
            cameras=cameras,
            device=device,
            visibility_score_pruned_sorted_indices=visibility_score_pruned_sorted_indices,
            visibility_score_pruned_top_k_pdf=visibility_score_pruned_top_k_pdf,
        )
    else:
        rgb_offset = average_embedding_fusing(
            gaussian_model,
            renderer,
            n_average_cameras=n_average_cameras,
            cameras=cameras,
            visibility_score_pruned_sorted_indices=visibility_score_pruned_sorted_indices,
            visibility_score_pruned_top_k_pdf=visibility_score_pruned_top_k_pdf,
            view_dir_average_mode=embedding_view_dir_mode,
        )

    # ===

    sh_offset = RGB2SH(rgb_offset)
    gaussian_model.shs_dc = gaussian_model.shs_dc + sh_offset.unsqueeze(1).to(device=gaussian_model.shs_dc.device)

    return gaussian_model


@torch.no_grad()
def update_ckpt(gaussian_model, ckpt):
    # remove `GSplatAppearanceEmbeddingRenderer`'s states from ckpt
    state_dict_key_to_delete = []
    for i in ckpt["state_dict"]:
        if i.startswith("renderer."):
            state_dict_key_to_delete.append(i)
    for i in state_dict_key_to_delete:
        del ckpt["state_dict"][i]

    ckpt["optimizer_states"] = []

    # replace `AppearanceFeatureGaussian` with `VanillaGaussian`
    from internal.models.vanilla_gaussian import VanillaGaussian
    ckpt["hyper_parameters"]["gaussian"] = VanillaGaussian(sh_degree=gaussian_model.max_sh_degree)

    # remove existing Gaussians from ckpt
    for i in list(ckpt["state_dict"].keys()):
        if i.startswith("gaussian_model.gaussians."):
            del ckpt["state_dict"][i]

    # replace `GSplatAppearanceEmbeddingRenderer` with `GSPlatRenderer`
    ckpt["hyper_parameters"]["renderer"] = GSPlatRenderer()

    ckpt["state_dict"]["gaussian_model.gaussians.means"] = gaussian_model.means
    ckpt["state_dict"]["gaussian_model.gaussians.shs_dc"] = gaussian_model.shs_dc
    ckpt["state_dict"]["gaussian_model.gaussians.shs_rest"] = gaussian_model.shs_rest
    ckpt["state_dict"]["gaussian_model.gaussians.scales"] = gaussian_model.scales
    ckpt["state_dict"]["gaussian_model.gaussians.rotations"] = gaussian_model.rotations
    ckpt["state_dict"]["gaussian_model.gaussians.opacities"] = gaussian_model.opacities


def main(model_path, dataset_path, output=None):
    device = "cpu"
    n_average_cameras = 2
    image_dir = os.path.join(dataset_path, "images")
    n_images =  len([name for name in os.listdir(image_dir) if os.path.isfile(name)])
    #n_average_cameras = min(n_average_cameras, n_images)
    camera_chunk_size = 2
    #camera_chunk_size = min(camera_chunk_size, n_images)
    embedding_view_dir_mode = "view_direction"
    ckpt_file = GaussianModelLoader.search_load_file(model_path)
    assert ckpt_file.endswith(".ckpt") is True

    if output is None:
        output = os.path.join(os.path.dirname(ckpt_file), "fused.ckpt")
        
    ckpt = torch.load(ckpt_file, map_location="cpu")

    gaussian_model = fuse(
        ckpt,
        device=torch.device(device),
        n_average_cameras=n_average_cameras,
        camera_chunk_size=camera_chunk_size,
        mode="color",
        embedding_view_dir_mode=embedding_view_dir_mode,
        dataset_path_override=dataset_path,
    )

    update_ckpt(gaussian_model, ckpt)

    torch.save(ckpt, output)

    print("Saved to '{}'".format(output))


if __name__ == "__main__":
    main()
