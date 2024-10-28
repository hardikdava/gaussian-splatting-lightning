from typing import Optional
import torch
from .renderer import Renderer
from gsplat_light.hit_pixel_count import hit_pixel_count
from gsplat_light import project_gaussians


DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_ANTI_ALIASED_STATUS: bool = True


class GSplatHitPixelCountRenderer(Renderer):
    @staticmethod
    def hit_pixel_count(
            means3D: torch.Tensor,  # xyz
            opacities: torch.Tensor,
            scales: Optional[torch.Tensor],
            rotations: Optional[torch.Tensor],  # remember to normalize them yourself
            viewpoint_camera,
            scaling_modifier=1.0,
            anti_aliased: bool = True,
            block_size: int = 16,
            extra_projection_kwargs: dict = None,
    ):
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = GSplatHitPixelCountRenderer.project(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            viewpoint_camera=viewpoint_camera,
            scaling_modifier=scaling_modifier,
            block_size=block_size,
            extra_projection_kwargs=extra_projection_kwargs,
        )

        if anti_aliased is True:
            opacities = opacities * comp[:, None]

        count, opacity_score, alpha_score, visibility_score = hit_pixel_count(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            opacities,
            img_height=int(viewpoint_camera.height.item()),
            img_width=int(viewpoint_camera.width.item()),
            block_width=block_size,
        )

        return count, opacity_score, alpha_score, visibility_score

    @staticmethod
    def project(
            means3D: torch.Tensor,  # xyz
            scales: Optional[torch.Tensor],
            rotations: Optional[torch.Tensor],  # remember to normalize them yourself
            viewpoint_camera,
            scaling_modifier=1.0,
            block_size: int = DEFAULT_BLOCK_SIZE,
            extra_projection_kwargs: dict = None,
    ):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        return project_gaussians(  # type: ignore
            means3d=means3D,
            scales=scales,
            glob_scale=scaling_modifier,
            quats=rotations,
            viewmat=viewpoint_camera.world_to_camera.T[:3, :],
            # projmat=viewpoint_camera.full_projection.T,
            fx=viewpoint_camera.fx.item(),
            fy=viewpoint_camera.fy.item(),
            cx=viewpoint_camera.cx.item(),
            cy=viewpoint_camera.cy.item(),
            img_height=img_height,
            img_width=img_width,
            block_width=block_size,
            **({} if extra_projection_kwargs is None else extra_projection_kwargs),
        )
