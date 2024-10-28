from .renderer import *

from gsplat.rendering import rasterization

DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_ANTI_ALIASED_STATUS: bool = True


class GSPlatRenderer(Renderer):
    _RGB_REQUIRED = 1
    _ALPHA_REQUIRED = 1 << 1
    _ACC_DEPTH_REQUIRED = 1 << 2
    _ACC_DEPTH_INVERTED_REQUIRED = 1 << 3
    _EXP_DEPTH_REQUIRED = 1 << 4
    _EXP_DEPTH_INVERTED_REQUIRED = 1 << 5
    _INVERSE_DEPTH_REQUIRED = 1 << 6
    _HARD_DEPTH_REQUIRED = 1 << 7
    _HARD_INVERSE_DEPTH_REQUIRED = 1 << 8

    RENDER_TYPE_BITS = {
        "rgb": _RGB_REQUIRED,
        "alpha": _ALPHA_REQUIRED | _ACC_DEPTH_REQUIRED,
        "acc_depth": _ACC_DEPTH_REQUIRED,
        "acc_depth_inverted": _ACC_DEPTH_REQUIRED | _ACC_DEPTH_INVERTED_REQUIRED,
        "exp_depth": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED,
        "exp_depth_inverted": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED | _EXP_DEPTH_INVERTED_REQUIRED,
        "inverse_depth": _INVERSE_DEPTH_REQUIRED,
        "hard_depth": _HARD_DEPTH_REQUIRED,
        "hard_inverse_depth": _HARD_INVERSE_DEPTH_REQUIRED,
    }

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE, anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS) -> None:
        super().__init__()
        self.block_size = block_size
        self.anti_aliased = anti_aliased

    def parse_render_types(self, render_types: list) -> int:
        if render_types is None:
            return self._RGB_REQUIRED
        else:
            bits = 0
            for i in render_types:
                bits |= self.RENDER_TYPE_BITS[i]
            return bits

    @staticmethod
    def is_type_required(bits: int, type: int) -> bool:
        return bits & type != 0

    def forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):

        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        K = torch.tensor([[viewpoint_camera.fx.item(), 0, viewpoint_camera.cx.item()],
                          [0, viewpoint_camera.fy.item(), viewpoint_camera.cy.item()],
                            [0, 0, 1]]).unsqueeze(0).to(viewpoint_camera.R.device)

        opacities = pc.get_opacity.squeeze(-1)
        colors = pc.get_features

        viewmats = viewpoint_camera.world_to_camera.T  # [1, 4, 4]
        viewmats = viewmats.unsqueeze(0)

        render, alpha, info = rasterization(
            means=pc.get_xyz,
            quats=pc.get_rotation / pc.get_rotation.norm(dim=-1, keepdim=True),  # rasterization does normalization internally
            scales=pc.get_scaling,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=img_width,
            height=img_height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=pc.active_sh_degree,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="antialiased",
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        alpha = alpha[:, ...]
        rgb = render[:, ..., :3] + (1 - alpha) * bg_color
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb = rgb.squeeze(0)
        rgb = rgb.permute(2, 0, 1)
        alpha = alpha.squeeze(0)
        info["means2d"].retain_grad()
        acc_depth_im = None
        acc_depth_inverted_im = None
        exp_depth_im = None
        exp_depth_inverted_im = None
        inverse_depth_im = None
        hard_depth_im = None
        hard_inverse_depth_im = None

        radii = info["radii"].squeeze(0)
        # xys = info["means2d"].squeeze(0)

        return {
            "render": rgb,
            "alpha": alpha,
            "acc_depth": acc_depth_im,
            "acc_depth_inverted": acc_depth_inverted_im,
            "exp_depth": exp_depth_im,
            "exp_depth_inverted": exp_depth_inverted_im,
            "inverse_depth": inverse_depth_im,
            "hard_depth": hard_depth_im,
            "hard_inverse_depth": hard_inverse_depth_im,
            "viewspace_points": info["means2d"],
            "viewspace_points_grad_scale": 0.5 * torch.tensor([[img_width, img_height]]).to(info["means2d"].device),
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            "alpha": RendererOutputInfo("alpha", type=RendererOutputTypes.GRAY),
            "acc_depth": RendererOutputInfo("acc_depth", type=RendererOutputTypes.GRAY),
            "acc_depth_inverted": RendererOutputInfo("acc_depth_inverted", type=RendererOutputTypes.GRAY),
            "exp_depth": RendererOutputInfo("exp_depth", type=RendererOutputTypes.GRAY),
            "exp_depth_inverted": RendererOutputInfo("exp_depth_inverted", type=RendererOutputTypes.GRAY),
            "inverse_depth": RendererOutputInfo("inverse_depth", type=RendererOutputTypes.GRAY),
            "hard_depth": RendererOutputInfo("hard_depth", type=RendererOutputTypes.GRAY),
            "hard_inverse_depth": RendererOutputInfo("hard_inverse_depth", type=RendererOutputTypes.GRAY),
        }
