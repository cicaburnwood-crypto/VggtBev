#!/usr/bin/env python3
"""Single-GPU RGB visual-navigation baseline runtime.

PointGoal methods accept RGB observations plus an ego-local metric target.
NoMaD instead receives the RGB image at the same current pre-planned subgoal,
because the published model is natively ImageGoal-conditioned. Habitat GT,
navmesh, depth and simulator poses are not part of the learned-model request.
Models are loaded lazily and cached after first use.
"""

from __future__ import annotations

import argparse
import base64
import gc
import fcntl
import importlib.util
import io
import json
import math
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from native_platform_controls import (
    RobotPlatform,
    nomad_control,
    nomad_metric_spacing,
    omni_control,
    curvature_limit,
    footprint_pixels,
)
from robot_contract import SPEC, SAFETY_MARGIN_M


METHODS = {
    "straight_line": {
        "label": "Straight-Line",
        "input": "ego-local metric target only (RGB deliberately ignored)",
        "privileged": False,
        "recommended_inference_hz": 10.0,
        "controller": "direct-heading differential-drive controller",
    },
    "limo_tel": {
        "label": "LiMo-TEL",
        "input": "latest RGB + SE(2) point goal",
        "privileged": False,
        "recommended_inference_hz": 10.0,
        "controller": "native SE(2) path; benchmark differential-drive tracker (not a released LiMo controller)",
    },
    "limo_aug": {
        "label": "LiMo-AUG",
        "input": "latest RGB + SE(2) point goal",
        "privileged": False,
        "recommended_inference_hz": 10.0,
        "controller": "native SE(2) path; benchmark differential-drive tracker (not a released LiMo controller)",
    },
    "omnivla": {
        "label": "OmniVLA-8B original full checkpoint",
        "input": "latest RGB + metric pose goal",
        "privileged": False,
        "recommended_inference_hz": 3.0,
        "controller": "official OmniVLA waypoint-4 PD controller",
    },
    "mbra_logonav": {
        "label": "MBRA/LoGoNav (RA-L 2025)",
        "input": "last 6 RGB frames + metric point goal",
        "privileged": False,
        "recommended_inference_hz": 5.0,
        "controller": "official MBRA-PG PD controller",
    },
    "nomad": {
        "label": "NoMaD (official ICRA 2024 checkpoint)",
        "input": "last 4 RGB frames + current pre-planned subgoal RGB",
        "privileged": False,
        "recommended_inference_hz": 4.0,
        "controller": "official waypoint-index-2 PD controller",
    },
    "genie_samtp": {
        "label": "GeNIE SAM-TP (RA-L 2025)",
        "input": "latest RGB + metric point goal + calibrated camera mount",
        "privileged": False,
        "recommended_inference_hz": 5.0,
        "controller": "official SAM-TP BEV polynomial path planner",
    },
}

EXPECTED_FILE_SIZES = {
    "limo_tel": {
        "weights/limo/limo_trained_on_D_tel.safetensors": 126_291_012,
    },
    "limo_aug": {
        "weights/limo/limo_trained_on_D_aug.safetensors": 126_291_012,
    },
    "omnivla": {
        "weights/omnivla-original/config.json": 60_802,
        "weights/omnivla-original/model.safetensors.index.json": 94_764,
        "weights/omnivla-original/model-00001-of-00004.safetensors": 4_925_122_448,
        "weights/omnivla-original/model-00002-of-00004.safetensors": 4_947_392_496,
        "weights/omnivla-original/model-00003-of-00004.safetensors": 4_947_417_456,
        "weights/omnivla-original/model-00004-of-00004.safetensors": 262_668_432,
        "weights/omnivla-original/action_head--120000_checkpoint.pt": 201_513_842,
        "weights/omnivla-original/proprio_projector--120000_checkpoint.pt": 67_209_720,
    },
    "mbra_logonav": {
        "src/navigation-model-zoo/MBRA_PG_Official/inference.py": 5_751,
        "src/navigation-model-zoo/MBRA_PG_Official/mbra.onnx": 253_629_409,
        "src/navigation-model-zoo/MBRA_PG_Official/model_info.yaml": 492,
    },
    "nomad": {
        # Filled from the official release; an exact size check prevents a
        # partial Google Drive download from being accepted as a checkpoint.
        "weights/nomad/nomad.pth": 76_473_631,
    },
    "genie_samtp": {
        "weights/genie_samtp/checkpoint_2.pt": 409_330_592,
    },
}


def _decode_rgb(encoded: str) -> Image.Image:
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB").copy()


def _finite_path(path: Any) -> np.ndarray:
    result = np.asarray(path, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] < 2 or len(result) < 1:
        raise RuntimeError("model returned no 2-D trajectory")
    result = result[:, :2]
    if not np.isfinite(result).all():
        raise RuntimeError("model trajectory contains non-finite values")
    # All adapters return x=forward, y=left in metres.  The verifier contract
    # is x=right, z=forward.  Prepending the current origin is coordinate
    # bookkeeping, not a goal-directed correction or hidden planner.
    right_forward = np.column_stack((-result[:, 1], result[:, 0]))
    if np.linalg.norm(right_forward[0]) > 1e-6:
        right_forward = np.vstack((np.zeros((1, 2)), right_forward))
    return right_forward.astype(np.float32)


def _finite_control(
    *,
    forward_m_s: float,
    left_m_s: float = 0.0,
    angular_left_rad_s: float,
) -> dict[str, float]:
    """Validate and convert a model-native velocity to verifier axes."""

    values = (forward_m_s, left_m_s, angular_left_rad_s)
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("model controller returned a non-finite velocity")
    return {
        "forward_m_s": float(forward_m_s),
        "right_m_s": float(-left_m_s),
        "angular_left_rad_s": float(angular_left_rad_s),
    }


class Runtime:
    def __init__(self, assets_root: Path, device: str, robot_platform: RobotPlatform | None = None) -> None:
        self.assets_root = assets_root.resolve()
        self.robot_platform = robot_platform or RobotPlatform.from_environment()
        self.device = device
        self.dinov2_source = (
            self.assets_root
            / "torch_cache/hub/facebookresearch_dinov2_main"
        )
        if self.dinov2_source.is_dir():
            os.environ.setdefault("DINOV2_HUB_LOCAL", str(self.dinov2_source))
        self.lock = threading.Lock()
        self.active_family: str | None = None
        self.active_model: Any = None
        self.active_aux: Any = None
        self.model_cache: dict[str, tuple[Any, Any]] = {}
        self.last_load_seconds: float | None = None
        self.request_count = 0
        inference_lock_path = Path(
            os.environ.get(
                "VERIFIER_GPU_INFERENCE_LOCK",
                "/tmp/vggtbev_gpu5_sequential_inference.lock",
            )
        )
        inference_lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.inference_lock_path = inference_lock_path
        self.inference_lock_file = inference_lock_path.open("a+")

    @property
    def weights(self) -> Path:
        return self.assets_root / "weights"

    @property
    def sources(self) -> Path:
        return self.assets_root / "src"

    def availability(self) -> dict[str, dict[str, Any]]:
        def exact(method: str) -> bool:
            return all(
                (self.assets_root / relative).is_file()
                and (self.assets_root / relative).stat().st_size == expected
                for relative, expected in EXPECTED_FILE_SIZES[method].items()
            )

        checks = {
            "straight_line": True,
            "limo_tel": exact("limo_tel")
            and (self.dinov2_source / "hubconf.py").is_file(),
            "limo_aug": exact("limo_aug")
            and (self.dinov2_source / "hubconf.py").is_file(),
            "omnivla": exact("omnivla"),
            "mbra_logonav": exact("mbra_logonav"),
            "nomad": exact("nomad")
            and (self.sources / "visualnav-transformer/train/config/nomad.yaml").is_file()
            and (self.sources / "diffusion_policy/diffusion_policy").is_dir(),
            "genie_samtp": exact("genie_samtp")
            and (self.sources / "GENIE-SAMTP/sam2/sam_tp.py").is_file()
            and (
                self.sources
                / "GENIE-SAMTP/sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml"
            ).is_file()
            and (self.sources / "GENIE-SAMTP/genie_path_planner/planner.py").is_file(),
        }
        return {
            key: {**METHODS[key], "available": bool(checks[key])}
            for key in METHODS
        }

    def _unload(self) -> None:
        self.model_cache.clear()
        self.active_model = None
        self.active_aux = None
        self.active_family = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _ensure_limo(self, method: str) -> None:
        if method in self.model_cache:
            self.active_model, self.active_aux = self.model_cache[method]
            self.active_family = method
            return
        if self.active_family == method:
            return
        started = time.monotonic()
        import torch
        from safetensors.torch import load_file

        source = self.sources / "less-is-more"
        sys.path.insert(0, str(source))
        from limo.src.models.components.limo_net import LimoNet

        # LiMo hard-codes ``torch.hub.load("facebookresearch/dinov2", ...)``
        # even when pretrained=False.  On an isolated deployment that call
        # still reaches GitHub and can hold the cross-runtime GPU lock for
        # minutes.  Route only this known repository through the complete
        # local DINOv2 source bundled with the verifier.
        if not self.dinov2_source.is_dir():
            raise RuntimeError(
                f"local DINOv2 source is missing: {self.dinov2_source}"
            )
        original_hub_load = torch.hub.load

        def local_dinov2_load(repo_or_dir: str, model_name: str, *args: Any, **kwargs: Any) -> Any:
            if repo_or_dir == "facebookresearch/dinov2":
                repo_or_dir = str(self.dinov2_source)
                kwargs["source"] = "local"
                kwargs["trust_repo"] = True
            return original_hub_load(repo_or_dir, model_name, *args, **kwargs)

        torch.hub.load = local_dinov2_load
        try:
            model = LimoNet(pretrained=False)
        finally:
            torch.hub.load = original_hub_load
        suffix = "tel" if method == "limo_tel" else "aug"
        state = load_file(str(self.weights / f"limo/limo_trained_on_D_{suffix}.safetensors"))
        model.load_state_dict(state)
        model.eval().to(self.device)
        self.active_model = model
        self.active_aux = None
        self.active_family = method
        self.model_cache[method] = (self.active_model, self.active_aux)
        self.last_load_seconds = time.monotonic() - started

    def _ensure_omnivla(self) -> None:
        if "omnivla" in self.model_cache:
            self.active_model, self.active_aux = self.model_cache["omnivla"]
            self.active_family = "omnivla"
            return
        if self.active_family == "omnivla":
            return
        started = time.monotonic()
        source = self.sources / "OmniVLA"
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location(
            "vggtbev_omnivla_inference", source / "inference/run_omnivla.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load OmniVLA inference module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.pose_goal = True
        module.satellite = False
        module.image_goal = False
        module.lan_prompt = False
        cfg = module.InferenceConfig()
        cfg.vla_path = str(self.weights / "omnivla-original")
        cfg.resume_step = 120000
        cfg.num_images_in_input = 2
        model_tuple = module.define_model(cfg)
        self.active_model = module
        self.active_aux = model_tuple
        self.active_family = "omnivla"
        self.model_cache["omnivla"] = (self.active_model, self.active_aux)
        self.last_load_seconds = time.monotonic() - started

    def _ensure_mbra(self) -> None:
        if "mbra_logonav" in self.model_cache:
            self.active_model, self.active_aux = self.model_cache["mbra_logonav"]
            self.active_family = "mbra_logonav"
            return
        if self.active_family == "mbra_logonav":
            return
        started = time.monotonic()
        source = self.sources / "navigation-model-zoo/MBRA_PG_Official"
        spec = importlib.util.spec_from_file_location(
            "vggtbev_mbra_pg_inference", source / "inference.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the official MBRA-PG inference adapter")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.MBRAPGNavigator(
            onnx_path=str(source / "mbra.onnx"),
            device=self.device,
            max_v=self.robot_platform.max_v_m_s,
            max_w=max(0.65,self.robot_platform.max_w_rad_s),
        )
        self.active_model = model
        self.active_aux = module
        self.active_family = "mbra_logonav"
        self.model_cache["mbra_logonav"] = (self.active_model, self.active_aux)
        self.last_load_seconds = time.monotonic() - started

    def _ensure_nomad(self) -> None:
        if "nomad" in self.model_cache:
            self.active_model, self.active_aux = self.model_cache["nomad"]
            self.active_family = "nomad"
            return
        if self.active_family == "nomad":
            return
        started = time.monotonic()
        import torch
        import yaml
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        source = self.sources / "visualnav-transformer"
        train_source = source / "train"
        diffusion_source = self.sources / "diffusion_policy"
        sys.path.insert(0, str(train_source))
        # The official diffusion-policy tree is a PEP 420 namespace package
        # (no top-level __init__.py).  Its minimal setup.py therefore produces
        # an empty editable-package map with recent setuptools; expose the
        # checked-out official source directly, as NoMaD's own deployment does.
        sys.path.insert(0, str(diffusion_source))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
        from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

        config = yaml.safe_load((source / "train/config/nomad.yaml").read_text())
        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            ),
            dist_pred_net=DenseNetwork(embedding_dim=config["encoding_size"]),
        )
        checkpoint = torch.load(
            self.weights / "nomad/nomad.pth", map_location=self.device
        )
        incompatible = model.load_state_dict(checkpoint, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "NoMaD checkpoint does not exactly match the official config: "
                f"missing={incompatible.missing_keys[:5]} "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )
        model.eval().to(self.device)
        scheduler = DDPMScheduler(
            num_train_timesteps=int(config["num_diffusion_iters"]),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        data_config = yaml.safe_load(
            (source / "train/vint_train/data/data_config.yaml").read_text()
        )
        action_stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in data_config["action_stats"].items()
        }
        generator = torch.Generator(device=self.device)
        generator.manual_seed(20260907)
        aux = {
            "scheduler": scheduler,
            "config": config,
            "action_stats": action_stats,
            "generator": generator,
        }
        self.active_model = model
        self.active_aux = aux
        self.active_family = "nomad"
        self.model_cache["nomad"] = (model, aux)
        self.last_load_seconds = time.monotonic() - started

    def _ensure_genie_samtp(self) -> None:
        if "genie_samtp" in self.model_cache:
            self.active_model, self.active_aux = self.model_cache["genie_samtp"]
            self.active_family = "genie_samtp"
            return
        if self.active_family == "genie_samtp":
            return
        started = time.monotonic()
        source = self.sources / "GENIE-SAMTP"
        sys.path.insert(0, str(source))
        from genie_path_planner.planner import PlannerConfig, plan_on_bev
        from genie_path_planner.path_sampling import sample_paths_polynomial
        from genie_path_planner.projection import (
            logits_to_traversability,
            project_score_to_bev,
        )
        from sam2.sam_tp import SAM_TP

        model = SAM_TP(
            str(
                source
                / "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml"
            ),
            str(self.weights / "genie_samtp/checkpoint_2.pt"),
            score_thresh=0.0,
            multimask=False,
        )
        # These are the released Stretch planner defaults.  In particular,
        # unknown_cost=0.2 and the fixed polynomial bank are part of GeNIE's
        # native local-planning behavior, not benchmark-side repair.
        planner_config = PlannerConfig(
            grid_size=240,
            unknown_cost=0.2,
            smooth_kernel=3,
            num_goals=30,
            num_mid_points_per_goal=20,
            path_num_samples=100,
            footprint_px=18,
            threshold_cost=0.50,
            threshold_points_ratio=0.05,
            number_of_points_to_filter=60,
            alpha=1.0,
            best_k=12,
            use_clustering=True,
            max_clusters=4,
            cluster_angle_threshold_deg=40.0,
            random_seed=42,
            include_goal_in_path_bank=False,
            include_random_goals=True,
        )
        # GeNIE's released planner uses a deterministic, goal-independent
        # polynomial bank (include_goal_in_path_bank=False, seed=42).  Generate
        # it once when the runtime starts instead of rebuilding the identical
        # 600 paths for every online replan.
        candidate_path_bank = sample_paths_polynomial(
            robot=(planner_config.grid_size - 1, planner_config.grid_size // 2),
            num_goals=planner_config.num_goals,
            num_mid_points_per_goal=planner_config.num_mid_points_per_goal,
            num_samples=planner_config.path_num_samples,
            grid_size=planner_config.grid_size,
            goal=None,
            include_random_goals=planner_config.include_random_goals,
            random_seed=planner_config.random_seed,
        )
        if not candidate_path_bank:
            raise RuntimeError("GENIE fixed candidate path bank is empty")
        aux = {
            "logits_to_traversability": logits_to_traversability,
            "project_score_to_bev": project_score_to_bev,
            "plan_on_bev": plan_on_bev,
            "planner_config": planner_config,
            "candidate_path_bank": candidate_path_bank,
        }
        self.active_model = model
        self.active_aux = aux
        self.active_family = "genie_samtp"
        self.model_cache["genie_samtp"] = (model, aux)
        self.last_load_seconds = time.monotonic() - started

    @staticmethod
    def _goal_forward_left(target_right_forward: list[float]) -> np.ndarray:
        right, forward = map(float, target_right_forward)
        return np.asarray([forward, -right], dtype=np.float32)

    def _predict_limo(
        self, method: str, images: list[Image.Image], goal: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        self._ensure_limo(method)
        import torch

        yaw = math.atan2(float(goal[1]), float(goal[0]))
        image = self.active_model.transform(images[-1]).unsqueeze(0).to(self.device)
        goal_tensor = torch.tensor(
            [[float(goal[0]), float(goal[1]), yaw]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            output = self.active_model({"image_front": image, "goal": goal_tensor})
        trajectory = output[0].float().cpu().numpy()
        if trajectory.ndim != 2 or trajectory.shape[1] < 3 or len(trajectory) < 2:
            raise RuntimeError("LiMo returned an invalid SE(2) trajectory")

        # The public LiMo release supplies a native SE(2) path, not a deployed
        # low-level controller. Do not misrepresent the former hand-written
        # holonomic timed-lookahead helper as official or execute lateral motion
        # on the benchmark's differential-drive robot. The worker tracks the
        # untouched planar path using its declared downstream path controller.
        control = _finite_control(
            forward_m_s=0.0,
            angular_left_rad_s=0.0,
        )
        # Keep the released SE(2) heading. Dropping it made a centimetre-scale
        # backwards sample look like a request to turn the whole robot around.
        return trajectory[:, :3], control

    def _predict_omnivla(self, images: list[Image.Image], goal: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        self._ensure_omnivla()
        module = self.active_model
        vla, action_head, pose_projector, device_id, num_patches, tokenizer, processor = self.active_aux
        bearing = math.atan2(float(goal[1]), float(goal[0]))
        goal_pose = np.asarray(
            [goal[0] / 0.1, goal[1] / 0.1, math.cos(bearing), math.sin(bearing)],
            dtype=np.float32,
        )
        current = images[-1]
        inference = module.Inference(
            save_dir="/tmp",
            lan_inst_prompt="xxxx",
            goal_utm=(0.0, 0.0),
            goal_compass=bearing,
            goal_image_PIL=current,
            action_tokenizer=tokenizer,
            processor=processor,
        )
        batch = inference.data_transformer_omnivla(
            current, "xxxx", current, goal_pose,
            prompt_builder=module.PurePromptBuilder,
            action_tokenizer=tokenizer,
            processor=processor,
        )
        output, _modality = inference.run_forward_pass(
            vla=vla.eval(), action_head=action_head.eval(),
            noisy_action_projector=None, pose_projector=pose_projector.eval(),
            batch=batch, action_tokenizer=tokenizer, device_id=device_id,
            use_l1_regression=True, use_diffusion=False, use_film=False,
            num_patches=num_patches, mode="vali",
        )
        raw = output[0].float().cpu().numpy()
        if raw.ndim != 2 or raw.shape[1] < 4 or len(raw) < 1:
            raise RuntimeError("OmniVLA returned an invalid action chunk")
        waypoints = raw[:, :2] * 0.1

        # Official index-4 PD, metric spacing0.1 and nominal3Hz remain intact.
        # Platform velocity limits are explicitly configured for this robot.
        waypoint = raw[min(4, len(raw) - 1)].copy()
        waypoint[:2] *= 0.1
        limited_v, limited_w = omni_control(waypoint[:4].tolist(), self.robot_platform)
        control = _finite_control(
            forward_m_s=limited_v,
            angular_left_rad_s=limited_w,
        )
        return waypoints, control

    def _predict_mbra(
        self, images: list[Image.Image], goal: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run the released MBRA-PG wrapper without changing its policy."""

        self._ensure_mbra()
        frames = list(images[-6:])
        while len(frames) < 6:
            frames.insert(0, frames[0])
        obs = np.stack(
            [
                np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
                for image in frames
            ],
            axis=0,
        )[None]
        velocity, trajectory = self.active_model.inference_vw(obs, goal_xy=goal)
        velocity_np = velocity.detach().float().cpu().numpy()[0]
        linear,angular=curvature_limit(float(velocity_np[0]),float(velocity_np[1]),self.robot_platform)
        control = _finite_control(
            forward_m_s=linear,
            angular_left_rad_s=angular,
        )
        return np.asarray(trajectory[0], dtype=np.float32), control

    def _predict_nomad(
        self, images: list[Image.Image], goal_image: Image.Image
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run the official goal-conditioned NoMaD diffusion policy."""

        self._ensure_nomad()
        import torch

        frames = list(images[-4:])
        while len(frames) < 4:
            frames.insert(0, frames[0])

        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

        def transform(image: Image.Image) -> torch.Tensor:
            resized = image.resize((96, 96))
            array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            return torch.from_numpy((array - mean) / std)

        obs = torch.cat([transform(image) for image in frames], dim=0).unsqueeze(0)
        goal = transform(goal_image).unsqueeze(0)
        obs = obs.to(self.device)
        goal = goal.to(self.device)
        mask = torch.zeros(1, dtype=torch.long, device=self.device)
        config = self.active_aux["config"]
        scheduler = self.active_aux["scheduler"]
        with torch.inference_mode():
            condition = self.active_model(
                "vision_encoder",
                obs_img=obs,
                goal_img=goal,
                input_goal_mask=mask,
            )
            condition = condition.repeat(8, 1)
            action = torch.randn(
                (8, int(config["len_traj_pred"]), 2),
                device=self.device,
                generator=self.active_aux["generator"],
            )
            scheduler.set_timesteps(int(config["num_diffusion_iters"]))
            for timestep in scheduler.timesteps:
                noise = self.active_model(
                    "noise_pred_net",
                    sample=action,
                    timestep=timestep,
                    global_cond=condition,
                )
                action = scheduler.step(noise, timestep, action).prev_sample

        normalized_deltas = action.float().cpu().numpy()
        stats = self.active_aux["action_stats"]
        deltas = (normalized_deltas + 1.0) * 0.5
        deltas = deltas * (stats["max"] - stats["min"]) + stats["min"]
        trajectories = np.cumsum(deltas, axis=1)
        # Preserve declared action calibration separately from actuator caps.
        # Changing a robot speed ceiling does not rescale a metric prediction.
        trajectories *= nomad_metric_spacing(self.robot_platform)
        trajectory = trajectories[0]
        linear, angular = nomad_control(
            trajectory[min(2, len(trajectory) - 1)].tolist(), self.robot_platform
        )
        control = _finite_control(
            forward_m_s=linear,
            angular_left_rad_s=angular,
        )
        return trajectory.astype(np.float32), control

    def _predict_genie_samtp(
        self,
        images: list[Image.Image],
        target_right_forward: list[float],
        camera_calibration: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run the released RGB-only SAM-TP projection and path planner."""

        self._ensure_genie_samtp()
        camera_k = np.asarray(camera_calibration.get("intrinsics"), dtype=np.float64)
        camera_pose = np.asarray(
            camera_calibration.get("T_ground_camera"), dtype=np.float64
        )
        if camera_k.shape != (3, 3):
            raise ValueError("GENIE camera intrinsics must be a 3x3 matrix")
        if camera_pose.shape != (4, 4):
            raise ValueError("GENIE T_ground_camera must be a 4x4 matrix")
        if not np.isfinite(camera_k).all() or not np.isfinite(camera_pose).all():
            raise ValueError("GENIE camera calibration contains non-finite values")

        rgb = np.asarray(images[-1], dtype=np.uint8)
        result = self.active_model.run_sam2_inference(
            rgb,
            return_heatmap=False,
        )
        traversability = self.active_aux["logits_to_traversability"](
            np.asarray(result["logits"], dtype=np.float32), transform="sigmoid"
        )
        resolution_m = 0.03
        bev, observed, _projection_meta = self.active_aux["project_score_to_bev"](
            traversability,
            camera_k=camera_k,
            camera_pose=camera_pose,
            ground_z=float(camera_calibration.get("ground_z_m", 0.0)),
            bev_resolution_m_per_px=resolution_m,
            bev_forward_range_m=4.0,
            bev_side_range_m=2.0,
            max_ray_distance_m=6.0,
        )
        right, forward = map(float, target_right_forward)
        from dataclasses import replace
        native_config=self.active_aux["planner_config"]
        native_config=replace(native_config,footprint_px=footprint_pixels(
            SPEC.width_m,SPEC.length_m,SAFETY_MARGIN_M,bev.shape,
            resolution_m,native_config.grid_size))
        plan = self.active_aux["plan_on_bev"](
            bev_traversability=bev,
            observed_mask=observed,
            goal_x_m=right,
            goal_y_m=forward,
            bev_resolution_m=resolution_m,
            config=native_config,
            candidate_path_bank=self.active_aux["candidate_path_bank"],
        )
        native_xy = np.asarray(plan.final_path_xy_m, dtype=np.float32)
        if native_xy.ndim != 2 or native_xy.shape[1] != 2 or len(native_xy) < 2:
            status = plan.metadata.get("status", "no_native_path")
            raise RuntimeError(f"GENIE native planner failed: {status}")
        # GeNIE emits [right, forward].  The common adapter boundary accepts
        # [forward, left] and performs the single audited conversion below.
        forward_left = np.column_stack((native_xy[:, 1], -native_xy[:, 0]))
        control = _finite_control(
            forward_m_s=0.0,
            angular_left_rad_s=0.0,
        )
        return forward_left.astype(np.float32), control

    def reset(self) -> dict[str, Any]:
        """Clear episode-local controller state without reloading weights."""

        with self.lock:
            cached = self.model_cache.get("mbra_logonav")
            if cached is not None and hasattr(cached[0], "reset"):
                cached[0].reset()
            cached = self.model_cache.get("nomad")
            if cached is not None:
                cached[1]["generator"].manual_seed(20260907)
        return {
            "accepted": True,
            "controllers_reset": ["mbra_logonav", "nomad_diffusion_rng"],
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        method = str(payload.get("method", ""))
        available = self.availability()
        if method not in available:
            raise ValueError(f"unsupported method: {method}")
        if not available[method]["available"]:
            raise RuntimeError(f"assets for {method} are incomplete")
        target = payload.get("target_metric_m")
        if not isinstance(target, list) or len(target) != 2:
            raise ValueError("target_metric_m must be [right, forward]")
        if not all(math.isfinite(float(value)) for value in target):
            raise ValueError("target_metric_m must be finite")
        gt_history_raw = payload.get("gt_trajectory_history_metric_m", [])
        if not isinstance(gt_history_raw, list):
            raise ValueError("gt_trajectory_history_metric_m must be a list")
        if gt_history_raw:
            gt_history = np.asarray(gt_history_raw, dtype=np.float64)
            if (
                gt_history.ndim != 2
                or gt_history.shape[1] != 2
                or not np.isfinite(gt_history).all()
            ):
                raise ValueError(
                    "gt_trajectory_history_metric_m must contain finite "
                    "[right, forward] points"
                )
        else:
            gt_history = np.empty((0, 2), dtype=np.float64)
        encoded_images = payload.get("images_base64", [])
        if method != "straight_line" and not encoded_images:
            raise ValueError("at least one RGB frame is required")
        images = [_decode_rgb(str(value)) for value in encoded_images]
        goal_image = None
        if method == "nomad":
            encoded_goal_image = payload.get("goal_image_base64")
            if not isinstance(encoded_goal_image, str) or not encoded_goal_image:
                raise ValueError("NoMaD requires goal_image_base64")
            goal_image = _decode_rgb(encoded_goal_image)
        camera_calibration: dict[str, Any] | None = None
        if method == "genie_samtp":
            raw_calibration = payload.get("camera_calibration")
            if not isinstance(raw_calibration, dict):
                raise ValueError("GENIE requires camera_calibration")
            camera_calibration = raw_calibration
        goal = self._goal_forward_left(target)
        started = time.monotonic()
        with self.lock:
            fcntl.flock(self.inference_lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # This field is per request, not the stale duration of whichever
                # model happened to be loaded most recently.
                self.last_load_seconds = 0.0
                result = self._predict_locked(
                    method,
                    goal,
                    images,
                    gt_history,
                    goal_image,
                    target,
                    camera_calibration,
                )
            finally:
                fcntl.flock(self.inference_lock_file.fileno(), fcntl.LOCK_UN)
        forward_left, control = result
        path = _finite_path(forward_left)
        headings = None
        if method in {'limo_tel','limo_aug'}:
            headings=np.asarray(forward_left,dtype=float)[:,2].tolist()
            if len(path)==len(headings)+1: headings=[0.]+headings
            if len(path)!=len(headings) or not np.isfinite(headings).all():
                raise RuntimeError('LiMo planar path / native heading correspondence broken')
        return {
            "success": True,
            "method": method,
            "label": METHODS[method]["label"],
            "path_metric_m": path.tolist(),
            "path_headings_left_rad": headings,
            "waypoint_count": int(len(path)),
            "control_velocity": control,
            "inference_seconds": time.monotonic() - started,
            "model_load_seconds": self.last_load_seconds,
            "coordinate_frame": "current ego: x=right, z=forward, metres",
            "runtime_inputs": (
                ["RGB history", "current pre-planned subgoal RGB"]
                if method == "nomad"
                else ["latest RGB", "ego-local metric point goal", "camera calibration"]
                if method == "genie_samtp"
                else ["RGB", "GT ego-local point goal"]
                if method != "straight_line"
                else ["GT ego-local point goal"]
            ),
            "target_input": (
                "current pre-planned ImageGoal RGB; metric target used only by evaluator"
                if method == "nomad"
                else "GT-refreshed ego-local metric [right, forward]"
            ),
            "gt_trajectory_history_points_received": int(len(gt_history)),
            "gt_trajectory_history_used_by_native_model": False,
            "forbidden_inputs": [
                "GT BEV",
                "navmesh",
                "GT obstacles",
                "GT depth",
                "other robots' state",
            ],
            "recommended_inference_hz": METHODS[method]["recommended_inference_hz"],
            "controller": METHODS[method]["controller"],
            "robot_platform": {
                "max_v_m_s": self.robot_platform.max_v_m_s,
                "max_w_rad_s": self.robot_platform.max_w_rad_s,
                "holonomic": False,
                "speed_semantics": "maximum/cruise envelope; native controller may slow or stop",
                "nomad_spacing_m": self.robot_platform.nomad_spacing_m,
                "body_dimensions_m": [SPEC.width_m,SPEC.length_m,SPEC.height_m],
                "planner_safety_margin_m": SAFETY_MARGIN_M,
                "actuator_adapter": "curvature_preserving_saturation_v5",
            },
            "native_controller_dt_s": {"omnivla": 1.0 / 3.0, "mbra_logonav": 2.5, "nomad": 0.25}.get(method),
            "metric_waypoint_spacing_m": {"omnivla": 0.1, "mbra_logonav": 0.8, "nomad": nomad_metric_spacing(self.robot_platform)}.get(method),
            "native_low_level_controller_released": method in {"omnivla", "mbra_logonav", "nomad"},
            "control_velocity_is_placeholder": method in {"limo_tel", "limo_aug", "genie_samtp"},
            "execution_preference": (
                "control_velocity"
                if method in {"omnivla", "mbra_logonav", "nomad"}
                else "native_path_1m_receding_horizon"
                if method == "genie_samtp"
                else "native_path"
            ),
            "global_inference_serialized": True,
        }

    def _predict_locked(
        self,
        method: str,
        goal: np.ndarray,
        images: list[Image.Image],
        gt_history: np.ndarray,
        goal_image: Image.Image | None,
        target_right_forward: list[float],
        camera_calibration: dict[str, Any] | None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run one method while the cross-process GPU lock is held.

        ``gt_history`` is deliberately present in the common planner contract.
        The published model adapters do not define a trajectory-history input,
        so it is audited but not injected into their learned architectures.
        """

        del gt_history

        control: dict[str, float]
        if method == "straight_line":
            distance = float(np.linalg.norm(goal))
            clipped_goal = goal.copy()
            if distance > 5.0:
                clipped_goal *= 5.0 / distance
            count = max(2, int(math.ceil(min(distance, 5.0) / 0.10)) + 1)
            forward_left = np.linspace(np.zeros(2), clipped_goal, count, dtype=np.float32)
            bearing = math.atan2(float(goal[1]), float(goal[0]))
            if abs(bearing) > math.radians(5.0):
                control = _finite_control(
                    forward_m_s=0.0,
                    angular_left_rad_s=float(np.clip(bearing * 2.0, -0.3, 0.3)),
                )
            else:
                control = _finite_control(
                    forward_m_s=min(0.6, distance * 1.5),
                    angular_left_rad_s=0.0,
                )
        elif method in {"limo_tel", "limo_aug"}:
            forward_left, control = self._predict_limo(method, images, goal)
        elif method == "omnivla":
            forward_left, control = self._predict_omnivla(images, goal)
        elif method == "mbra_logonav":
            forward_left, control = self._predict_mbra(images, goal)
        elif method == "nomad":
            if goal_image is None:  # validated by predict()
                raise AssertionError("missing NoMaD ImageGoal")
            forward_left, control = self._predict_nomad(images, goal_image)
        elif method == "genie_samtp":
            if camera_calibration is None:  # validated by predict()
                raise AssertionError("missing GENIE camera calibration")
            forward_left, control = self._predict_genie_samtp(
                images, target_right_forward, camera_calibration
            )
        else:  # pragma: no cover
            raise AssertionError(method)
        self.request_count += 1
        return forward_left, control


class Server(ThreadingHTTPServer):
    allow_reuse_address = True


def make_handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json({
                    "ready": True,
                    "device": runtime.device,
                    "active_family": runtime.active_family,
                    "single_gpu_lazy_loading": True,
                    "methods": runtime.availability(),
                    "request_count": runtime.request_count,
                    "global_inference_serialized": True,
                    "inference_lock_path": str(runtime.inference_lock_path),
                    "robot_platform": {
                        "max_v_m_s": runtime.robot_platform.max_v_m_s,
                        "max_w_rad_s": runtime.robot_platform.max_w_rad_s,
                        "nomad_spacing_m":runtime.robot_platform.nomad_spacing_m,
                        "body_dimensions_m": [SPEC.width_m,SPEC.length_m,SPEC.height_m],
                        "planner_safety_margin_m": SAFETY_MARGIN_M,
                    },
                    "runtime_contract": "realtime-native-platform-v6",
                })
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                if self.path not in {"/predict", "/reset"}:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 64 * 1024 * 1024:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/reset":
                    self._json(runtime.reset())
                else:
                    self._json(runtime.predict(payload))
            except ValueError as error:
                self._json({"success": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._json(
                    {"success": False, "error": str(error), "type": type(error).__name__},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format_string: str, *args: Any) -> None:
            if self.path != "/health":
                super().log_message(format_string, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--require-all-assets", action="store_true")
    parser.add_argument("--robot-max-v", type=float, default=float(os.environ.get("NAV_ROBOT_MAX_V_M_S", "1.0")))
    parser.add_argument("--robot-max-w", type=float, default=float(os.environ.get("NAV_ROBOT_MAX_W_RAD_S", str(math.pi / 2))))
    parser.add_argument("--nomad-metric-spacing",type=float,default=0.05)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runtime = Runtime(args.assets_root, args.device, RobotPlatform(args.robot_max_v, args.robot_max_w,args.nomad_metric_spacing))
    unavailable = [
        key for key, value in runtime.availability().items()
        if not value["available"]
    ]
    if args.require_all_assets and unavailable:
        raise RuntimeError(
            "required full baseline assets are unavailable or have unexpected "
            f"sizes: {', '.join(unavailable)}"
        )
    server = Server((args.host, args.port), make_handler(runtime))
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", "methods": runtime.availability()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
