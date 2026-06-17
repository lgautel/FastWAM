from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Literal, Set

import torch
import numpy as np
from copy import deepcopy
from torchvision.io import write_video
from ..utils.normalizer import LinearNormalizer, NormMode
from fastwam.utils.pytorch_utils import dict_apply
from fastwam.utils.logging_config import get_logger
from .base_processor import BaseProcessor

logger = get_logger(__name__)

_DEFAULT_TRANSFORM_TEST_DIR = (
    "/home/Luogang/SRC/RL/RLinf/b/test/trnsf_tst"
)
EpisodeFrameLoader = Callable[[int, str, int], torch.Tensor]
_episode_frame_loader: Optional[EpisodeFrameLoader] = None
_dumped_episodes: Set[int] = set()
_dump_episode_count = 0


def register_episode_frame_loader(loader: Optional[EpisodeFrameLoader]) -> None:
    """Register a callback that loads full-episode camera frames for MP4 dump."""
    global _episode_frame_loader
    _episode_frame_loader = loader


def reset_episode_transform_dump_state() -> None:
    """Reset dump counters (for smoke tests / repeated runs in one process)."""
    global _dumped_episodes, _dump_episode_count
    _dumped_episodes = set()
    _dump_episode_count = 0


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _lerobot_image_key(meta: Dict[str, Any]) -> str:
    key = meta["key"]
    return meta.get("lerobot_key") or (
        f"observation.images.{key}" if key != "default" else "observation.images"
    )


def _tensor_to_video_u8(image: torch.Tensor) -> torch.Tensor:
    """Convert ``[T, C, H, W]`` float or uint8 tensor to ``[T, H, W, C]`` uint8."""
    image = image.detach().cpu()
    if image.dtype == torch.uint8:
        return image.permute(0, 2, 3, 1).contiguous()
    return image.clamp(0.0, 1.0).permute(0, 2, 3, 1).mul(255.0).to(torch.uint8)


def dump_episode_transform_mp4(
    processor: "FastWAMProcessor",
    episode_index: int,
    dataset_index: int = 0,
    *,
    force: bool = False,
) -> None:
    """Dump one full episode (all cameras) with train/val transforms applied.

    Requires ``register_episode_frame_loader`` (wired from ``BaseLerobotDataset``).
    """
    global _dump_episode_count

    if _episode_frame_loader is None:
        raise RuntimeError(
            "Episode frame loader not registered; call BaseLerobotDataset.set_processor first"
        )
    if not force and episode_index in _dumped_episodes:
        return

    out_dir = Path(
        os.environ.get("RLINF_TRANSFORM_TEST_DIR", _DEFAULT_TRANSFORM_TEST_DIR)
    )
    fps = int(round(float(os.environ.get("FASTWAM_DUMP_TRANSFORM_FPS", "14"))))
    transforms = processor.train_transforms if processor.is_train else processor.val_transforms

    out_dir.mkdir(parents=True, exist_ok=True)
    for meta in processor.shape_meta["images"]:
        key = meta["key"]
        lerobot_key = _lerobot_image_key(meta)
        image = _episode_frame_loader(episode_index, lerobot_key, dataset_index)
        if image.ndim != 4:
            raise ValueError(
                f"Episode loader must return [T, C, H, W], got {tuple(image.shape)}"
            )

        current_transforms = transforms[key] if isinstance(transforms, dict) else transforms
        for trans in current_transforms:
            image = trans(image)

        out_path = out_dir / f"episode_{episode_index:06d}_{key}.mp4"
        video_u8 = _tensor_to_video_u8(image)
        write_video(str(out_path), video_u8, fps=fps)
        logger.info("Dumped episode transform MP4: %s (%d frames)", out_path, video_u8.shape[0])

    _dumped_episodes.add(episode_index)
    _dump_episode_count += 1


def _maybe_dump_episode_transform_mp4(processor: "FastWAMProcessor", data: Dict[str, Any]) -> None:
    """Dump full-episode augmented videos when enabled during ``preprocess``.

    See ``RLinf/b/test/trnsf_tst/README.md``.
    """
    if os.environ.get("FASTWAM_DUMP_TRANSFORM_MP4") != "1":
        return
    if _episode_frame_loader is None:
        return

    episode_index = _to_int(data.get("episode_index"))
    if episode_index is None:
        logger.warning("FASTWAM_DUMP_TRANSFORM_MP4 enabled but episode_index missing in sample")
        return
    if episode_index in _dumped_episodes:
        return

    episode_filter = os.environ.get("FASTWAM_DUMP_EPISODE_INDICES", "").strip()
    if episode_filter:
        allowed = {int(x.strip()) for x in episode_filter.split(",") if x.strip()}
        if episode_index not in allowed:
            return

    max_episodes = int(os.environ.get("FASTWAM_DUMP_TRANSFORM_MAX", "10"))
    if _dump_episode_count >= max_episodes:
        return

    dataset_index = _to_int(data.get("dataset_index")) or 0
    try:
        dump_episode_transform_mp4(processor, episode_index, dataset_index)
    except Exception as exc:
        logger.warning(
            "Failed to dump episode transform MP4 for episode %s: %s", episode_index, exc
        )


class FastWAMProcessor(BaseProcessor):
    def __init__(
        self,
        # keys
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int, 
        action_output_dim: int,
        proprio_output_dim: int,

        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        norm_exception_mode: Dict[str, Dict[str, NormMode]], 

        action_state_merger, 

        # image transform
        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None, 

        action_state_transforms: Optional[List[Any]],
        proprio_augmentations: Optional[List[Any]] = None,
        
        # instruction transform
        drop_high_level_prob: float = 1.0,
        use_zh_instruction: bool = False,

        tokenizer: Optional[Any] = None,
        delta_action_dim_mask: Optional[Dict[str, List[bool]]] = None,
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        # image
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self._is_train = None

        self.action_state_transforms = action_state_transforms
        self.proprio_augmentations = proprio_augmentations
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)

        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self._normalizer = None

        self.tokenizer = tokenizer
        if delta_action_dim_mask is None:
            self.delta_action_dim_mask = None
        else:
            action_meta = self.shape_meta["action"]
            expected_keys = [m["key"] for m in action_meta]
            provided_keys = list(delta_action_dim_mask.keys())
            if set(provided_keys) != set(expected_keys):
                raise ValueError(
                    f"`delta_action_dim_mask` keys mismatch. Expected {expected_keys}, got {provided_keys}."
                )

            self.delta_action_dim_mask = {}
            for meta in action_meta:
                key = meta["key"]
                expected_dim = meta["shape"]
                mask = delta_action_dim_mask[key]
                if len(mask) != expected_dim:
                    raise ValueError(
                        f"`delta_action_dim_mask[{key}]` length must be {expected_dim}, got {len(mask)}."
                    )
                self.delta_action_dim_mask[key] = torch.as_tensor(mask, dtype=torch.bool)

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError("is_train has not been set. Please call train() and eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError("normalizer has not been set. Please call set_normalizer_from_stats() first.")
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
        )

    def augment_instruction(self, data: Dict[str, str] | List[str]) -> List[str]:
        """
        Args:
            data: Dict[str, str] | List[str], lerobot sample in raw mcap

        Returns:
            List[str], processed instructions
        """
        # if single instruction, convert to list
        if "coarse_task" in data:
            high_level_instruction = data["coarse_task"]
        else:
            high_level_instruction = ""
        if "task" not in data:
            return f"[high] {high_level_instruction}"

        low_level_instruction = data["task"]
        # Galaxea lerobot use @ to split Chinese and English instruction
        if "@" in low_level_instruction:
            zh, eng = low_level_instruction.split("@")
            low_level_instruction = zh if self.use_zh_instruction else eng

        if np.random.rand() < self.drop_high_level_prob:
            instruction = f"{low_level_instruction}"
        else: 
            instruction = f"[High]: {high_level_instruction}, [Low]: {low_level_instruction}"
        
        return instruction

    def action_state_transform(self, batch):
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
                    
        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
        
        if self.action_state_transforms is not None: 
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)
        
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."
        
        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual transformed shape {actual_shape} mismatch with meta raw shape {meta_shape}."
        
        return batch

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for the policy model.
        
        Args:
            Data: Dict[str, Any], lerobot sample in raw mcap obtained from dataset __getitem__:
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "idx": int, sample index
                
        Returns:
            Sample: Dict[str, Any], which can collated:
                - "input_ids": torch.Tensor -> [max_image_text_tokens,]
                - "attention_mask": torch.Tensor -> [max_image_text_tokens,]
                - "pixel_values": torch.Tensor -> [num_input_cameras, C, H, W]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "proprio": torch.Tensor -> [num_obs_steps, proprio_dim]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action: Optional, deepcopy of input action for open loop eval, which is left untouched
                - "idx": int, sample index
        """
        sample = {}
        # 1. instruction
        sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]

        # 2. image
        processed_images = []
        for meta in self.shape_meta["images"]:
            key, shape = meta["key"], meta["shape"]
            image = data["images"][key]  # [num_obs_steps, C, H, W]
            assert image.ndim == 4, f"Expected 4 dimensions (num_obs_steps, C, H, W), got shape {image.shape}"
            
            # Apply transforms efficiently on the merged batch
            transforms = self.train_transforms if self.is_train else self.val_transforms
            current_transforms = transforms[key] if isinstance(transforms, dict) else transforms
            for trans in current_transforms:
                image = trans(image)

            meta_shape = [self.num_obs_steps] + shape
            assert list(image.shape) == meta_shape, \
                f"Expected shape {meta_shape}, got {image.shape} after transforms for key {key}"

            processed_images.append(image)
        # _maybe_dump_episode_transform_mp4(self, data) #@#按episode保存训练时被数据增强的图片
        pixel_values = torch.stack(processed_images, dim=0) # [num_input_cameras, T, C, H, W]
        
        if self.num_output_cameras > pixel_values.shape[0]:
            out = torch.zeros((self.num_output_cameras,) + pixel_values.shape[1:], device=pixel_values.device, dtype=pixel_values.dtype)
            out[0: pixel_values.shape[0]] = pixel_values
            sample["pixel_values"] = out
        elif self.num_output_cameras < pixel_values.shape[0]:
            logger.warning(f"num_output_cameras {self.num_output_cameras} is less than the number of cameras in data {pixel_values.shape[0]}, "
                           f"truncating the input to the first {self.num_output_cameras} cameras.")
            sample["pixel_values"] = pixel_values[:self.num_output_cameras]
        else:
            sample["pixel_values"] = pixel_values

        # Copy action before transform for open-loop evaluation, 
        # disabled for training dataset as it may cause collating key problem.
        if not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        # 3. action & state
        if "action" in data and self.delta_action_dim_mask is not None:
            action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
            if bool(action_is_pad.any().item()):
                for key, dim_mask in self.delta_action_dim_mask.items():
                    cur_action = data["action"][key]
                    cur_action_is_pad = action_is_pad.to(device=cur_action.device)
                    cur_dim_mask = dim_mask.to(device=cur_action.device)
                    pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
                    cur_action[pad_delta_mask] = 0.0
        data = self.action_state_transform(data)
        if self.is_train and self.proprio_augmentations is not None:
            for aug in self.proprio_augmentations:
                data = aug(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"] # [action_horizon, action_dim]
            sample["action_is_pad"] = data["action_is_pad"] # [action_horizon,]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"] # [action_dim,]
            assert sample["action"].shape[-1] == self.action_output_dim
            # sample["action"][sample["action_is_pad"], :-1] = 0.0 # NOTE: we assume use delta_eef_pose + gripper， so pad action is 0

        
        # TODO: rename all "state" into "proprio"
        sample["proprio"] = data["state"] # [num_obs_steps, proprio_dim]
        sample["proprio_is_pad"] = data["state_is_pad"] # [num_obs_steps,]
        sample["proprio_dim_is_pad"] = data["state_dim_is_pad"] # [proprio_dim,]
        assert sample["proprio"].shape[-1] == self.proprio_output_dim

        sample["idx"] = data["idx"]

        # sample = self.tokenizer(sample)
        
        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the data for the policy model.
        
        Args:
            data: Dict[str, Any], lerobot sample in raw mcap

        Returns:
            data: Dict[str, Any], processed data including unnormalized action
        """
        assert "action" in data, "Action is required in postprocess"
        data["state"] = data.pop("proprio")
        data = self.action_state_merger.backward(data)
        data = self.normalizer.backward(data)
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)

        start_obs_step = self.num_obs_steps - 1
        data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
        return data
