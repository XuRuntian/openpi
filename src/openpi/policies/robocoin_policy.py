import dataclasses

import einops
import numpy as np

from openpi import transforms


def make_robocoin_example() -> dict:
    return {
        "observation.state": np.random.rand(16),
        "observation.images.cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCoinInputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation.images.cam_high"])
        left_wrist_image = _parse_image(data["observation.images.cam_left_wrist"])
        right_wrist_image = _parse_image(data["observation.images.cam_right_wrist"])

        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        inputs["state"] = transforms.pad_to_dim(data["observation.state"], 32)
        inputs["prompt"] = data["prompt"]

        if "action" in data:
            inputs["action"] = transforms.pad_to_dim(data["action"], 32)

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCoinOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"action": np.asarray(data["action"][:, :16])}