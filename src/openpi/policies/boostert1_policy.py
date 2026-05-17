import dataclasses

import einops
import numpy as np

from openpi import transforms

# 拿到的是干净的7joints +1gripper的16维动作空间。
def make_boostert1_example() -> dict:
    return {
        "observation.state": np.random.rand(16),
        "observation.images.image_top": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
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
class Boostert1Inputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation.images.image_top"])
        
        inputs = {
            "image": {
                "base_0_rgb": base_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
            },
        }

        inputs["state"] = transforms.pad_to_dim(data["observation.state"], 32)        
        inputs["prompt"] = data["prompt"]

        if "action" in data:
            inputs["action"] = transforms.pad_to_dim(data["action"], 32)

        return inputs


@dataclasses.dataclass(frozen=True)
class Boostert1Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"action": np.asarray(data["action"][:, :16])}