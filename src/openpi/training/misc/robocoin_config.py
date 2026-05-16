"""RoboCoin policy configs."""

import dataclasses
import pathlib
import tyro
from collections.abc import Sequence
from typing import TypeAlias
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.transforms as _transforms


ModelType: TypeAlias = _model.ModelType


def get_robocin_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import TrainConfig
    import openpi.training.weight_loaders as weight_loaders
    from ..config import DataConfigFactory, DataConfig, ModelTransformFactory
    from ...policies import robocoin_policy

    @dataclasses.dataclass(frozen=True)
    class LeRobotRoboCoinDataConfig(DataConfigFactory):

        use_delta_joint_actions: bool = False

        repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
            default=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation.images.cam_high": "observation.images.cam_high",
                            "observation.images.cam_left_wrist": "observation.images.cam_left_wrist",
                            "observation.images.cam_right_wrist": "observation.images.cam_right_wrist",
                            "observation.state": "observation.state",
                            "action": "action", # action -> actions
                            "prompt": "prompt",
                        }
                    )
                ]
            )
        )

        action_sequence_keys: Sequence[str] = ("action",)

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
            data_transforms = _transforms.Group(
                inputs=[robocoin_policy.RoboCoinInputs()],
                outputs=[robocoin_policy.RoboCoinOutputs()],
            )
            if self.use_delta_joint_actions:
                delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=self.repack_transforms,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
            )

    return [
        TrainConfig(
            name="pi0_robocoin",
            model=pi0_config.Pi0Config(),
            data=LeRobotoboCoinDataConfig(
                repo_id="robocoin/eval_v1",
                use_delta_joint_actions=False,
                # base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi0_robocoin_lora",
            model=pi0_config.Pi0Config(),
            data=LeRobotoboCoinDataConfig(
                repo_id="robocoin/eval_v1_anno",
                use_delta_joint_actions=False,
                # base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
        ),
        TrainConfig(
            name="pi0_robocoin_delta",
            model=pi0_config.Pi0Config(),
            data=LeRobotoboCoinDataConfig(
                repo_id="robocoin/eval_v1",
                use_delta_joint_actions=True,
                # base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi0_robocoin_lora_delta",
            model=pi0_config.Pi0Config(),
            data=LeRobotoboCoinDataConfig(
                repo_id="robocoin/eval_v1",
                use_delta_joint_actions=True,
                # base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
        ),
        TrainConfig(
            name="pi0_robocoin_lora_debug",
            model=pi0_config.Pi0Config(),
            data=LeRobotoboCoinDataConfig(
                repo_id="robocoin/eval_v1_anno",
                use_delta_joint_actions=False,
                # base_config=DataConfig(prompt_from_task=True),
            ),
            batch_size=1,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
        ),
    ]