from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor, nn

from .physics import HierarchicalMultiScaleFusion, MotionContinuityPreservation, SpatialStructurePreservation
from .pose_regressor import PoseRegressionNetwork


class AgileMmWaveHPE(nn.Module):
    """SSP -> MCP -> HMSF -> global pooling + PRN, as specified in the paper."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.ssp = SpatialStructurePreservation(
            config["range_bins"], config["angle_bins"], config["range_resolution_m"],
            config["angle_min_deg"], config["angle_max_deg"], config["spatial_bounds_m"], config["spatial_bounds_deg"],
        )
        self.mcp = MotionContinuityPreservation(
            config["doppler_bins"], config["max_velocity_mps"], config["velocity_bounds_mps"],
            config["velocity_std_bounds_mps"], config["local_window_radius"],
        )
        self.hmsf = HierarchicalMultiScaleFusion(config["coarse_kernel"], config["medium_kernel"])
        # Three pooled HMSF channels plus the three global MCP descriptors in Equation (10).
        self.prn = PoseRegressionNetwork(6, config["hidden_dims"], config["num_joints"])

    def front_end(self, radar_cube: Tensor) -> Tensor:
        spatial_cube = self.ssp(radar_cube)
        motion_cube, motion_descriptors = self.mcp(spatial_cube)
        multiscale = self.hmsf(motion_cube)
        pooled_multiscale = multiscale.mean(dim=(2, 3, 4))
        return torch.cat((pooled_multiscale, motion_descriptors), dim=1)

    def forward(self, radar_cube: Tensor) -> Tensor:
        if radar_cube.ndim != 4:
            raise ValueError("Expected radar_cube with shape [batch, range, angle, doppler].")
        return self.prn(self.front_end(radar_cube))