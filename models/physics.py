"""Deterministic physics-guided front-end modules from the paper."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class SpatialStructurePreservation(nn.Module):
    """Equation (1)-(2): anthropometric range-angle masking."""

    def __init__(
        self,
        range_bins: int,
        angle_bins: int,
        range_resolution_m: float,
        angle_min_deg: float,
        angle_max_deg: float,
        range_bounds_m: Sequence[float],
        angle_bounds_deg: Sequence[float],
    ) -> None:
        super().__init__()
        ranges = torch.arange(range_bins, dtype=torch.float32) * range_resolution_m
        angles = torch.linspace(angle_min_deg, angle_max_deg, angle_bins)
        mask = (
            (ranges[:, None] >= range_bounds_m[0])
            & (ranges[:, None] <= range_bounds_m[1])
            & (angles[None, :] >= angle_bounds_deg[0])
            & (angles[None, :] <= angle_bounds_deg[1])
        )
        self.register_buffer("spatial_mask", mask.to(torch.float32))

    def forward(self, radar_cube: Tensor) -> Tensor:
        return radar_cube * self.spatial_mask[None, :, :, None]


class MotionContinuityPreservation(nn.Module):
    """Equation (3)-(7): dominant Doppler velocity and local consistency mask."""

    def __init__(
        self,
        doppler_bins: int,
        max_velocity_mps: float,
        velocity_bounds_mps: Sequence[float],
        velocity_std_bounds_mps: Sequence[float],
        local_window_radius: int,
    ) -> None:
        super().__init__()
        self.doppler_bins = doppler_bins
        self.max_velocity_mps = max_velocity_mps
        self.velocity_min, self.velocity_max = velocity_bounds_mps
        self.std_min, self.std_max = velocity_std_bounds_mps
        self.local_window_radius = local_window_radius

    def forward(self, spatial_cube: Tensor) -> Tuple[Tensor, Tensor]:
        magnitudes = spatial_cube.abs()
        dominant_bin = magnitudes.argmax(dim=-1).to(spatial_cube.dtype)
        velocity = self.max_velocity_mps * (
            (dominant_bin - self.doppler_bins / 2) / (self.doppler_bins / 2)
        )

        radius = self.local_window_radius
        kernel = 2 * radius + 1
        velocity_image = velocity.unsqueeze(1)
        local_mean = functional.avg_pool2d(
            velocity_image, kernel_size=kernel, stride=1, padding=radius, count_include_pad=False
        ).squeeze(1)
        local_mean_square = functional.avg_pool2d(
            velocity_image.square(), kernel_size=kernel, stride=1, padding=radius, count_include_pad=False
        ).squeeze(1)
        local_std = (local_mean_square - local_mean.square()).clamp_min(0).sqrt()
        motion_mask = (
            (velocity.abs() >= self.velocity_min)
            & (velocity.abs() <= self.velocity_max)
            & (local_std >= self.std_min)
            & (local_std <= self.std_max)
        ).to(spatial_cube.dtype)
        motion_cube = spatial_cube * motion_mask.unsqueeze(-1)
        masked_velocity = velocity * motion_mask
        maximum_velocity = masked_velocity.abs().reshape(masked_velocity.shape[0], -1).max(dim=1).values
        descriptors = torch.stack(
            (masked_velocity.mean(dim=(1, 2)), masked_velocity.std(dim=(1, 2), unbiased=False), maximum_velocity),
            dim=1,
        )
        return motion_cube, descriptors


class HierarchicalMultiScaleFusion(nn.Module):
    """Equation (8)-(9): body-aligned coarse, medium, and fine 3D representations."""

    def __init__(self, coarse_kernel: Sequence[int], medium_kernel: Sequence[int]) -> None:
        super().__init__()
        self.coarse_kernel = tuple(coarse_kernel)
        self.medium_kernel = tuple(medium_kernel)

    @staticmethod
    def _pool_and_restore(volume: Tensor, kernel: Tuple[int, int, int]) -> Tensor:
        pooled = functional.avg_pool3d(volume, kernel_size=kernel, stride=kernel, ceil_mode=False)
        return functional.interpolate(pooled, size=volume.shape[-3:], mode="trilinear", align_corners=False)

    def forward(self, motion_cube: Tensor) -> Tensor:
        # PyTorch 3D operations use (Doppler, Range, Angle); config kernels are (Range, Angle, Doppler).
        volume = motion_cube.permute(0, 3, 1, 2).unsqueeze(1)
        coarse = self._pool_and_restore(volume, (self.coarse_kernel[2], self.coarse_kernel[0], self.coarse_kernel[1]))
        medium = self._pool_and_restore(volume, (self.medium_kernel[2], self.medium_kernel[0], self.medium_kernel[1]))
        return torch.cat((coarse, medium, volume), dim=1)