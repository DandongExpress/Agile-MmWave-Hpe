from __future__ import annotations

from torch import Tensor, nn


class PoseRegressionNetwork(nn.Module):
    """Equation (11): the only learnable component of the proposed pipeline."""

    def __init__(self, input_features: int, hidden_dims: list[int], num_joints: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[1], num_joints * 3),
        )
        self.num_joints = num_joints

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).view(features.shape[0], self.num_joints, 3)