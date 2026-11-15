"""A no-attention residual control using the same own-variable summaries."""
import torch
from torch import nn

from .hourly_architecture_trials import TrialConfig, ordered_residual_quantiles


class SummaryResidualMLP(nn.Module):
    def __init__(self, variables, horizons=(1, 4, 8, 24), hidden=32, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.variables = tuple(variables)
        self.horizons = tuple(horizons)
        self.hidden, self.seed = hidden, seed
        self.weight = nn.Parameter(torch.randn(len(variables), 10, hidden) * .05)
        self.bias = nn.Parameter(torch.zeros(len(variables), hidden))
        self.output_weight = nn.Parameter(torch.zeros(len(variables), hidden, 3))
        self.output_bias = nn.Parameter(torch.zeros(len(variables), 3))

    def forward(self, model_args, own, baseline_z):
        horizon = own.new_tensor(self.horizons)[None, :, None, None] / 24.
        x = torch.cat((own[:, None].expand(-1, len(self.horizons), -1, -1),
                       horizon.expand(len(own), -1, len(self.variables), -1)), -1)
        hidden = torch.nn.functional.gelu(torch.einsum("bhvi,vik->bhvk", x, self.weight) + self.bias)
        correction = torch.einsum("bhvk,vkq->bhvq", hidden, self.output_weight) + self.output_bias
        return ordered_residual_quantiles(baseline_z, correction, TrialConfig())


def load_summary_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = SummaryResidualMLP(state["variables"], state["horizons"], state["hidden"], state["seed"])
    model.load_state_dict(state["state_dict"])
    return model.eval(), state
