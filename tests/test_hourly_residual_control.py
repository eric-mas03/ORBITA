import torch
from astronaut_twin.hourly_residual_control import SummaryResidualMLP, load_summary_checkpoint


def test_control_initial_identity_and_input_gradients():
    model = SummaryResidualMLP(["HR", "Resp"])
    own = torch.randn(3, 2, 9)
    base = torch.randn(3, 4, 2, 3).sort(-1).values
    output = model(None, own, base)
    assert torch.equal(output, base)
    output.sum().backward()
    assert model.output_bias.grad.abs().sum() > 0
    assert model.output_weight.grad.abs().sum() > 0


def test_control_frozen_reload(tmp_path):
    model = SummaryResidualMLP(["HR", "Resp"])
    own = torch.randn(3, 2, 9)
    base = torch.randn(3, 4, 2, 3).sort(-1).values
    with torch.no_grad():
        model.output_weight.normal_(std=.1)
    path = tmp_path / "control.pt"
    torch.save(dict(state_dict=model.state_dict(), variables=model.variables, horizons=model.horizons,
                    hidden=model.hidden, seed=model.seed), path)
    restored, state = load_summary_checkpoint(path)
    assert torch.equal(model(None, own, base), restored(None, own, base))
    assert (restored(None, own, base).diff(dim=-1) >= 0).all()
