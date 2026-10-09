"""Tests for spodnn.dfnn: the network and its Lightning wrapper."""

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spodnn.dfnn import LitMLPModel

N_INPUTS, N_OUTPUTS, N_SAMPLES = 2, 18, 100


def make_trainer(tmp_path):
    # CPU for reproducible numbers; no logs, progress bars or summaries in the test output.
    return pl.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )


@pytest.fixture
def loader():
    x = torch.rand(N_SAMPLES, N_INPUTS, generator=torch.Generator().manual_seed(0))
    dummy_targets = torch.zeros(N_SAMPLES, N_OUTPUTS)  # predict_step expects (inputs, targets)
    return DataLoader(TensorDataset(x, dummy_targets), batch_size=32)


def predict(model, loader, tmp_path, ckpt_path=None):
    batches = make_trainer(tmp_path).predict(model, dataloaders=loader, ckpt_path=ckpt_path)
    return torch.cat(batches)


def test_forward_shape():
    model = LitMLPModel(N_INPUTS, N_OUTPUTS)
    assert model(torch.rand(7, N_INPUTS)).shape == (7, N_OUTPUTS)


def test_trainer_predict_shape(loader, tmp_path):
    out = predict(LitMLPModel(N_INPUTS, N_OUTPUTS), loader, tmp_path)
    assert out.shape == (N_SAMPLES, N_OUTPUTS)


def test_predict_from_checkpoint_restores_the_weights(loader, tmp_path):
    torch.manual_seed(0)
    trained = LitMLPModel(N_INPUTS, N_OUTPUTS)
    trainer = make_trainer(tmp_path)
    expected = torch.cat(trainer.predict(trained, dataloaders=loader))
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)

    torch.manual_seed(1)
    fresh = LitMLPModel(N_INPUTS, N_OUTPUTS)  # different random initial weights
    restored = predict(fresh, loader, tmp_path, ckpt_path=ckpt)
    assert torch.equal(restored, expected)


def test_checkpoint_keys_are_prefixed_with_mlp(loader, tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.predict(LitMLPModel(N_INPUTS, N_OUTPUTS), dataloaders=loader)
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)
    state = torch.load(ckpt, weights_only=True)["state_dict"]
    assert "mlp.input.weight" in state and "mlp.output.bias" in state
