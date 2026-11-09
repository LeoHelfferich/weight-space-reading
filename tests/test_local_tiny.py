"""The local runner finishes and writes aggregate files."""

from pathlib import Path

from wsp.local_experiment import run


def test_tiny_local_run_writes_summary_and_figures(tmp_path):
    config = {
        "seed": 0,
        "iou_threshold": 0.5,
        "ridge_l2": 10.0,
        "tasks": [
            {
                "name": "2d",
                "dim": 2,
                "classes": ["circle", "square"],
                "grid": 8,
                "hidden": 4,
                "train_per_class": 3,
                "test_per_class": 2,
                "inr_steps": 5,
                "inr_lr": 0.05,
                "init_modes": ["shared"],
                "center_limit": 0.1,
                "scale_min": 0.4,
                "scale_max": 0.5,
                "full_dimensional_control": False,
                "shuffled_model_control": False,
                "diffusion": {
                    "pca_components": 2,
                    "timesteps": 5,
                    "beta_end": 0.05,
                    "train_steps": 4,
                    "batch_size": 4,
                    "hidden": 8,
                    "lr": 0.001,
                    "time_dim": 4,
                    "samples_per_class": 1,
                    "denoise_timestep": 1,
                },
            }
        ],
    }
    summary = run(config, tmp_path)
    assert (tmp_path / "local_summary.json").exists()
    assert list((tmp_path / "figures").glob("*.png"))
    shared = summary["tasks"]["2d"]["shared"]
    trained = shared["controls_iou_to_own_target"]["trained"]["mean"]
    permuted = shared["controls_iou_to_own_target"]["permuted"]["mean"]
    assert trained == permuted
    assert "denoise_mse" in shared["diffusion_pca"]
