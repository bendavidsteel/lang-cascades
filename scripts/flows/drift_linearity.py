"""How much of a trained landscape's drift field is one linear contraction.

The objective scores the model against a momentum rule, and momentum cannot
predict reversion, so a field that is nothing but a uniform pull toward the
origin already beats it. That field is also what a mean-reverting latent prior
writes into the state between observations. This separates the two: a drift
field a single linear map reproduces has learned no landscape, whatever it
scores.

Run against a checkpoint with the overrides that trained it:

    python scripts/flows/drift_linearity.py <the trial's overrides>
"""

import logging

import hydra
import jax
import jax.numpy as jnp
import numpy as np

import latent_space
import splits
import sweep_runs

from plnn.models import DeepTimePhiPLNN

logger = logging.getLogger(__name__)


def drift(model, t0, x0, batch=8192):
    """Deterministic drift -grad_phi at each row's own time.

    Dropout off: the diagnostic is about the field the model represents, not
    about the spread of the ensemble around it.
    """
    f = jax.jit(jax.vmap(lambda t, y: -model.eval_grad_phi(t, y, None, True)))
    out = [np.asarray(f(jnp.asarray(t0[i:i + batch]), jnp.asarray(x0[i:i + batch])))
           for i in range(0, len(x0), batch)]
    return np.concatenate(out)


def linear_fit(x, v):
    """Least squares v ~ A x + b over every dimension at once."""
    X = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    W, *_ = np.linalg.lstsq(X, v, rcond=None)
    pred = X @ W
    ss_res = float(((v - pred) ** 2).sum())
    ss_tot = float(((v - v.mean(0)) ** 2).sum())
    return W[:-1].T, W[-1], 1 - ss_res / ss_tot


def report(model, t0, x0, v_obs=None):
    """Linearity of the model's drift, and of the displacements it was fitted to.

    r2 near 1 is the collapse this exists to catch. eig_real_mean is the
    contraction rate of the linear part, in the same time unit as t0, and is
    negative for a field that pulls inward.
    """
    v = drift(model, t0, x0)
    A, _, r2 = linear_fit(x0, v)
    eig = np.linalg.eigvals(A).real
    out = {'drift/linear_r2': r2,
           'drift/eig_real_mean': float(eig.mean()),
           'drift/eig_real_max': float(eig.max()),
           'drift/magnitude': float(np.linalg.norm(v, axis=1).mean())}
    if v_obs is not None:
        A_obs, _, r2_obs = linear_fit(x0, v_obs)
        out['drift/observed_linear_r2'] = r2_obs
        out['drift/observed_eig_real_mean'] = float(np.linalg.eigvals(A_obs).real.mean())
    return out


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    # imported here rather than at the top because nn_potential imports
    # this module for the same report
    from nn_potential import build_training_pairs, load_latent_df, run_dir

    spec = splits.SplitSpec.from_cfg(cfg)
    dir_path = run_dir(cfg)
    state = sweep_runs.state_path(dir_path)
    if state is None:
        raise SystemExit(f'no checkpoint under {dir_path}')
    logger.info(f'loading {state}')
    model, _ = DeepTimePhiPLNN.load(state, dtype=jnp.float32)

    target_df = load_latent_df(cfg, spec)
    smooth = cfg.latents.method != 'gpfa'
    grid_days = cfg.latents.interp_days or 2 * cfg.latents.bin_factor
    pairs = build_training_pairs(cfg, target_df, smooth=smooth,
                                 max_step_days=10 if smooth else 1.5 * grid_days)
    labelled = splits.label_pairs(pairs, spec, time_col='next_createtime',
                                  key=latent_space.split_key(cfg))
    held = splits.select(labelled, 'val', ['out'])
    logger.info(f'{len(held)} val_out pairs')

    x0 = held['x0'].to_numpy().astype(np.float32)
    t0 = held['t0'].to_numpy().astype(np.float32)
    x1 = held['x1'].to_numpy().astype(np.float32)
    t1 = held['t1'].to_numpy().astype(np.float32)
    v_obs = (x1 - x0) / (t1 - t0)[:, None]

    for k, val in report(model, t0, x0, v_obs).items():
        print(f'{k:34s} {val: .4f}')


if __name__ == '__main__':
    main()
