import datetime
import json
import os

import hydra
import jax
import jax.numpy as jnp
import matplotlib.animation
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import numpy as np
import omegaconf
import polars as pl
from tqdm import tqdm

from plnn.models import DeepTimePhiPLNN
from plnn.pl.plot_plnn import compute_grad_phi

import latent_space
import sweep_runs
from nn_potential import INITIAL_DATE, UNIT_DAYS, run_dir
from pca_density import create_kde_background, get_top_component_features, format_pca_axis_label


def compute_marginalized_grad_phi(model, t, z_2d, dim_1, dim_2, marginal_samples,
                                  mc_dropout=None, key=None, n_marginal=256):
    """Compute Boltzmann-weighted marginalized gradient over non-displayed dimensions.

    For each 2D grid point (x_i, x_j), constructs full-dimensional points by
    combining with samples from the empirical marginal distribution of the
    remaining dimensions, then computes:

        grad_ij = sum_k w_k * grad_phi(x_full_k)[dims] / sum_k w_k

    where w_k = exp(-phi(x_full_k)), i.e. the free energy projection.

    Args:
        model: PLNN model trained on all dimensions.
        t: Time scalar.
        z_2d: 2D grid points, shape (N, 2).
        dim_1, dim_2: Which dimensions are being displayed.
        marginal_samples: Full-dimensional data array, shape (M, D).
        mc_dropout: Number of MC dropout samples (or None).
        key: JAX PRNG key.
        n_marginal: Number of marginal samples to draw.

    Returns:
        (grad_phi_mean, grad_phi_std): each shape (N, 2), the marginalized
            gradient mean and std on the displayed dimensions.
    """
    n_grid = z_2d.shape[0]
    n_dims = marginal_samples.shape[1]
    display_dims = [dim_1, dim_2]
    rest_dims = [d for d in range(n_dims) if d not in display_dims]

    # Sample marginal points for the non-displayed dimensions
    rng = np.random.default_rng(42)
    idx = rng.choice(marginal_samples.shape[0], size=n_marginal, replace=True)
    rest_values = marginal_samples[idx][:, rest_dims]  # (n_marginal, D-2)

    # For each grid point, build full-dimensional points
    # z_2d: (N, 2), rest_values: (n_marginal, D-2)
    # We process in chunks to avoid memory issues
    grad_means = []
    grad_stds = []
    chunk_size = max(1, 2500 // n_marginal)  # keep total points manageable

    n_chunks = (n_grid + chunk_size - 1) // chunk_size
    for start in tqdm(range(0, n_grid, chunk_size), total=n_chunks, desc="Marginalizing grad_phi", leave=False):
        end = min(start + chunk_size, n_grid)
        z_chunk = z_2d[start:end]  # (C, 2)
        C = z_chunk.shape[0]

        # Build full-dimensional points: (C, n_marginal, D)
        z_full = np.zeros((C, n_marginal, n_dims), dtype=np.float64)
        z_full[:, :, dim_1] = z_chunk[:, 0:1]  # broadcast over marginal
        z_full[:, :, dim_2] = z_chunk[:, 1:2]
        z_full[:, :, rest_dims] = rest_values[np.newaxis, :, :]

        z_flat = jnp.array(z_full.reshape(-1, n_dims), dtype=jnp.float64)  # (C*n_marginal, D)

        t_jnp = jnp.array(t, dtype=jnp.float64)
        if mc_dropout:
            all_grad_phis = []
            all_phis = []
            for _ in range(mc_dropout):
                gp = model.grad_phi(t_jnp, z_flat, key=key, inference=False)
                p = model.phi(t_jnp, z_flat, key=key, inference=False)
                all_grad_phis.append(gp)
                all_phis.append(p)
            # Stack: (mc, C*n_marginal, D) and (mc, C*n_marginal)
            grad_phi_stack = jnp.stack(all_grad_phis)
            phi_stack = jnp.stack(all_phis)
            # Average over MC samples first
            grad_phi_all = jnp.mean(grad_phi_stack, axis=0)  # (C*n_marginal, D)
            phi_all = jnp.mean(phi_stack, axis=0)  # (C*n_marginal,)
            grad_phi_mc_std = jnp.std(grad_phi_stack, axis=0)  # for uncertainty
        else:
            grad_phi_all = model.grad_phi(t_jnp, z_flat, key=key, inference=False)
            phi_all = model.phi(t_jnp, z_flat, key=key, inference=False)
            grad_phi_mc_std = None

        # Reshape to (C, n_marginal, D) and (C, n_marginal)
        grad_phi_all = np.array(grad_phi_all).reshape(C, n_marginal, n_dims)
        phi_all = np.array(phi_all).reshape(C, n_marginal)

        # Extract only the displayed dimensions: (C, n_marginal, 2)
        grad_phi_2d = grad_phi_all[:, :, display_dims]

        # Boltzmann weights: w_k = exp(-phi_k), use log-sum-exp for stability
        neg_phi = -phi_all  # (C, n_marginal)
        log_weights = neg_phi - neg_phi.max(axis=1, keepdims=True)  # stabilize
        weights = np.exp(log_weights)  # (C, n_marginal)
        weights_sum = weights.sum(axis=1, keepdims=True)  # (C, 1)
        weights_norm = weights / weights_sum  # (C, n_marginal)

        # Weighted mean: (C, 2)
        chunk_grad_mean = np.sum(weights_norm[:, :, np.newaxis] * grad_phi_2d, axis=1)

        # Weighted std for uncertainty
        if grad_phi_mc_std is not None:
            mc_std_2d = np.array(grad_phi_mc_std).reshape(C, n_marginal, n_dims)[:, :, display_dims]
            chunk_grad_std = np.sum(weights_norm[:, :, np.newaxis] * mc_std_2d, axis=1)
        else:
            # Use weighted std across marginal samples as uncertainty
            diff = grad_phi_2d - chunk_grad_mean[:, np.newaxis, :]
            chunk_grad_std = np.sqrt(np.sum(weights_norm[:, :, np.newaxis] * diff**2, axis=1))

        grad_means.append(chunk_grad_mean)
        grad_stds.append(chunk_grad_std)

    grad_mean = jnp.array(np.concatenate(grad_means, axis=0))  # (N, 2)
    grad_std = jnp.array(np.concatenate(grad_stds, axis=0))    # (N, 2)
    return grad_mean, grad_std

def t_to_datetime(t):
    return INITIAL_DATE + datetime.timedelta(days=t * UNIT_DAYS)

def show_x_dim_labels(ax, dimension_labels, dim):
    # Get dimension 0 (x-axis) labels
    dim0 = dimension_labels.get(str(dim), {}).get('3_cat', {})
    x_tick_labels = []
    x_tick_positions = []

    if 'negative' in dim0:
        x_tick_labels.append(dim0['negative'])
        x_tick_positions.append(dim0['negative_threshold'])
    if 'neutral' in dim0:
        x_tick_labels.append(dim0['neutral'])
        x_tick_positions.append(dim0['mean'])
    if 'positive' in dim0:
        x_tick_labels.append(dim0['positive'])
        x_tick_positions.append(dim0['positive_threshold'])

    if x_tick_positions:
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=8, rotation=15, ha='right')

def show_y_dim_labels(ax, dimension_labels, dim):
    # Get dimension 1 (y-axis) labels
    dim1 = dimension_labels.get(str(dim), {}).get('3_cat', {})
    y_tick_labels = []
    y_tick_positions = []

    if 'negative' in dim1:
        y_tick_labels.append(dim1['negative'])
        y_tick_positions.append(dim1['negative_threshold'])
    if 'neutral' in dim1:
        y_tick_labels.append(dim1['neutral'])
        y_tick_positions.append(dim1['mean'])
    if 'positive' in dim1:
        y_tick_labels.append(dim1['positive'])
        y_tick_positions.append(dim1['positive_threshold'])

    if y_tick_positions:
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels, fontsize=8, rotation=15, ha='right')

def load_dimension_labels(dimension_labels_path):
    """Labels from describe_dimensions.py, or none: a new representation has
    none until it has been described, and the axes read fine without them."""
    if not os.path.exists(dimension_labels_path):
        print(f'no dimension labels at {dimension_labels_path}; '
              'axes will show loadings only')
        return {}
    with open(dimension_labels_path, 'r') as f:
        return json.load(f)


def setup_x_axis_labels(ax, components, feature_names, dim_1=0, weights=None):
    """Setup axis labels with PCA feature information and optional dimension descriptions

    `weights` is latents.rank_by_volume's, so an axis is named by the same
    targets the dimension table lists rather than by whichever loading the fit
    left largest.
    """
    top_features = get_top_component_features(components, feature_names,
                                              n_features=3, weights=weights)
    x_label = format_pca_axis_label(dim_1 + 1, top_features[f'PC{dim_1 + 1}'])
    ax.set_xlabel(x_label, fontsize=8)

def setup_y_axis_labels(ax, components, feature_names, dim_2=1, weights=None):
    """Setup axis labels with PCA feature information and optional dimension descriptions"""
    top_features = get_top_component_features(components, feature_names,
                                              n_features=3, weights=weights)
    y_label = format_pca_axis_label(dim_2 + 1, top_features[f'PC{dim_2 + 1}'])
    ax.set_ylabel(y_label, fontsize=8)
    

# Streamplot linewidth scaling parameters (log10-based so weak flows remain visible).
MIN_LW = 0.3
MAX_LW = 3.0
LOG_DECADES = 3


def flow_to_linewidth(flow_magnitude, max_val):
    """Map flow magnitude to streamplot linewidth via log10 scaling with a floor.

    Linewidths are linearly spaced in log10(flow) over LOG_DECADES orders of
    magnitude below ``max_val``; anything smaller is clipped to MIN_LW so weak
    flows remain visible.
    """
    if max_val <= 0:
        return np.full_like(flow_magnitude, 0.5 * (MIN_LW + MAX_LW))
    log_max_v = np.log10(max_val)
    log_min_v = log_max_v - LOG_DECADES
    log_range = log_max_v - log_min_v
    log_flow = np.log10(np.maximum(flow_magnitude, 10 ** log_min_v))
    return MIN_LW + (MAX_LW - MIN_LW) * (log_flow - log_min_v) / log_range


def linewidth_to_flow(lw, max_val):
    """Inverse of flow_to_linewidth — used to label legend entries."""
    if max_val <= 0:
        return 0.0
    log_max_v = np.log10(max_val)
    log_min_v = log_max_v - LOG_DECADES
    log_range = log_max_v - log_min_v
    log_flow = log_min_v + (lw - MIN_LW) / (MAX_LW - MIN_LW) * log_range
    return 10 ** log_flow


def build_legend_elements(flow_percentiles=None, show_hatching=True):
    """Build legend elements for streamplot and uncertainty hatching."""
    if flow_percentiles is not None:
        _, _, _, _, max_val = flow_percentiles
        fixed_lws = [2.5, 1.5, 0.5]
        legend_elements = []
        for lw in fixed_lws:
            flow_val = linewidth_to_flow(lw, max_val)
            legend_elements.append(
                matplotlib.lines.Line2D([0], [0], color='red', alpha=0.7,
                       linewidth=lw,
                       label=f'Flow = {flow_val:.2e}')
            )
    else:
        legend_elements = [
            matplotlib.lines.Line2D([0], [0], color='red', alpha=0.7, linewidth=2,
                   label='Flow patterns'),
        ]
    if show_hatching:
        legend_elements.extend([
            Patch(facecolor='none', edgecolor='black', hatch='/', label='Low uncertainty'),
            Patch(facecolor='none', edgecolor='black', hatch='//', label='Medium uncertainty'),
            Patch(facecolor='none', edgecolor='black', hatch='xxx', label='High uncertainty')
        ])
    return legend_elements

def add_legend(ax, flow_percentiles=None, show_hatching=True, **kwargs):
    """Add legend for streamplot and uncertainty hatching to an axes."""
    legend_elements = build_legend_elements(flow_percentiles, show_hatching=show_hatching)
    ax.legend(handles=legend_elements, loc='upper right', **kwargs)

def add_fig_legend(fig, flow_percentiles=None, show_hatching=True, **kwargs):
    """Add legend for streamplot and uncertainty hatching to a figure."""
    legend_elements = build_legend_elements(flow_percentiles, show_hatching=show_hatching)
    fig.legend(handles=legend_elements, loc='upper right', **kwargs)

def compute_flow_magnitude(model, t, xs, ys, confinement_threshold, mc_dropout, key,
                           dim_1=None, dim_2=None, marginal_samples=None, n_marginal=256):
    """Compute flow and uncertainty magnitudes without plotting, for pre-computing global normalization.

    Returns:
        (flow_magnitude, variance_magnitude): both with shape matching ``xs``.
    """
    z = np.array([xs.flatten(), ys.flatten()]).T
    z_norms = np.linalg.norm(z, ord=2, axis=1, keepdims=True)

    if marginal_samples is not None and dim_1 is not None and dim_2 is not None:
        grad_phi, grad_phi_std = compute_marginalized_grad_phi(
            model, t, z, dim_1, dim_2, marginal_samples,
            mc_dropout=mc_dropout, key=key, n_marginal=n_marginal)
    else:
        z = jnp.array(z, dtype=jnp.float64)
        grad_phi, grad_phi_std = compute_grad_phi(model, t, z, mc_dropout=mc_dropout, key=key)
    f = -np.array(grad_phi)
    f = np.where(z_norms > confinement_threshold, 0.0, f)

    fu, fv = f.T
    fu = fu.reshape(xs.shape)
    fv = fv.reshape(ys.shape)

    flow_magnitude = np.sqrt(fu**2 + fv**2)

    grad_phi_std_np = np.array(grad_phi_std)
    variance_magnitude = np.linalg.norm(grad_phi_std_np, axis=1).reshape(xs.shape)
    variance_magnitude = np.where(z_norms.reshape(xs.shape) > confinement_threshold, 0.0, variance_magnitude)

    return flow_magnitude, variance_magnitude

def plot_flow_field(ax, model, t, xs, ys, confinement_threshold, mc_dropout, key, max_flow=None, min_flow=None,
                    dim_1=None, dim_2=None, marginal_samples=None, n_marginal=256,
                    variance_levels=None, hatch_patterns=None, show_hatching=True):
    """Compute flow field from model gradient.

    If ``variance_levels`` and ``hatch_patterns`` are provided, they are used
    directly for the uncertainty hatch overlay — enabling consistent hatching
    thresholds across multiple axes. Otherwise they are derived from local
    quantiles of the variance magnitude on this grid.
    """
    z = np.array([xs.flatten(), ys.flatten()]).T
    z_norms = np.linalg.norm(z, ord=2, axis=1, keepdims=True)

    if marginal_samples is not None and dim_1 is not None and dim_2 is not None:
        grad_phi, grad_phi_std = compute_marginalized_grad_phi(
            model, t, z, dim_1, dim_2, marginal_samples,
            mc_dropout=mc_dropout, key=key, n_marginal=n_marginal)
    else:
        z = jnp.array(z, dtype=jnp.float64)
        grad_phi, grad_phi_std = compute_grad_phi(model, t, z, mc_dropout=mc_dropout, key=key)
    f = -np.array(grad_phi)

    # zero out flow outside confinement threshold
    f = np.where(z_norms > confinement_threshold, 0.0, f)

    fu, fv = f.T
    fu = fu.reshape(xs.shape)
    fv = fv.reshape(ys.shape)

    # Calculate flow magnitude and linewidth (log10-scaled so weak flows are visible)
    flow_magnitude = np.sqrt(fu**2 + fv**2)
    max_val = max_flow if max_flow is not None else flow_magnitude.max()
    min_val = min_flow if min_flow is not None else flow_magnitude.min()
    lw = flow_to_linewidth(flow_magnitude, max_val)
    flow_percentiles = (
        min_val,
        np.percentile(flow_magnitude, 25),
        np.percentile(flow_magnitude, 50),
        np.percentile(flow_magnitude, 75),
        max_val,
    )

    # Create streamplot
    ax.streamplot(xs, ys, fu, fv,
                 density=1.5, color='red',
                 arrowsize=1.5, linewidth=lw, arrowstyle='->')

    if not show_hatching:
        return fu, fv, grad_phi_std, flow_percentiles

    # Uncertainty hatching — shared across axes when caller supplies thresholds.
    grad_phi_std_np = np.array(grad_phi_std)
    grad_phi_std_mag = np.linalg.norm(grad_phi_std_np, axis=1).reshape(xs.shape)
    grad_phi_std_mag = np.where(z_norms.reshape(xs.shape) > confinement_threshold, 0.0, grad_phi_std_mag)

    if variance_levels is None or hatch_patterns is None:
        if np.quantile(grad_phi_std_mag, 0.25) > 0:
            variance_levels = [0, np.quantile(grad_phi_std_mag, 0.25), np.quantile(grad_phi_std_mag, 0.5),
                              np.quantile(grad_phi_std_mag, 0.75), grad_phi_std_mag.max()]
            hatch_patterns = ['', '/', '//', 'xxx']
        elif np.quantile(grad_phi_std_mag, 0.5) > 0:
            variance_levels = [0, np.quantile(grad_phi_std_mag, 0.5),
                              np.quantile(grad_phi_std_mag, 0.75), grad_phi_std_mag.max()]
            hatch_patterns = ['', '//', 'xxx']
        elif np.quantile(grad_phi_std_mag, 0.75) > 0:
            variance_levels = [0, np.quantile(grad_phi_std_mag, 0.75), grad_phi_std_mag.max()]
            hatch_patterns = ['', 'xxx']
        else:
            return fu, fv, grad_phi_std, flow_percentiles  # No variance to plot

    # Ensure the supplied top level exceeds any local data so contourf doesn't clip silently.
    local_max = float(grad_phi_std_mag.max())
    if local_max > variance_levels[-1]:
        variance_levels = list(variance_levels[:-1]) + [local_max]

    ax.contourf(xs, ys, grad_phi_std_mag,
               levels=variance_levels,
               hatches=hatch_patterns,
               colors='none',  # transparent fill
               alpha=0.0)

    return fu, fv, grad_phi_std, flow_percentiles

def build_global_variance_levels(all_variances):
    """Derive shared hatch thresholds from a pool of variance magnitudes.

    Returns (variance_levels, hatch_patterns) or (None, None) if there's no
    positive variance to overlay.
    """
    if len(all_variances) == 0:
        return None, None
    q25 = np.quantile(all_variances, 0.25)
    q50 = np.quantile(all_variances, 0.5)
    q75 = np.quantile(all_variances, 0.75)
    vmax = all_variances.max()
    if q25 > 0:
        return [0, q25, q50, q75, vmax], ['', '/', '//', 'xxx']
    if q50 > 0:
        return [0, q50, q75, vmax], ['', '//', 'xxx']
    if q75 > 0:
        return [0, q75, vmax], ['', 'xxx']
    return None, None


def add_colorbar(contours, ax, fig=None):
    if fig is None:
        fig = plt.gcf()
    cbar = fig.colorbar(contours, ax=ax, shrink=0.3, format='%.1e', aspect=40, pad=0.017)
    cbar.set_label('Log Stance State Density', rotation=270, labelpad=15)

def animate_density_streamplot(
        fig_path,
        model,
        target_df,
        components,
        coords,
        feature_names,
        t_range,
        spatial_res=100,
        temporal_res=100,
        temporal_stride=0.1,
        xrange=None,
        yrange=None,
        mc_dropout=None,
        key=None,
        t_to_datetime=None,
        confinement_threshold=None,
        dimension_labels=None,
        marginal_samples=None,
        n_marginal=256,
        n_dims=21,
        target_weights=None,
    ):
    coord_col = f'coord_{n_dims}d'
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Create KDE background
    # filter into t_range
    ts = np.linspace(t_range[0], t_range[1] - temporal_stride, temporal_res)
    filtered_df = filter_df_to_time_range(target_df, [ts[0], ts[0] + temporal_stride])

    x_min, x_max = target_df[coord_col].to_numpy()[:,0].min(), target_df[coord_col].to_numpy()[:,0].max()
    y_min, y_max = target_df[coord_col].to_numpy()[:,1].min(), target_df[coord_col].to_numpy()[:,1].max()

    coords = filtered_df[coord_col].to_numpy()
    
    contours, (grid_x, grid_y), density = create_kde_background(coords, ax, alpha=0.7, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    
    # Add colorbar for KDE density
    if contours is not None:
        add_colorbar(contours, ax)
    
    # Calculate flow field for streamplot
    print("Calculating flow field...")

    # Get grid
    x = np.linspace(*xrange, spatial_res)
    y = np.linspace(*yrange, spatial_res)
    xs, ys = np.meshgrid(x, y)
    t = ts[0] + (temporal_stride/2.0)

    # plot flow field
    _, _, _, flow_percentiles = plot_flow_field(
        ax, model, t, xs, ys, confinement_threshold, mc_dropout, key,
        dim_1=0, dim_2=1, marginal_samples=marginal_samples, n_marginal=n_marginal)


    # Setup axis labels
    setup_x_axis_labels(ax, components, feature_names, dim_1=0,
                        weights=target_weights)
    setup_y_axis_labels(ax, components, feature_names, dim_2=1,
                        weights=target_weights)

    show_x_dim_labels(ax, dimension_labels, dim=0)
    show_y_dim_labels(ax, dimension_labels, dim=1)

    ax.set_title(f't={t_to_datetime(t)}', fontsize=14, pad=20)

    # Add legend
    add_legend(ax, flow_percentiles=flow_percentiles)

    def update(start_t):
        ax.clear()

        filtered_df = filter_df_to_time_range(target_df, [start_t, start_t + temporal_stride])
        coords = filtered_df[coord_col].to_numpy()

        # Recreate KDE background
        create_kde_background(coords, ax, alpha=0.7, levels=contours._levels, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        # Calculate flow field
        t = start_t + (temporal_stride/2.0)
        _, _, _, flow_pctls = plot_flow_field(
            ax, model, t, xs, ys, confinement_threshold, mc_dropout, key,
            dim_1=0, dim_2=1, marginal_samples=marginal_samples, n_marginal=n_marginal)

        # Re-add axis labels and title
        setup_x_axis_labels(ax, components, feature_names, dim_1=0,
                        weights=target_weights)
        setup_y_axis_labels(ax, components, feature_names, dim_2=1,
                        weights=target_weights)

        show_x_dim_labels(ax, dimension_labels, dim=0)
        show_y_dim_labels(ax, dimension_labels, dim=1)

        ax.set_title(f't={t_to_datetime(t).date()}', fontsize=14, pad=20)

        # Re-add legend
        add_legend(ax, flow_percentiles=flow_pctls)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.set_box_aspect(1)

        return ax,
    
    fig.tight_layout()
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=tqdm(ts), interval=100)
    ani.save(f'{fig_path}/nn_potential_density_streamplot_animation.mp4')


def figure_density_streamplot(fig_path, model, target_df, components, feature_names, t_range, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_density_streamplot(fig, ax, model, target_df, components, feature_names, t_range, **kwargs)
    fig.tight_layout()
    fig.savefig(f'{fig_path}/nn_potential_density_streamplot.png', dpi=150, bbox_inches='tight', pad_inches=0.2)
    return fig
    
def filter_df_to_time_range(target_df, t_range):
    start_date = INITIAL_DATE + datetime.timedelta(days=t_range[0] * UNIT_DAYS)
    end_date = INITIAL_DATE + datetime.timedelta(days=t_range[1] * UNIT_DAYS)
    target_df = target_df.filter(
        (pl.col('createtime') >= start_date) &
        (pl.col('createtime') <= end_date)
    )
    return target_df

def plot_density_streamplot(
        fig,
        ax,
        model,
        target_df,
        components,
        feature_names,
        t_range,
        dim_1=0,
        dim_2=1,
        spatial_res=100,
        levels=10,
        xrange=None,
        yrange=None,
        mc_dropout=None,
        key=None,
        t_to_datetime=None,
        confinement_threshold=None,
        show_colorbar=True,
        dimension_labels=None,
        show_x_axis_labels=True,
        show_y_axis_labels=True,
        show_x_tick_labels=True,
        show_y_tick_labels=True,
        show_legend=True,
        max_flow=None,
        min_flow=None,
        marginal_samples=None,
        n_marginal=256,
        n_dims=21,
        variance_levels=None,
        hatch_patterns=None,
        show_kde=True,
        show_hatching=True,
        target_weights=None,
        **kwargs
    ):
    coord_col = f'coord_{n_dims}d'
    if show_kde:
        # filter into t_range
        target_df = filter_df_to_time_range(target_df, t_range)
        coords = target_df[coord_col].to_numpy()[:, [dim_1, dim_2]]
        contours, _, _ = create_kde_background(coords, ax, alpha=0.7, levels=levels, x_min=xrange[0], x_max=xrange[1], y_min=yrange[0], y_max=yrange[1])

        # Add colorbar for KDE density
        if show_colorbar and contours is not None:
            add_colorbar(contours, ax, fig=fig)
    else:
        contours = None

    # Calculate flow field for streamplot
    print("Calculating flow field...")

    # Get grid
    x = np.linspace(*xrange, spatial_res, dtype=np.float64)
    y = np.linspace(*yrange, spatial_res, dtype=np.float64)
    xs, ys = np.meshgrid(x, y)
    t = (t_range[0] + t_range[1]) / 2.0

    # Compute flow field
    _, _, _, flow_percentiles = plot_flow_field(
        ax, model, t, xs, ys, confinement_threshold, mc_dropout, key,
        max_flow=max_flow, min_flow=min_flow,
        dim_1=dim_1, dim_2=dim_2, marginal_samples=marginal_samples, n_marginal=n_marginal,
        variance_levels=variance_levels, hatch_patterns=hatch_patterns,
        show_hatching=show_hatching)

    # Setup axis labels (conditionally)
    if show_x_axis_labels:
        setup_x_axis_labels(ax, components, feature_names, dim_1=dim_1,
                            weights=target_weights)
    if show_y_axis_labels:
        setup_y_axis_labels(ax, components, feature_names, dim_2=dim_2,
                            weights=target_weights)

    if show_x_tick_labels:
        # Still set up tick labels without axis labels
        show_x_dim_labels(ax, dimension_labels, dim=dim_1)
    else:
        ax.set_xticks([])
    if show_y_tick_labels:
        show_y_dim_labels(ax, dimension_labels, dim=dim_2)
    else:
        ax.set_yticks([])

    # Set explicit axis limits to ensure streamplot/hatching fills the axes
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_box_aspect(1)

    # Add legend (conditionally)
    if show_legend:
        add_legend(ax, flow_percentiles=flow_percentiles)

    return ax, contours

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    # Whatever cfg.latents.method says the model was trained on. The helpers
    # below index the coordinate column by dimensionality, so it is renamed to
    # the name they expect rather than threading a column name through them.
    coord_col = f'coord_{cfg.n_dims}d'
    target_df, components, stance_cols = latent_space.load(cfg)
    target_df = target_df.rename({latent_space.COORD: coord_col})

    assert len(stance_cols) == components.shape[1]
    # the same ranking the dimension table uses, so a figure's axes are named
    # by the targets that carry them rather than by the largest raw loading
    target_weights = (latent_space.target_volumes(cfg, stance_cols)
                      if cfg.latents.get('rank_by_volume', True) else None)
    coords = target_df[coord_col].to_numpy()
    x_range = (np.percentile(coords[:,0], 0.5), np.percentile(coords[:,0], 99.5))
    y_range = (np.percentile(coords[:,1], 0.5), np.percentile(coords[:,1], 99.5))

    # Load dimension labels if available
    trend_path = cfg.trend_path
    trend_name = os.path.basename(trend_path.rstrip('/'))
    dimension_labels_path = os.path.join(trend_path, f'{latent_space.name(cfg)}_dimension_labels.json')
    dimension_labels = load_dimension_labels(dimension_labels_path)

    # with jax.default_device(jax.devices("cpu")[0]):
    seed = 42
    rng = np.random.default_rng(seed=seed)
    key = jax.random.PRNGKey(int(rng.integers(2**32)))
    key, modelkey, initkey, trainkey = jax.random.split(key, 4)

    dtype = jnp.float32

    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))

    fig_path = f'./figs/{trend_name}'
    os.makedirs(fig_path, exist_ok=True)

    # The platform panel needs one model per platform, trained separately.
    PLOT_ANI = False
    PLOT_TIME = True
    PLOT_PLATFORM = False
    PLOT_TRAJECTORIES = True
    PLOT_POTENTIAL_LANDSCAPES = False
    PLOT_SINGLE = True

    state_path = sweep_runs.state_path(run_dir(cfg))
    model, _ = DeepTimePhiPLNN.load(state_path, dtype=dtype)

    # Full-dimensional coords for marginalizing over non-displayed dims
    n_model_dims = cfg.n_dims
    marginal_samples = coords[:, :n_model_dims] if n_model_dims > 2 else None

    start_date = datetime.date(2022, 1, 1)
    # the precomputed coords stored createtime as a Date, a gpfa latent stores
    # it as a Datetime, and only one of those subtracts from a date
    end_date = target_df['createtime'].max()
    if isinstance(end_date, datetime.datetime):
        end_date = end_date.date()
    trange = ( (start_date - INITIAL_DATE.date()).days / UNIT_DAYS, (end_date - INITIAL_DATE.date()).days / UNIT_DAYS )

    plot_kwargs = {
        'mc_dropout': cfg.mc_dropout,
        'key': modelkey,
        't_to_datetime': t_to_datetime,
        'xrange': x_range,
        'yrange': y_range,
        't_range': trange,
        'confinement_threshold': model.confinement_threshold,
        'dimension_labels': dimension_labels,
        'spatial_res': 50,
        'temporal_res': 100,
        'marginal_samples': marginal_samples,
        'n_marginal': 256,
        'n_dims': cfg.n_dims,
        'target_weights': target_weights,
    }

    PLATFORMS = ['twitter', 'tiktok', 'instagram', 'bluesky']

    if PLOT_ANI:
        animate_density_streamplot(fig_path, model, target_df, components, coords, stance_cols, **plot_kwargs)

    if PLOT_SINGLE:
        figure_density_streamplot(fig_path, model, target_df, components, stance_cols, **plot_kwargs)
    

    if PLOT_TIME:
        years = [2023, 2024, 2025]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

        # Pre-compute flow magnitudes across all years for global linewidth normalization
        all_flow_magnitudes = []
        spatial_res = plot_kwargs.get('spatial_res', 50)
        x = np.linspace(*plot_kwargs['xrange'], spatial_res, dtype=np.float64)
        y = np.linspace(*plot_kwargs['yrange'], spatial_res, dtype=np.float64)
        xs, ys = np.meshgrid(x, y)
        for yr in tqdm(years, desc="Pre-computing flow (time)"):
            t_range = (datetime.datetime(yr, 1, 1) - INITIAL_DATE).days / UNIT_DAYS, (datetime.datetime(yr, 12, 31) - INITIAL_DATE).days / UNIT_DAYS
            t_mid = (t_range[0] + t_range[1]) / 2.0
            mag, _ = compute_flow_magnitude(
                model, t_mid, xs, ys,
                plot_kwargs['confinement_threshold'],
                plot_kwargs.get('mc_dropout'), plot_kwargs.get('key'),
                dim_1=0, dim_2=1, marginal_samples=marginal_samples,
                n_marginal=plot_kwargs.get('n_marginal', 256)
            )
            all_flow_magnitudes.append(mag)

        all_mags = np.concatenate([m.flatten() for m in all_flow_magnitudes])
        global_min = all_mags.min()
        global_p25 = np.percentile(all_mags, 25)
        global_p50 = np.percentile(all_mags, 50)
        global_p75 = np.percentile(all_mags, 75)
        global_max = all_mags.max()

        for i, yr in enumerate(tqdm(years, desc="Plotting time snapshots")):
            t_range = (datetime.datetime(yr, 1, 1) - INITIAL_DATE).days / UNIT_DAYS, (datetime.datetime(yr, 12, 31) - INITIAL_DATE).days / UNIT_DAYS
            plot_kwargs['t_range'] = t_range
            plot_kwargs['show_colorbar'] = False
            plot_kwargs['max_flow'] = global_max
            plot_kwargs['min_flow'] = global_min
            plot_kwargs['show_legend'] = False
            # one caption under the middle panel: the loading list is wide
            # enough that three of them collide across the row
            plot_kwargs['show_x_axis_labels'] = (i == len(years) // 2)
            plot_kwargs['show_y_axis_labels'] = (i == 0)
            plot_kwargs['show_x_tick_labels'] = True
            plot_kwargs['show_y_tick_labels'] = (i == 0)
            # the density is what changes between years; the flow barely does
            plot_kwargs['show_kde'] = True
            plot_kwargs['show_hatching'] = False
            plot_density_streamplot(fig, axes[i], model, target_df, components, stance_cols, **plot_kwargs)
            axes[i].set_title(f'{t_to_datetime(t_range[0]).strftime("%Y")}')

        fig.subplots_adjust(
            left=0.05,
            right=0.85,
            top=0.93,
            bottom=0.15,
            wspace=0.08
        )

        # Add shared legend on top-right axes
        global_flow_percentiles = (global_min, global_p25, global_p50, global_p75, global_max)
        add_legend(axes[-1], flow_percentiles=global_flow_percentiles, show_hatching=False, bbox_to_anchor=(1.5, 1.0))

        fig.savefig(f'{fig_path}/nn_potential_density_streamplot_time_snapshots.png', bbox_inches='tight', dpi=150, pad_inches=0)

    if PLOT_PLATFORM:
        platform_pretty = {
            'twitter': 'Twitter',
            'tiktok': 'TikTok',
            'instagram': 'Instagram',
            'bluesky': 'Bluesky'
        }

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

        # One model per platform, each trained by nn_potential.py under a cfg
        # differing only in `platform`, so run_dir locates it without a second
        # opinion about where a run is written. cfg is that run's own
        # configuration at platform='all': a handle-keyed representation
        # projected onto the pooled fit, which is the space every panel shares.
        # Per-platform marginal_samples come from the matching subset, so
        # marginalising over the non-displayed dimensions reflects that
        # platform's own distribution.
        platform_models = {}
        platform_dfs = {}
        platform_marginal_samples = {}
        for platform in PLATFORMS:
            platform_cfg = omegaconf.OmegaConf.merge(cfg, {'platform': platform})
            platform_state_path = sweep_runs.state_path(run_dir(platform_cfg))
            platform_models[platform], _ = DeepTimePhiPLNN.load(platform_state_path, dtype=dtype)

            platform_dfs[platform] = latent_space.keep_platform(platform_cfg, target_df)
            plat_coords = platform_dfs[platform][f'coord_{cfg.n_dims}d'].to_numpy()
            platform_marginal_samples[platform] = plat_coords[:, :cfg.n_dims] if cfg.n_dims > 2 else None

        # Pre-compute flow magnitudes across all platforms for global linewidth normalization
        all_flow_magnitudes = []
        spatial_res = plot_kwargs.get('spatial_res', 50)
        x = np.linspace(*plot_kwargs['xrange'], spatial_res, dtype=np.float64)
        y = np.linspace(*plot_kwargs['yrange'], spatial_res, dtype=np.float64)
        xs, ys = np.meshgrid(x, y)
        t_mid = (trange[0] + trange[1]) / 2.0
        for platform in tqdm(PLATFORMS, desc="Pre-computing flow (platform)"):
            mag, _ = compute_flow_magnitude(
                platform_models[platform], t_mid, xs, ys,
                platform_models[platform].confinement_threshold,
                plot_kwargs.get('mc_dropout'), plot_kwargs.get('key'),
                dim_1=0, dim_2=1, marginal_samples=platform_marginal_samples[platform],
                n_marginal=plot_kwargs.get('n_marginal', 256)
            )
            all_flow_magnitudes.append(mag)

        all_mags = np.concatenate([m.flatten() for m in all_flow_magnitudes])
        global_min = all_mags.min()
        global_p25 = np.percentile(all_mags, 25)
        global_p50 = np.percentile(all_mags, 50)
        global_p75 = np.percentile(all_mags, 75)
        global_max = all_mags.max()

        for i, platform in enumerate(tqdm(PLATFORMS, desc="Plotting platforms")):
            platform_model = platform_models[platform]
            platform_df = platform_dfs[platform]

            platform_plot_kwargs = plot_kwargs.copy()
            platform_plot_kwargs['t_range'] = trange
            platform_plot_kwargs['show_colorbar'] = False
            platform_plot_kwargs['max_flow'] = global_max
            platform_plot_kwargs['min_flow'] = global_min
            platform_plot_kwargs['show_legend'] = False
            platform_plot_kwargs['confinement_threshold'] = platform_model.confinement_threshold
            platform_plot_kwargs['marginal_samples'] = platform_marginal_samples[platform]
            platform_plot_kwargs['show_x_tick_labels'] = True
            platform_plot_kwargs['show_y_tick_labels'] = (i == 0)
            platform_plot_kwargs['show_x_axis_labels'] = True
            platform_plot_kwargs['show_y_axis_labels'] = (i == 0)
            platform_plot_kwargs['show_kde'] = False
            platform_plot_kwargs['show_hatching'] = False

            plot_density_streamplot(
                fig, axes[i], platform_model, platform_df, components, stance_cols,
                **platform_plot_kwargs
            )
            axes[i].set_title(platform_pretty[platform])

        fig.subplots_adjust(
            left=0.05,
            right=0.85,
            top=0.93,
            bottom=0.15,
            wspace=0.08
        )

        # Add shared legend on top-right axes
        global_flow_percentiles = (global_min, global_p25, global_p50, global_p75, global_max)
        add_legend(axes[-1], flow_percentiles=global_flow_percentiles, show_hatching=False, bbox_to_anchor=(1.5, 1.0))

        fig.savefig(f'{fig_path}/nn_potential_density_streamplot_platform_snapshots.png', bbox_inches='tight', dpi=150, pad_inches=0)

    NUM_DIMS = cfg.n_dims

    if (PLOT_TRAJECTORIES or PLOT_POTENTIAL_LANDSCAPES) \
            and cfg.latents.method != 'gpfa':
        n_dims = cfg.n_dims
        target_df = target_df.sort(['filter_value', 'createtime']) \
            .with_columns([
                pl.col(f'coord_{cfg.n_dims}d').arr.get(i).alias(f'dim_{i}') for i in range(n_dims)
            ]) \
            .rolling('createtime', period=f'{cfg.rolling_mean_window}d', group_by='filter_value') \
            .agg([pl.col(f'dim_{i}').mean() for i in range(n_dims)]) \
            .with_columns(
                pl.concat_arr([f'dim_{i}' for i in range(n_dims)]).alias(f'coord_{cfg.n_dims}d')
            ) \
            .drop([f'dim_{i}' for i in range(n_dims)])

    if PLOT_TRAJECTORIES or PLOT_POTENTIAL_LANDSCAPES:
        target_df = filter_df_to_time_range(target_df, trange)

    if PLOT_TRAJECTORIES:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for user_target_df in target_df.partition_by('filter_value'):
            ax.plot(
                user_target_df[f'coord_{cfg.n_dims}d'].arr.get(0),
                user_target_df[f'coord_{cfg.n_dims}d'].arr.get(1),
                alpha=0.1
            )

        setup_x_axis_labels(ax, components, stance_cols, dim_1=0,
                            weights=target_weights)
        show_x_dim_labels(ax, dimension_labels, dim=0)
        setup_y_axis_labels(ax, components, stance_cols, dim_2=1,
                            weights=target_weights)
        show_y_dim_labels(ax, dimension_labels, dim=1)

        x_range = (np.percentile(coords[:,0], 0.5), np.percentile(coords[:,0], 99.5))
        y_range = (np.percentile(coords[:,1], 0.5), np.percentile(coords[:,1], 99.5))
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_box_aspect(1)

        fig.savefig(f'{fig_path}/nn_potential_trajectories.png', dpi=150, bbox_inches='tight', pad_inches=0.2)

    if PLOT_POTENTIAL_LANDSCAPES and NUM_DIMS >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        dim_configs = [(0, 1), (0, 2), (1, 2)]

        # Pre-compute flow and variance magnitudes for global normalization (single model, marginalized)
        all_flow_magnitudes = []
        all_variance_magnitudes = []

        t_mid = (trange[0] + trange[1]) / 2.0
        for dim_1, dim_2 in tqdm(dim_configs, desc="Pre-computing flow (dims)"):
            x_r = (np.percentile(coords[:, dim_1], 0.5), np.percentile(coords[:, dim_1], 99.5))
            y_r = (np.percentile(coords[:, dim_2], 0.5), np.percentile(coords[:, dim_2], 99.5))
            x = np.linspace(*x_r, plot_kwargs.get('spatial_res', 50), dtype=np.float64)
            y = np.linspace(*y_r, plot_kwargs.get('spatial_res', 50), dtype=np.float64)
            xs, ys = np.meshgrid(x, y)
            mag, var_mag = compute_flow_magnitude(
                model, t_mid, xs, ys,
                plot_kwargs['confinement_threshold'],
                plot_kwargs.get('mc_dropout'), plot_kwargs.get('key'),
                dim_1=dim_1, dim_2=dim_2, marginal_samples=marginal_samples,
                n_marginal=plot_kwargs.get('n_marginal', 256)
            )
            all_flow_magnitudes.append(mag)
            all_variance_magnitudes.append(var_mag)

        all_mags = np.concatenate([m.flatten() for m in all_flow_magnitudes])
        global_min = all_mags.min()
        global_p25 = np.percentile(all_mags, 25)
        global_p50 = np.percentile(all_mags, 50)
        global_p75 = np.percentile(all_mags, 75)
        global_max = all_mags.max()

        all_vars = np.concatenate([m.flatten() for m in all_variance_magnitudes])
        variance_levels, hatch_patterns = build_global_variance_levels(all_vars)

        plot_kwargs['show_colorbar'] = False
        plot_kwargs['show_legend'] = False
        plot_kwargs['max_flow'] = global_max
        plot_kwargs['min_flow'] = global_min
        plot_kwargs['variance_levels'] = variance_levels
        plot_kwargs['hatch_patterns'] = hatch_patterns
        plot_kwargs['show_x_tick_labels'] = True
        plot_kwargs['show_x_axis_labels'] = True

        for i, (dim_1, dim_2) in enumerate(dim_configs):
            x_range = (np.percentile(coords[:, dim_1], 0.5), np.percentile(coords[:, dim_1], 99.5))
            y_range = (np.percentile(coords[:, dim_2], 0.5), np.percentile(coords[:, dim_2], 99.5))
            plot_kwargs['xrange'] = x_range
            plot_kwargs['yrange'] = y_range
            plot_kwargs['show_y_tick_labels'] = (i == 0)
            plot_kwargs['show_y_axis_labels'] = (i == 0)

            plot_density_streamplot(fig, axes[i], model, target_df, components, stance_cols,
                                    dim_1=dim_1, dim_2=dim_2, **plot_kwargs)

        global_flow_percentiles = (global_min, global_p25, global_p50, global_p75, global_max)
        add_legend(axes[-1], flow_percentiles=global_flow_percentiles,
                   bbox_to_anchor=(1.4, 1.0))

        fig.savefig(f'{fig_path}/nn_potential_landscape_row.png', dpi=150, bbox_inches='tight', pad_inches=0.2)

    
if __name__ == '__main__':
    main()  