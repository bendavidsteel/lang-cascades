import textwrap
import os

import hydra
from KDEpy import TreeKDE
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import polars as pl
import scipy.spatial
from scipy.stats import gaussian_kde
from tqdm import tqdm

def load_df(dir_path, filter_type, group_by_every='2d', min_filter_count=10, targets=None, keywords=None):
    target_file_paths = [os.path.join(dir_path, target_file_name) for target_file_name in os.listdir(dir_path) if target_file_name.endswith('trends.parquet.zstd') and not target_file_name.startswith('loaded_trends')]
    dfs = []
    for target_path in tqdm(target_file_paths, desc="Loading data files"):
        try:
            file_df = pl.read_parquet(target_path, columns=['createtime', 'volume', 'trend_mean', 'target', 'filter_type', 'filter_value'])\
                .group_by_dynamic('createtime', every=group_by_every, group_by='filter_value')\
                .agg([
                    pl.col('volume').sum(), 
                    pl.col('trend_mean').mean(), 
                    pl.col('target').first(), 
                    pl.col('filter_type').first()
                ])
            if min_filter_count is not None:
                file_df = file_df.join(
                    file_df.group_by('filter_value')\
                        .agg(pl.col('volume').sum())\
                        .filter(pl.col('volume') > min_filter_count)\
                        .select(['filter_value']), 
                    on='filter_value', 
                    how='inner'
                )
            elif targets is not None:
                file_df = file_df.filter(pl.col('target').is_in(targets))
            else:
                raise ValueError("Either min_filter_count or targets must be specified")
            dfs.append(file_df)
        except Exception as e:
            print(f"Error loading {target_path}: {e}")
    df = pl.concat(dfs, how='vertical_relaxed')
    if keywords:
        df = df.filter(pl.col('target').str.contains_any(keywords))
    df = df.filter(pl.col('filter_type') == filter_type)
    df = df.unique(['filter_value', 'target', 'createtime'])
    df = df.filter(pl.col('filter_value') != '')
    return df

def get_top_component_features(components, feature_names, n_features=3,
                               weights=None):
    """Get the top contributing features for each PCA component.

    `weights` is a per-feature observation count. A loading is a coefficient
    per unit of the component, so ranking by its magnitude alone treats a
    feature the fit barely constrains as if it defined the axis; weighting by
    sqrt(weights) ranks by the feature's share of the variance the component
    explains instead, which is n_j * loading^2 up to a constant. Reported
    loadings are unweighted either way -- only the order changes.
    """
    top_features = {}

    for i, component in enumerate(components):
        # Get absolute values to find strongest contributors regardless of direction
        abs_loadings = np.abs(component)
        score = abs_loadings if weights is None else abs_loadings * np.sqrt(weights)
        # Get indices of top contributing features
        top_indices = np.argsort(score)[-n_features:][::-1]
        # Get feature names and their loadings
        top_feature_info = []
        for idx in top_indices:
            feature_name = feature_names[idx]
            loading = component[idx]
            top_feature_info.append((feature_name, loading))
        
        top_features[f'PC{i+1}'] = top_feature_info
    
    return top_features

def format_pca_axis_label(component_num, top_features, max_chars=100,
                          wrap=None, prefix='PC'):
    """Format axis label with top contributing features.

    `wrap` breaks the loading list over lines of that width. A y label runs
    along the shorter side of a wide panel, where one long line overruns the
    axes and is clipped.
    """
    base_label = f'{prefix}{component_num}'
    
    # Format feature contributions
    feature_strs = []
    for feature_name, loading in top_features:
        # Truncate long feature names
        sign = '+' if loading > 0 else '-'
        feature_type = 'μ' if 'trend_mean' in feature_name else 'V' if 'volume' in feature_name else ''
        feature_desc = feature_name.replace('trend_mean_', '').replace('volume_', '')
        feature_strs.append(f'{sign}{feature_desc}')

    features_text = ', '.join(feature_strs)
    
    if wrap:
        features_text = textwrap.fill(features_text, width=wrap)
    full_label = f'{base_label}\n({features_text}, ...)'
    return full_label

def create_kde_background(coords, ax, alpha=0.3, levels=10, x_min=None, x_max=None, y_min=None, y_max=None):
    """Create kernel density estimation background with logarithmic scaling"""
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    
    # Create a grid for KDE evaluation
    if x_min is None:
        x_min = x_coords.min()
    if x_max is None:
        x_max = x_coords.max()
    if y_min is None:
        y_min = y_coords.min()
    if y_max is None:
        y_max = y_coords.max()

    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
        
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 50),
        np.linspace(y_min, y_max, 50)
    )

    # Calculate KDE
    coords_stacked = np.vstack([x_coords, y_coords])
    scipy_kde = gaussian_kde(coords_stacked, bw_method='silverman')
    # Calculate KDE using TreeKDE (faster than scipy.stats.gaussian_kde)    
    # use scipy for bw calculation as multi-variate case is not implemented in KDEpy                                                                                                 
    kde = TreeKDE(kernel='gaussian', bw=scipy_kde.factor)                                                                                                                               
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T                                                                                                                      
    density = kde.fit(coords_stacked.T).evaluate(grid_points)                                                                                               
    density = density.reshape(xx.shape)

    # positions = np.vstack([xx.ravel(), yy.ravel()])
    # density = kde(positions).reshape(xx.shape)

    gamma = 0.1
    vmin, vmax = density.min(), density.max()
    norm = matplotlib.colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # Create levels evenly spaced in normalized (power-transformed) space
    # This ensures the full color palette is used
    if isinstance(levels, int):
        # Generate evenly spaced values in [0, 1] (normalized space)
        normalized_levels = np.linspace(0, 1, levels + 1)
        # Transform back to data space: level = vmin + (vmax - vmin) * norm_val^(1/gamma)
        data_levels = vmin + (vmax - vmin) * (normalized_levels ** (1 / gamma))
    else:
        # levels already specified as array
        data_levels = levels

    # Plot density as contours with logarithmic normalization
    # Use LogNorm to display in log scale but keep original density values
    contours = ax.contourf(xx, yy, density, levels=data_levels, alpha=alpha, cmap='viridis', norm=norm)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return contours, (xx, yy), density

def calculate_flow_field(target_df, xx, yy, influence_radius=0.5):
    """Calculate flow directions at grid points based on trajectory data"""
    
    active_user_df = target_df.group_by('filter_value').len()
    target_df = target_df.join(active_user_df.filter(pl.col('len') < 2).select('filter_value'), on='filter_value', how='anti')\
        .unique(['filter_value', 'createtime'])\
        .sort(['filter_value', 'createtime']) # Sort by user and time
    target_df = target_df.with_columns([
        pl.col('coord_2d').arr.get(0).alias('x'),
        pl.col('coord_2d').arr.get(1).alias('y')
    ]).with_columns([
        (pl.col('x').shift(-1) - pl.col('x')).alias('u'), # TODO ensure normalizing by time interval (should be anyway but should check)
        (pl.col('y').shift(-1) - pl.col('y')).alias('v')
    ]).with_columns([
        pl.concat_arr(['u', 'v']).alias('flow_vector')
    ]).filter(pl.col('u').is_not_null())
    flow_positions = target_df['coord_2d'].to_numpy()
    flow_vectors = target_df['flow_vector'].to_numpy()

    flow_vectors = np.clip(flow_vectors, np.percentile(flow_vectors, 1, axis=0), np.percentile(flow_vectors, 99, axis=0))

    # Calculate flow field at grid points
    grid_shape = xx.shape
    # Build KDTree for efficient neighbor search
    tree = scipy.spatial.cKDTree(flow_positions)

    # Query all grid points at once
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    bandwidth = 0.1  # kernel bandwidth

    # Find all neighbors within radius for each grid point
    indices_list = tree.query_ball_point(grid_points, r=bandwidth * 3)  # 3-sigma cutoff

    flowmap = np.zeros((len(grid_points), 2))
    prior_mean = np.zeros(2)
    prior_variance = 1e-9

    for i, indices in tqdm(enumerate(indices_list), total=len(indices_list), desc="Computing flow field"):
        if len(indices) > 0:
            distances = np.linalg.norm(flow_positions[indices] - grid_points[i], axis=1)
            
            # Gaussian kernel instead of inverse distance
            weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            weights = weights / weights.sum()
            
            # bayesian update
            n = len(flow_vectors[indices])
            sample_mean = np.average(flow_vectors[indices], axis=0, weights=weights)
            obs_variance = np.var(flow_vectors[indices], ddof=1) # TODO keep obs variance, easimates diffusion

            weight = (n * prior_variance) / (obs_variance + n * prior_variance)
            flowmap[i] = (1 - weight) * prior_mean + weight * sample_mean

    flowmap = flowmap.reshape(grid_shape + (2,))

    u = flowmap[:, :, 0]
    v = flowmap[:, :, 1]

    return u, v

def plot_streamplot(ax, target_df_with_coords, grid_x, grid_y):
    u, v = calculate_flow_field(target_df_with_coords, grid_x, grid_y, influence_radius=0.3)
        
    # Create streamplot
    # Use a coarser grid for streamlines to avoid overcrowding
    step = 2  # Use every 2nd point for subplots
    x_stream = grid_x[::step, ::step]
    y_stream = grid_y[::step, ::step]
    u_stream = u[::step, ::step]
    v_stream = v[::step, ::step]
    
    # Only plot streamlines where there's significant flow
    flow_magnitude = np.sqrt(u_stream**2 + v_stream**2)
    # Create streamplot
    lw = 5 * flow_magnitude / flow_magnitude.max()
    ax.streamplot(x_stream, y_stream, u_stream, v_stream, 
                    density=1, linewidth=lw, color='gray',
                    arrowsize=1, arrowstyle='->')

def plot_kde_with_streamplot_subplot(target_df, coords, components, feature_names, ax, title, xlims=None, ylims=None, max_sample_points=500):
    """Plot logarithmic KDE background with streamplot for subplot"""
    
    if len(target_df) == 0 or len(coords) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title, fontsize=10)
        if xlims:
            ax.set_xlim(xlims)
        if ylims:
            ax.set_ylim(ylims)
        return
    
    # Add coordinates to dataframe
    target_df_with_coords = target_df.with_columns([
        pl.Series('x', coords[:, 0]),
        pl.Series('y', coords[:, 1])
    ])
    
    # Sample points for flow field calculation if dataset is large
    if len(target_df_with_coords) > max_sample_points:
        target_df_with_coords = target_df_with_coords.sample(n=max_sample_points)
    
    # Create KDE background
    contours, (grid_x, grid_y) = create_kde_background(coords, ax, alpha=0.7)
    
    # Calculate flow field for streamplot
    if grid_x is not None and grid_y is not None:
        plot_streamplot(ax, target_df_with_coords, grid_x, grid_y)
    
    # Set consistent axis limits
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
    
    # Get top contributing features for each PCA component
    top_features = get_top_component_features(components, feature_names, n_features=2)
    
    # Format axis labels with top contributing features
    x_label = format_pca_axis_label(1, top_features['PC1'])
    y_label = format_pca_axis_label(2, top_features['PC2'])
    
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_title(title, fontsize=10)

def plot_kde_with_streamplot(target_df, components, feature_names):
    """Plot logarithmic KDE background with streamplot showing flow patterns"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create KDE background
    coords = target_df.select(['x', 'y']).to_numpy()
    contours, (grid_x, grid_y) = create_kde_background(coords, ax, alpha=0.7)
    
    # Add colorbar for KDE density
    if contours is not None:
        cbar = plt.colorbar(contours, ax=ax, shrink=0.8)
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        cbar.set_label('Stance State Density', rotation=270, labelpad=15)

    # Calculate flow field for streamplot
    print("Calculating flow field...")
    u, v = calculate_flow_field(target_df, grid_x, grid_y, influence_radius=0.3)

    # Create streamplot
    # Use a coarser grid for streamlines to avoid overcrowding
    step = 3  # Use every 3rd point
    x_stream = grid_x[::step, ::step]
    y_stream = grid_y[::step, ::step]
    u_stream = u[::step, ::step]
    v_stream = v[::step, ::step]

    # Only plot streamlines where there's significant flow
    flow_magnitude = np.sqrt(u_stream**2 + v_stream**2)
    mask = flow_magnitude > np.percentile(flow_magnitude[flow_magnitude > 0], 25)  # Top 75% of flows

    # Create streamplot with variable linewidth
    lw = 5 * flow_magnitude / flow_magnitude.max()
    streams = ax.streamplot(x_stream, y_stream, u_stream, v_stream,
                           density=1.5, linewidth=lw, color='red',
                           arrowsize=1.5, arrowstyle='->')

    # Get top contributing features for each PCA component
    top_features = get_top_component_features(components, feature_names, n_features=3)

    # Format axis labels with top contributing features
    x_label = format_pca_axis_label(1, top_features['PC1'])
    y_label = format_pca_axis_label(2, top_features['PC2'])

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title('Logarithmic Density Stance Landscape with Flow\n(Streamlines show stance movement trends)',
                fontsize=14, pad=20)

    # Add legend explaining streamplot
    legend_elements = [
        Line2D([0], [0], color='red', alpha=0.7, linewidth=0.5,
               label='Weak flow'),
        Line2D([0], [0], color='red', alpha=0.7, linewidth=2,
               label='Moderate flow'),
        Line2D([0], [0], color='red', alpha=0.7, linewidth=4,
               label='Strong flow')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Flow Magnitude')
    
    plt.tight_layout()
    return fig

def plot_by_year(target_df, components, feature_names):
    """Create subplots showing PCA analysis for each year"""
    
    # Calculate global axis limits from all coordinates
    coords = target_df.select(['x', 'y']).to_numpy()
    x_min, x_max = np.percentile(coords[:, 0], [10, 90])
    y_min, y_max = np.percentile(coords[:, 1], [10, 90])
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = 0.05 * x_range
    y_padding = 0.05 * y_range
    
    xlims = (x_min - x_padding, x_max + x_padding)
    ylims = (y_min - y_padding, y_max + y_padding)
    
    # Extract year from createtime
    target_df = target_df.with_columns([
        pl.col('createtime').dt.year().alias('year')
    ])
    
    # Get unique years
    years = sorted(target_df['year'].unique().to_list())
    
    # Calculate subplot dimensions
    n_years = len(years)
    n_cols = min(2, n_years)
    n_rows = (n_years + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_years == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, year in enumerate(years):
        row = i // n_cols
        col = i % n_cols
        if n_cols == 1:
            ax = axes[i]
        else:
            ax = axes[row, col]
        
        # Filter data for this year
        year_df = target_df.filter(pl.col('year') == year)
        year_coords = year_df.select(['x', 'y']).to_numpy()
        
        # Create subplot with consistent limits
        plot_kde_with_streamplot_subplot(
            year_df, year_coords, components, feature_names, ax, 
            f'Year {year} (n={len(year_df)})',
            xlims=xlims, ylims=ylims
        )
    
    # Hide unused subplots
    for i in range(n_years, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    fig.suptitle('PCA Analysis by Year', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig

def plot_by_platform(target_df, components, feature_names):
    """Create subplots showing PCA analysis for each platform"""
    
    # Calculate global axis limits from all coordinates
    coords = target_df.select(['x', 'y']).to_numpy()
    x_min, x_max = np.percentile(coords[:, 0], [10, 90])
    y_min, y_max = np.percentile(coords[:, 1], [10, 90])
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = 0.05 * x_range
    y_padding = 0.05 * y_range
    
    xlims = (x_min - x_padding, x_max + x_padding)
    ylims = (y_min - y_padding, y_max + y_padding)
    
    # Extract platform from filter_value (format: <id>-<platform>-<name>)
    target_df_with_platform = target_df.with_columns([
        pl.col('filter_value').str.split('-').list.get(1).alias('platform'),
        pl.Series('x', coords[:, 0]),
        pl.Series('y', coords[:, 1])
    ])
    
    # Get unique platforms
    platforms = sorted(target_df_with_platform['platform'].unique().to_list())
    
    # Calculate subplot dimensions
    n_platforms = len(platforms)
    n_cols = min(2, n_platforms)
    n_rows = (n_platforms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_platforms == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, platform in enumerate(platforms):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Filter data for this platform
        platform_data = target_df_with_platform.filter(pl.col('platform') == platform)
        platform_coords = platform_data.select(['x', 'y']).to_numpy()
        
        # Create subplot with consistent limits
        plot_kde_with_streamplot_subplot(
            platform_data, platform_coords, components, feature_names, ax, 
            f'{platform} (n={len(platform_data)})',
            xlims=xlims, ylims=ylims
        )
    
    # Hide unused subplots
    for i in range(n_platforms, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.set_visible(False)
    
    fig.suptitle('PCA Analysis by Platform', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig

def pivot_trends(df, missing_cols=None):
    print("Pivoting data...")
    df = df.with_columns([
        pl.col('trend_mean').cast(pl.Float32),
        pl.col('filter_value').cast(pl.Enum(df['filter_value'].unique()))
    ])
    target_df = df.pivot(on='target', index=['createtime', 'filter_value'], values=['trend_mean'])

    stance_cols = [c for c in target_df.columns if c not in ['createtime', 'filter_value']]

    if missing_cols is not None:
        for col in missing_cols:
            target_df = target_df.with_columns(pl.lit(np.nan).alias(col))
        stance_cols = stance_cols + missing_cols

    # Forward/backward fill within each filter_value
    target_df = target_df.with_columns(
        [pl.col(c).backward_fill().over('filter_value') for c in stance_cols]
    ).with_columns(
        [pl.col(c).forward_fill().over('filter_value') for c in stance_cols]
    )

    return target_df, stance_cols


def pivot_and_impute(df, missing_cols=None, impute_fancy=False):
    target_df, stance_cols = pivot_trends(df, missing_cols=missing_cols)

    print("Imputing missing values...")
    if impute_fancy:
        X = target_df.select(stance_cols).to_numpy().astype(np.float32)

        # print how many missing values there are
        n_missing = np.isnan(X).sum()
        print(f"Total missing values before imputation: {n_missing} ({n_missing / X.size:.2%})")

        rank = np.power(X.shape[1], 1/3).astype(int) # rough heuristic for rank

        from fancyimpute import IterativeSVD
        imp = IterativeSVD(rank=rank, svd_algorithm="arpack")
        X_imputed = imp.fit_transform(X)

        target_df = target_df.with_columns([
            pl.Series(name=stance_cols[i], values=X_imputed[:, i]) for i in range(X_imputed.shape[1])
        ])
    else:
        target_df = target_df.with_columns(
            [pl.col(c).fill_null(strategy='mean') for c in stance_cols]
        )

    return target_df, stance_cols

def do_pca(target_df, feature_cols, n_components=2):
    X = target_df.select(feature_cols).to_numpy()
    
    print("Applying PCA...")
    
    if False:
        import cuml
        pca = cuml.PCA(n_components=n_components)
        coords = pca.fit_transform(X)
        components = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
    if False:
        # use sklearn PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X)
        components = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
    if True:
        import faiss
        pca = faiss.PCAMatrix(X.shape[1], n_components)
        pca.train(X)
        coords = pca.apply(X)
        components = faiss.vector_to_array(pca.A).reshape(n_components, X.shape[1])

        # Faiss centers the data internally, get the mean
        mean = faiss.vector_to_array(pca.mean).reshape(1, -1)

        # Center X using faiss's mean
        X_centered = X - mean

        # Now compute explained variance ratio
        # Variance of coords = eigenvalues (for centered data)
        explained_variance = np.var(coords, axis=0, ddof=0) * coords.shape[0]

        # Total variance = sum of all eigenvalues = trace of covariance matrix
        total_variance = np.var(X_centered, axis=0, ddof=0).sum() * X_centered.shape[0]

        explained_variance_ratio = explained_variance / total_variance

    return pca, coords, components, explained_variance_ratio

def plot_pcas(target_df, feature_cols, coords, components, name, get_platform=False):
    # Add coordinates to dataframe
    target_df = target_df.with_columns([
        pl.Series('x', coords[:, 0]),
        pl.Series('y', coords[:, 1])
    ])

    # Print top contributing features for each component
    
    top_features = get_top_component_features(components, feature_cols, n_features=5)
    for component, features in top_features.items():
        print(f"\n{component} top contributing features:")
        for feature_name, loading in features:
            print(f"  {feature_name}: {loading:.4f}")
    
    # Create output directory if it doesn't exist
    dir_path = f"./figs/{name}"
    os.makedirs(dir_path, exist_ok=True)

    # Create original combined KDE + streamplot
    print("Creating original KDE with streamplot...")
    fig = plot_kde_with_streamplot(target_df, components, feature_cols)
    fig.savefig(f'{dir_path}/pca_kde_streamplot.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {dir_path}/pca_kde_streamplot.png")
    
    # Create year-based subplots
    print("Creating year-based subplots...")
    fig_year = plot_by_year(target_df, components, feature_cols)
    fig_year.savefig(f'{dir_path}/pca_kde_streamplot_by_year.png', dpi=300, bbox_inches='tight')
    print(f"Year-based plot saved to {dir_path}/pca_kde_streamplot_by_year.png")

    # Create platform-based subplots
    if get_platform:
        print("Creating platform-based subplots...")
        fig_platform = plot_by_platform(target_df, components, feature_cols)
        fig_platform.savefig(f'{dir_path}/pca_kde_streamplot_by_platform.png', dpi=300, bbox_inches='tight')
        print(f"Platform-based plot saved to {dir_path}/pca_kde_streamplot_by_platform.png")

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    print("Loading data...")

    trend_name = os.path.basename(cfg.trend_path.rstrip('/'))
    # keywords = ['climate', 'carbon', 'energy', 'fossil', 'fuel', 'gas', '\boil\b', '\bcoal\b', 'solar']
    # dir_name = f"{trend_name}/climate"
    keywords = None
    dir_name = f"{trend_name}/all"

    target_path = os.path.join(cfg.trend_path, 'pca_coords.parquet.zstd')
    target_head_df = pl.read_parquet(target_path, n_rows=1)
    target_df = pl.read_parquet(target_path, columns=['createtime', 'filter_value', 'coord_2d'])
    components = np.load(os.path.join(cfg.trend_path, 'pca_components.npy'))
    stance_cols = [col for col in target_head_df.columns if col.startswith('trend_mean_')]
    target_df = target_df.select(['createtime', 'filter_value', 'coord_2d'])
    coords = target_df['coord_2d'].to_numpy()

    plot_pcas(target_df, stance_cols, coords, components, f'{dir_name}/stance', get_platform=cfg.filter_column=='PlatformHandleID')


if __name__ == '__main__':
    main()