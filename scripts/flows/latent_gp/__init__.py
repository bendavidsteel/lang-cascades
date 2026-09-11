"""Latent-GP factor model over cell-level stance counts."""

__all__ = ['LatentConfig', 'build_latents', 'build_loadings', 'build_params',
           'loading_matrix', 'coord_cols', 'fit_dir', 'reference']


def __getattr__(name):
    # lazy so that the data-prep steps, which need neither jax nor a GPU, do
    # not initialise one just by importing the package
    if name in __all__:
        from . import latents
        return getattr(latents, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
