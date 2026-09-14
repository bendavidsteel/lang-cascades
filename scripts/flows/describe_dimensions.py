import json
import logging
import os

import hydra
import numpy as np
import polars as pl
from nltk.corpus import stopwords

from toponymy import KeyphraseBuilder, ClusterLayerText
from toponymy.embedding_wrappers import VLLMEmbedder
from toponymy.llm_wrappers import AsyncVLLMNamer

import latent_space
from latent_space import COORD
from pca_density import get_top_component_features, format_pca_axis_label

logger = logging.getLogger(__name__)

def load_text_df(cfg, columns=['id', 'createtime', 'seed', 'Document', 'Targets', 'Stances']):
    dir_path = cfg.base_stance_path
    df = pl.read_parquet([os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith('.parquet.zstd')], columns=columns)
    # The corpus stamps createtime UTC-aware; a gpfa latent's is naive, built
    # off the aggregate's bin grid. polars refuses to pick a supertype for the
    # two, so the date-range join below fails unless one side is normalised --
    # and it has to be this one, since the plotting path compares the latent's
    # createtime against naive datetimes.
    if df.schema['createtime'].time_zone is not None:
        df = df.with_columns(pl.col('createtime').dt.convert_time_zone('UTC')
                             .dt.replace_time_zone(None))
    return df

def band_documents(target_df, text_df, pca_feature_df, dim_col, filter_val_col,
                   n_exemplars):
    """Every document tagged with where its author stood the day they posted it.

    A trajectory crosses the axis over its life, so a document belongs to the
    band its author occupied at the time. Pooling everything between a user's
    first and last visit to a band instead hands the middle band, which every
    trajectory passes through, nearly the whole corpus.
    """
    rows = target_df.select([
        'filter_value', 'createtime', pl.col(dim_col).alias('band_value')
    ]).sort('createtime')
    # the bins are a shared grid, so one spacing covers every trajectory: a
    # document counts only if a latent observation sits within a bin of it
    grid = rows.select(pl.col('createtime').unique().sort().diff().median()).item()

    cols = ['id', 'Document', 'createtime', 'filter_value']
    if 'embedding' in text_df.columns:
        cols.append('embedding')

    def tag(on_loaded_targets):
        docs = text_df.explode(['Targets', 'Stances']) \
            .rename({'Targets': 'Target', 'Stances': 'Stance'})
        if on_loaded_targets:
            docs = docs.join(pca_feature_df, on='Target', how='inner')
        # one row per document: exploding targets would otherwise let a post
        # that names four loaded targets stand in for four posts
        return docs.rename({filter_val_col: 'filter_value'}) \
            .unique(subset=['id']).select(cols).sort('createtime') \
            .join_asof(rows, on='createtime', by='filter_value',
                       strategy='nearest', tolerance=grid) \
            .drop_nulls('band_value')

    tagged = tag(True)
    # a dimension whose targets are barely posted about leaves too little to
    # name; fall back to everything its users wrote
    if tagged.height < 5 * n_exemplars:
        tagged = tag(False)
    return tagged


def get_dimension_description(target_df: pl.DataFrame, dim: int, dim_pca_features: list, text_df: pl.DataFrame,
                              embedding_model, llm, num_cats=5,
                              n_exemplars=32, n_keyphrases=32,
                              exemplar_selection_method='random', filter_val_col='SeedName',
                              axis_prefix='PC', seed=42):
    dim_col = f'dim_{dim}'
    pca_feature_df = pl.from_records(dim_pca_features, schema={'Target': pl.String, 'Loading': pl.Float32})
    if num_cats == 5:
        # Get thresholds for this dimension using polars
        v_pos_threshold = target_df.select(pl.col(dim_col).quantile(0.99)).item()
        v_neg_threshold = target_df.select(pl.col(dim_col).quantile(0.01)).item()
        positive_threshold = target_df.select(pl.col(dim_col).quantile(0.9)).item()
        negative_threshold = target_df.select(pl.col(dim_col).quantile(0.1)).item()
        mean_val = target_df.select(pl.col(dim_col).mean()).item()

        # Split the documents by where their author stood when they posted
        tagged = band_documents(target_df, text_df, pca_feature_df, dim_col,
                                filter_val_col, n_exemplars)
        v = pl.col('band_value')
        v_pos_text_df = tagged.filter(v >= v_pos_threshold)\
            .with_columns(pl.lit(0).alias('Topic'))
        v_neg_text_df = tagged.filter(v <= v_neg_threshold)\
            .with_columns(pl.lit(1).alias('Topic'))
        pos_text_df = tagged.filter((v >= positive_threshold) & (v < v_pos_threshold))\
            .with_columns(pl.lit(2).alias('Topic'))
        neg_text_df = tagged.filter((v <= negative_threshold) & (v > v_neg_threshold))\
            .with_columns(pl.lit(3).alias('Topic'))
        neutral_text_df = tagged.filter((v < positive_threshold) & (v > negative_threshold))\
            .with_columns(pl.lit(4).alias('Topic'))
        
        assert min(len(v_pos_text_df), len(v_neg_text_df), len(pos_text_df), len(neg_text_df), len(neutral_text_df)) > n_exemplars, "Insufficient documents in one of the categories to select exemplars from. Consider reducing the number of categories or lowering the percentile thresholds."

        # Combine for ctfidf calculation and join with embeddings
        cols = ['id', 'Document', 'Topic']
        if 'embedding' in v_pos_text_df.columns:
            cols.append('embedding')
        combined_df = pl.concat([
            v_pos_text_df.select(cols),
            v_neg_text_df.select(cols),
            pos_text_df.select(cols),
            neg_text_df.select(cols),
            neutral_text_df.select(cols)
        ], how='diagonal_relaxed')

        logging.info(f"""Dimension {dim}: 
            {v_pos_text_df.shape[0]} very positive docs, 
            {v_neg_text_df.shape[0]} very negative docs,
            {pos_text_df.shape[0]} positive docs, 
            {neg_text_df.shape[0]} negative docs, 
            {neutral_text_df.shape[0]} neutral docs""")
    elif num_cats == 3:
        # Get thresholds for this dimension using polars
        positive_threshold = target_df.select(pl.col(dim_col).quantile(0.95)).item()
        negative_threshold = target_df.select(pl.col(dim_col).quantile(0.05)).item()
        mean_val = target_df.select(pl.col(dim_col).mean()).item()

        # Split the documents by where their author stood when they posted
        tagged = band_documents(target_df, text_df, pca_feature_df, dim_col,
                                filter_val_col, n_exemplars)
        v = pl.col('band_value')
        pos_text_df = tagged.filter(v >= positive_threshold)\
            .with_columns(pl.lit(2).alias('Topic'))
        neg_text_df = tagged.filter(v <= negative_threshold)\
            .with_columns(pl.lit(3).alias('Topic'))
        neutral_text_df = tagged.filter((v < positive_threshold) & (v > negative_threshold))\
            .with_columns(pl.lit(4).alias('Topic'))
        
        assert min(len(pos_text_df), len(neg_text_df), len(neutral_text_df)) > n_exemplars, "Insufficient documents in one of the categories to select exemplars from. Consider reducing the number of categories or lowering the percentile thresholds."

        # Combine for ctfidf calculation and join with embeddings
        cols = ['id', 'Document', 'Topic']
        if 'embedding' in pos_text_df.columns:
            cols.append('embedding')
        combined_df = pl.concat([
            pos_text_df.select(cols),
            neg_text_df.select(cols),
            neutral_text_df.select(cols)
        ], how='diagonal_relaxed')

        logging.info(f"""Dimension {dim}: 
            {pos_text_df.shape[0]} positive docs, 
            {neg_text_df.shape[0]} negative docs, 
            {neutral_text_df.shape[0]} neutral docs""")
    else:
        raise ValueError("num_cats must be 3 or 5")

    if combined_df.is_empty():
        logging.info(f"No documents found for dimension {dim}")
        return {}

    # Use Toponymy methods for keyphrase extraction, exemplar selection, and topic naming
    extra_french_stopwords = ['quoi', 'alors', 'pourquoi', 'depuis', 'maintenant', 'où']
    english_french_stop_words = stopwords.words('english') + stopwords.words('french') + ['rt', 'url'] + extra_french_stopwords

    # Filter out documents without embeddings (those that were too long)
    if 'embedding' in combined_df.columns:
        combined_df = combined_df.filter(pl.col('embedding').is_not_null()).with_row_index()
    documents = combined_df['Document'].to_list()
    topics = combined_df['Topic'].to_numpy()

    # Build keyphrase extractor
    keyphrase_builder = KeyphraseBuilder(
        ngram_range=(1, 3),
        stop_words=set(english_french_stop_words),
        embedder=embedding_model,
        verbose=True
    )
    keyphrase_matrix, keyphrases, keyphrase_vectors = keyphrase_builder.fit_transform(documents)

    if exemplar_selection_method == 'random':
        centroid_vectors = np.zeros((num_cats, 1))
        document_vectors = None
    else:
        # Compute centroid vectors for each topic
        # Extract embeddings from dataframe
        logging.info("Extracting pre-computed embeddings...")
        document_vectors = combined_df['embedding'].to_numpy()

        unique_topics = np.unique(topics)
        centroid_vectors = np.zeros((len(unique_topics), document_vectors.shape[1]))
        for i, topic in enumerate(unique_topics):
            topic_mask = topics == topic
            centroid_vectors[i] = document_vectors[topic_mask].mean(axis=0)

    # Create cluster layer for the 5 topics
    cluster_layer = ClusterLayerText(
        cluster_labels=topics,
        centroid_vectors=centroid_vectors,
        layer_id=0,
        n_keyphrases=n_keyphrases,
        n_exemplars=n_exemplars,
        verbose=True,
        show_progress_bar=True
    )

    # Generate keyphrases for each topic
    cluster_keyphrases = cluster_layer.make_keyphrases(
        keyphrase_list=keyphrases,
        object_x_keyphrase_matrix=keyphrase_matrix,
        keyphrase_vectors=keyphrase_vectors,
        method="information_weighted"
    )

    # Select exemplar texts for each topic, in cluster-label order: that is the
    # order toponymy returns names in, and the labels are read off by position
    indexed = combined_df if 'index' in combined_df.columns else combined_df.with_row_index()
    cluster_exemplars = [
        indexed.filter(pl.col('Topic') == topic).select(['index', 'Document'])
               .sample(n_exemplars, seed=seed).to_dict(as_series=False)
        for topic in sorted(set(topics.tolist()))
    ]
    cluster_layer.exemplars = [exemplar['Document'] for exemplar in cluster_exemplars]
    cluster_layer.exemplar_indices = [exemplar['index'] for exemplar in cluster_exemplars]

    detail_level = 0.5
    all_topic_names = [[]]
    object_description = 'social media posts'
    corpus_description = 'a collection of social media posts'
    cluster_layer.make_prompts(
        detail_level,
        all_topic_names,
        object_description,
        corpus_description
    )

    # Generate topic names using LLM
    topic_names_list = cluster_layer.name_topics(
        llm,
        detail_level,
        all_topic_names,
        object_description,
        corpus_description,
        embedding_model=embedding_model
    )
    # topic_names_list = [['Topic A', 'Topic B', 'Topic C', 'Topic D', 'Topic E']] if num_cats == 5 else [['Topic A', 'Topic B', 'Topic C']]  

    target_names = format_pca_axis_label(dim, dim_pca_features,
                                         prefix=axis_prefix)
    logger.info(f"Target names for dimension {dim}: {target_names}")

    if num_cats == 5:
        # Extract labels for positive and negative extremes
        v_pos_label = topic_names_list[0] if len(topic_names_list) > 0 else "Unknown"
        v_neg_label = topic_names_list[1] if len(topic_names_list) > 1 else "Unknown"
        pos_label = topic_names_list[2] if len(topic_names_list) > 2 else "Unknown"
        neg_label = topic_names_list[3] if len(topic_names_list) > 3 else "Unknown"
        neutral_label = topic_names_list[4] if len(topic_names_list) > 4 else "Unknown"

        dim_desc = {
            'very_positive': v_pos_label,
            'very_negative': v_neg_label,
            'positive': pos_label,
            'negative': neg_label,
            'neutral': neutral_label,
            'positive_threshold': float(positive_threshold),
            'negative_threshold': float(negative_threshold),
            'v_positive_threshold': float(v_pos_threshold),
            'v_negative_threshold': float(v_neg_threshold),
        }

        logging.info(f"Dimension {dim} - Very Positive: {v_pos_label}")
        logging.info(f"Dimension {dim} - Positive: {pos_label}")
        logging.info(f"Dimension {dim} - Neutral: {neutral_label}")
        logging.info(f"Dimension {dim} - Negative: {neg_label}")
        logging.info(f"Dimension {dim} - Very Negative: {v_neg_label}")
    elif num_cats == 3:
        # Extract labels for positive and negative extremes
        pos_label = topic_names_list[0] if len(topic_names_list) > 0 else "Unknown"
        neg_label = topic_names_list[1] if len(topic_names_list) > 1 else "Unknown"
        neutral_label = topic_names_list[2] if len(topic_names_list) > 2 else "Unknown"

        dim_desc = {
            'positive': pos_label,
            'negative': neg_label,
            'neutral': neutral_label,
            'positive_threshold': float(positive_threshold),
            'negative_threshold': float(negative_threshold),
        }

        logging.info(f"Dimension {dim} - Positive: {pos_label}")
        logging.info(f"Dimension {dim} - Neutral: {neutral_label}")
        logging.info(f"Dimension {dim} - Negative: {neg_label}")
    else:
        raise ValueError("num_cats must be 3 or 5")

    dim_desc['mean'] = float(mean_val)
    dim_desc['top_features'] = target_names

    return dim_desc

def get_dimension_descriptions(target_df: pl.DataFrame, pca_features, cfg):
    """
    For each dimension, get exemplar documents at the extremes and describe them.

    Args:
        coords: numpy array of shape (n_samples, 2) containing PCA coordinates
        target_df: polars dataframe with columns including 'createtime', 'filter_value', 'coord_2d'
        cfg: hydra config
        percentile: percentile threshold for selecting extreme points (default 95)

    Returns:
        dimension_labels: dict mapping dimension index to dict with 'positive' and 'negative' labels
    """
    text_df = load_text_df(cfg)
    text_df = text_df.with_columns(pl.col('seed').struct.field(cfg.filter_column)).drop('seed')

    exemplar_selection_method = 'random'

    # Compute embeddings for all documents once, before the loop
    logging.info("Computing document embeddings for all documents...")
    embedding_model = VLLMEmbedder('intfloat/multilingual-e5-small')
    # embedding_model = None

    # text_df = text_df.sample(1000000)

    if exemplar_selection_method != 'random':
        # Get unique documents and filter by length
        unique_docs_df = text_df.select(['id', 'Document']) \
            .unique(subset=['Document']) \
            .with_columns(pl.col('Document').str.len_chars().alias('doc_len')) \
            .filter(pl.col('doc_len') < 512) \
            .drop('doc_len')\
            .sample(500000)

        documents_list = unique_docs_df['Document'].to_list()
        document_embeddings = embedding_model.encode(documents_list, show_progress_bar=True, verbose=True)

        # Create embeddings dataframe - store as array type
        embeddings_df = unique_docs_df.with_columns(
            pl.Series('embedding', document_embeddings, dtype=pl.Array(pl.Float32, document_embeddings.shape[1]))
        )

        # Join embeddings back to text_df
        text_df = text_df.join(embeddings_df, on=['id', 'Document'], how='left')

    # Extract dimensions as separate columns for easier filtering
    num_dims = target_df.schema[COORD].shape[0]
    target_df = target_df.with_columns([pl.col(COORD).arr.get(i).alias(f'dim_{i}') for i in range(num_dims)])

    dimension_labels = {}

    # Set up LLM for topic naming
    model_name = 'google/gemma-4-E4B-it'
    llm = AsyncVLLMNamer(
        model=model_name,
        max_model_len=16000,
    )
    # llm = None

    num_cats = 3
    n_exemplars = 32
    n_keyphrases = 32

    for dim in range(num_dims):
        logging.info(f"\nProcessing dimension {dim}...")

        dim_pca_features = pca_features[f'PC{dim+1}']
        dimension_labels[dim] = {}

        dim_desc = get_dimension_description(
            target_df,
            dim,
            dim_pca_features,
            text_df,
            embedding_model,
            llm,
            num_cats=3,
            n_exemplars=n_exemplars,
            n_keyphrases=n_keyphrases,
            exemplar_selection_method=exemplar_selection_method,
            filter_val_col=cfg.filter_column,
            axis_prefix=latent_space.axis_prefix(cfg)
        )
        dimension_labels[dim]['3_cat'] = dim_desc
        dim_desc = get_dimension_description(
            target_df,
            dim,
            dim_pca_features,
            text_df,
            embedding_model,
            llm,
            num_cats=5,
            n_exemplars=n_exemplars,
            n_keyphrases=n_keyphrases,
            exemplar_selection_method=exemplar_selection_method,
            filter_val_col=cfg.filter_column,
            axis_prefix=latent_space.axis_prefix(cfg)
        )
        dimension_labels[dim]['5_cat'] = dim_desc

    return dimension_labels

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    logging.info("Loading data...")

    trend_path = cfg.trend_path
    trend_name = os.path.basename(trend_path.rstrip('/'))
    keywords = None
    dir_name = f"{trend_name}/all"

    target_df, components, targets = latent_space.load(cfg)
    n_dims = target_df.schema[COORD].shape[0]

    # get var(diff(coord)) for each dimension
    coord_diff_var = target_df.sort(['filter_value', 'createtime'])\
        .with_columns([
            pl.col(COORD).arr.get(i).diff().over('filter_value').alias(f'dim_{i}_diff') for i in range(n_dims)
        ])\
        .select([pl.col(f'dim_{i}_diff').var() for i in range(n_dims)])\
        .to_numpy()[0]

    # These features both name the dimension and choose the documents it is
    # named from. Which ranking is right depends on latents.w_ridge: with the
    # loadings unregularised the raw ranking is dominated by targets whose
    # loading is large because the fit barely constrains them, and volume
    # weighting is the repair; once they are shrunk, raw |loading| is already a
    # fair comparison and weighting only pulls every dimension toward the few
    # highest-volume targets, which makes the dimensions look alike.
    weights = (latent_space.target_volumes(cfg, targets)
               if cfg.latents.get('rank_by_volume', True) else None)
    component_features = get_top_component_features(
        components, targets, n_features=100, weights=weights)

    # Get dimension descriptions
    dimension_labels = get_dimension_descriptions(target_df, component_features, cfg)

    for dim in dimension_labels:
        dimension_labels[dim]['variance_of_derivative'] = float(coord_diff_var[dim])

    # Save dimension labels to file
    dim_label_path = latent_space.dimension_labels_path(cfg)
    with open(dim_label_path, 'w') as f:
        json.dump(dimension_labels, f, indent=2)

    logging.info(f"\nResults saved to {dim_label_path}")

    # Beside the labels rather than in them: distinctness is a property of the
    # table, not of any one dimension, and dimension_table keys the labels by
    # dimension index.
    quality = latent_space.ranking_quality(
        components, latent_space.target_volumes(cfg, targets))
    if quality:
        quality_path = os.path.join(
            trend_path, f'{latent_space.name(cfg)}_dimension_quality.json')
        with open(quality_path, 'w') as f:
            json.dump(quality, f, indent=2)
        logging.info(f"Ranking quality saved to {quality_path}")

if __name__ == '__main__':
    main()
