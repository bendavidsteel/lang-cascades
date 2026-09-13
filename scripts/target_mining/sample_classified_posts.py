"""Sample stance-classified posts for manual inspection.

Draws a sample that is stratified across platform x year x political party (or
actor type, for non-politicians), and within each stratum spreads the draw over
as many distinct accounts (influencers, politicians) as possible.

The output file gives, per post, a URL that opens the post in a browser, plus a
random sample of the post's extracted stance targets, the stance classification
of each of those targets, and the classifier's probability for each class.

Pass --include with an earlier sample (or the coded CSV exported by the coding
page) to grow a sample rather than redraw one: every pair already in that file is
kept, so only the newly added pairs need coding.

Examples:
    python scripts/target_mining/sample_classified_posts.py -n 100 \
        --output ./out/classified_post_sample.csv

    python scripts/target_mining/sample_classified_posts.py -n 400 \
        --include ./out/stance_coding_coder_2026-07-29.csv \
        --output ./out/classified_post_sample_400.csv
"""

import argparse
import datetime
import os
import random
import re
import sys

import polars as pl

STANCE_FILE_RE = re.compile(r'^(\d{4})_(\d{1,2})_doc_targets_with_stance\.parquet\.zstd$')

STANCE_FILE_SUFFIX = '_doc_targets_with_stance.parquet.zstd'
PROBS_FILE_SUFFIX = '_doc_targets_stance_probs.parquet.zstd'

# the classification head's class order, from stancemining's STANCE_LABELS_2_ID
PROB_LABELS = ['neutral', 'favor', 'against']
PROB_COLUMNS = [f'model_prob_{label}' for label in PROB_LABELS]

DEFAULT_STANCE_DIR = './data/stance_targets/noun_phrase_stance'
DEFAULT_RAW_DIR = '../sitrep/data/digital_trace/raw_platforms'

# a handful of federal party names are spelled inconsistently in the seed list
PARTY_ALIASES = {
    'Conservative Part of Canada': 'Conservative',
    'Conservative Party of Canada': 'Conservative',
    'Liberal Party of Canada': 'Liberal',
    'New Democratic Party': 'NDP',
}

# instagram shortcodes are the media pk written in this base-64 alphabet
IG_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'

# raw fields each platform's classified document can be built from, for text_source.
# Twitter is not listed: a tweet is always the author's own writing.
RAW_TEXT_COLUMNS = {
    'tiktok': ['video_id', 'desc'],
    'instagram': ['id', 'caption', 'description'],
    'bluesky': ['id', 'text', 'original_post_text', 'embed_title', 'embed_description'],
}

# federal and provincial parties are collapsed into families for stratification, so
# that we don't end up with one stratum per provincial party. The exact party name
# is kept in the output.
PARTY_FAMILIES = [
    ('NDP', r'ndp|new democrat'),
    ('Bloc Québécois', r'bloc'),
    ('Conservative', r'conservative|saskatchewan party|united conservative'),
    ('Liberal', r'liberal|libéral'),
    ('Green', r'green|parti vert'),
    ('PPC', r'ppc|people\'s party'),
    ('Independent', r'independent|indépendant'),
]


def get_stance_files(stance_dir):
    files = []
    for name in sorted(os.listdir(stance_dir)):
        if STANCE_FILE_RE.match(name):
            files.append(os.path.join(stance_dir, name))
    if not files:
        raise ValueError(f"No stance files found in {stance_dir}")
    return files


def _clean(column):
    return pl.col(column).fill_null('').str.strip_chars()


def _party_expr():
    """Party of the account, or null for accounts with no party affiliation."""
    return pl.when(_clean('Party') != '').then(_clean('Party'))\
        .when(_clean('ProvincialParty') != '').then(_clean('ProvincialParty'))\
        .when(_clean('FederalParty') != '').then(_clean('FederalParty'))\
        .otherwise(None)\
        .replace(PARTY_ALIASES)


def _party_family_expr():
    """Collapse the party into a family, so provincial wings share a stratum."""
    party = _party_expr()
    lowered = party.str.to_lowercase()
    family = pl.when(party.is_null()).then(None)
    for name, pattern in PARTY_FAMILIES:
        family = family.when(lowered.str.contains(pattern)).then(pl.lit(name))
    return family.when(party.is_not_null()).then(pl.lit('Other party')).otherwise(None)


def _actor_group_expr():
    """Stratification group: party family for politicians, actor type for everyone else."""
    actor_type = pl.when(_clean('MainType') != '').then(_clean('MainType')).otherwise(pl.lit('unknown'))
    family = _party_family_expr()
    return pl.when(family.is_not_null())\
        .then(pl.format('party:{}', family))\
        .otherwise(pl.format('type:{}', actor_type))


def load_candidates(stance_dir):
    """Read the metadata of every post that has at least one classified target."""
    files = get_stance_files(stance_dir)
    frames = []
    for i, path in enumerate(files):
        print(f"Reading {i + 1}/{len(files)}: {os.path.basename(path)}", end='\r', file=sys.stderr)
        df = pl.read_parquet(path, columns=['id', 'platform', 'createtime', 'seed', 'Stances'])
        df = df.unique(['id', 'platform'])
        df = df.with_columns(pl.col('Stances').list.drop_nulls().list.len().alias('n_classified'))
        df = df.filter(pl.col('n_classified') > 0)
        if df.is_empty():
            continue
        df = df.with_columns(pl.col('seed').struct.unnest())
        frames.append(df.select([
            'id',
            'platform',
            pl.col('createtime').dt.year().alias('year'),
            _clean('SeedName').alias('seed_name'),
            _actor_group_expr().alias('actor_group'),
            pl.lit(path).alias('source_file'),
        ]))
    print(file=sys.stderr)
    return pl.concat(frames, how='vertical')


def load_previous(paths):
    """Read pairs already sampled (or coded), so a bigger sample can contain them.

    Accepts either a sample CSV from this script or a coded CSV from the coding
    page -- both carry platform / post_id / target.
    """
    posts, pairs = set(), set()
    for path in paths:
        df = pl.read_csv(path) if path.endswith('.csv') else pl.read_parquet(path)
        missing = {'platform', 'post_id', 'target'} - set(df.columns)
        if missing:
            raise SystemExit(f'{path} is missing columns: {sorted(missing)}')
        for row in df.select(['platform', 'post_id', 'target']).iter_rows():
            platform, post_id, target = row
            posts.add((platform, post_id))
            pairs.add((platform, post_id, target))
    return posts, pairs


def _mark_keep(df, keys, columns, name='must_keep'):
    """Flag the rows of `df` whose `columns` tuple appears in `keys`."""
    if not keys:
        return df.with_columns(pl.lit(False).alias(name))
    keep_df = pl.DataFrame(
        {c: [k[i] for k in sorted(keys)] for i, c in enumerate(columns)},
        schema={c: pl.String for c in columns},
    ).with_columns(pl.lit(True).alias(name))
    return df.join(keep_df, on=columns, how='left').with_columns(
        pl.col(name).fill_null(False))


def allocate_quotas(capacities, total, seed, floors=None):
    """Split `total` draws as evenly as possible over cells of limited capacity.

    `floors` reserves a minimum per cell, used to keep an earlier sample inside
    the new one.
    """
    rng = random.Random(seed)
    floors = floors or {}
    quotas = {cell: min(floors.get(cell, 0), capacities[cell]) for cell in capacities}
    remaining = total - sum(quotas.values())
    if remaining < 0:
        print(f"Warning: the sample being carried over already covers "
              f"{sum(quotas.values())} posts, more than the {total} requested; "
              f"keeping all of them", file=sys.stderr)
        remaining = 0
    while remaining > 0:
        open_cells = sorted(cell for cell in capacities if quotas[cell] < capacities[cell])
        if not open_cells:
            break
        # the final round can't reach every cell, so don't let cell names decide who misses out
        rng.shuffle(open_cells)
        share = max(1, remaining // len(open_cells))
        for cell in open_cells:
            if remaining == 0:
                break
            take = min(share, capacities[cell] - quotas[cell], remaining)
            quotas[cell] += take
            remaining -= take
    return quotas


def choose_sample(candidate_df, total, seed, keep_posts=None):
    """Pick a stratified sample, spread over as many distinct accounts as possible.

    Posts in `keep_posts` are pulled to the front of their stratum and reserved a
    quota, so the result is a superset of an earlier sample.
    """
    df = candidate_df.with_columns(
        pl.concat_str([
            pl.col('platform'),
            pl.col('year').cast(pl.String),
            pl.col('actor_group'),
        ], separator=' | ').alias('cell')
    )

    # shuffle once, then order each cell so that we take one post per account
    # before taking a second post from any account
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    df = df.with_columns(pl.int_range(pl.len()).alias('shuffle_rank'))
    df = df.with_columns(pl.int_range(pl.len()).over(['cell', 'seed_name']).alias('account_rank'))
    df = _mark_keep(df, keep_posts, ['platform', 'id'])
    df = df.sort(['must_keep', 'account_rank', 'shuffle_rank'],
                 descending=[True, False, False])
    df = df.with_columns(pl.int_range(pl.len()).over('cell').alias('cell_rank'))

    capacities = {row['cell']: row['len'] for row in df.group_by('cell').len().to_dicts()}
    floors = {row['cell']: row['len'] for row in
              df.filter(pl.col('must_keep')).group_by('cell').len().to_dicts()}
    quotas = allocate_quotas(capacities, total, seed, floors)
    quota_df = pl.DataFrame(
        {'cell': list(quotas.keys()), 'quota': list(quotas.values())},
        schema={'cell': pl.String, 'quota': pl.Int64},
    )

    sample_df = df.join(quota_df, on='cell', how='inner').filter(pl.col('cell_rank') < pl.col('quota'))
    if total > sample_df.height:
        print(
            f"Warning: asked for {total} posts but only {sample_df.height} available "
            f"across {len(capacities)} strata",
            file=sys.stderr,
        )
    return sample_df.select(['id', 'platform', 'cell', 'source_file', 'must_keep'])


def load_sampled_posts(sample_df):
    """Fetch the full record of each sampled post from its stance file."""
    frames = []
    for source_file, keys in sample_df.group_by('source_file'):
        source_file = source_file[0]
        df = pl.read_parquet(source_file)
        df = df.unique(['id', 'platform'])
        df = df.join(keys.select(['id', 'platform', 'cell']), on=['id', 'platform'], how='inner')
        frames.append(df.with_columns(pl.col('seed').struct.unnest()).select([
            'id',
            'platform',
            'cell',
            'createtime',
            pl.col('createtime').dt.year().alias('year'),
            'Targets',
            'Stances',
            'Document',
            'ParentDocument',
            _clean('SeedName').alias('seed_name'),
            _clean('Handle').alias('handle'),
            _clean('MainType').alias('main_type'),
            _clean('SubType').alias('sub_type'),
            _clean('Province').alias('province'),
            _clean('ElectoralDistrict').alias('electoral_district'),
            _party_expr().alias('party'),
            _party_family_expr().alias('party_family'),
            _actor_group_expr().alias('actor_group'),
        ]))
    return pl.concat(frames, how='vertical')


def instagram_shortcode(post_id):
    """Instagram ids embed the media pk, which encodes to the shortcode in a post URL."""
    pk = post_id.split('|')[-1] if '|' in post_id else post_id.split('_')[0]
    try:
        pk = int(pk)
    except ValueError:
        return None
    if pk <= 0:
        return None
    shortcode = ''
    while pk > 0:
        shortcode = IG_ALPHABET[pk % 64] + shortcode
        pk //= 64
    return shortcode


def bluesky_parts(post_id):
    """The did and cid of a bluesky post; the crawl has written them in either order."""
    parts = [part for part in post_id.split('_') if part]
    did = next((part for part in parts if part.startswith('did:')), None)
    cid = next((part for part in parts if part.startswith('baf')), None)
    return did, cid


def resolve_bluesky_uris(post_df, raw_dirs):
    """Look up the at:// uri of each bluesky post in the raw crawl files.

    The stance data keeps the post cid, but a bsky.app URL needs the record key.
    Only some vintages of the crawl carry the uri, so each given dir is searched.
    """
    bluesky_df = post_df.filter(pl.col('platform') == 'bluesky')
    if bluesky_df.is_empty():
        return {}

    id_by_cid = {}
    for post_id in bluesky_df['id'].to_list():
        _, cid = bluesky_parts(post_id)
        if cid:
            id_by_cid[cid] = post_id

    uri_by_id = {}
    dates = sorted(_raw_dates(bluesky_df, 'bluesky'))
    for raw_dir in raw_dirs:
        if len(uri_by_id) == len(id_by_cid):
            break
        if not os.path.isdir(raw_dir):
            print(f"Warning: raw platform dir {raw_dir} not found, bluesky posts may fall "
                  f"back to profile URLs", file=sys.stderr)
            continue
        for date in dates:
            path = os.path.join(raw_dir, f'bluesky_{date.isoformat()}.parquet.zstd')
            if not os.path.exists(path):
                continue
            columns = pl.scan_parquet(path).collect_schema().names()
            if 'uri' not in columns:
                continue
            key = 'cid' if 'cid' in columns else 'id'
            for raw_key, uri in pl.read_parquet(path, columns=[key, 'uri']).iter_rows():
                _, cid = bluesky_parts(raw_key or '')
                post_id = id_by_cid.get(cid or raw_key)
                if post_id and uri:
                    uri_by_id.setdefault(post_id, uri)
            if len(uri_by_id) == len(id_by_cid):
                break

    missing = len(id_by_cid) - len(uri_by_id)
    if missing:
        print(f"Warning: could not resolve {missing}/{len(id_by_cid)} bluesky post URLs",
              file=sys.stderr)
    return uri_by_id


def _raw_dates(post_df, platform):
    """Days whose raw file could hold these posts, crawl date lagging the post date."""
    dates = set()
    posts = post_df.filter(pl.col('platform') == platform)
    for post_date in posts.select(pl.col('createtime').dt.date())['createtime'].to_list():
        for offset in (-1, 0, 1):
            dates.add(post_date + datetime.timedelta(days=offset))
    return sorted(dates)


def _read_raw(raw_dir, platform, dates, columns, wanted):
    """Raw records for the wanted ids, gathered over the candidate days.

    Raw schemas drift across the crawl, so only the columns a file actually has are
    read and the rest come back null.
    """
    frames = []
    found = set()
    id_column = columns[0]
    for date in dates:
        path = os.path.join(raw_dir, f'{platform}_{date.isoformat()}.parquet.zstd')
        if not os.path.exists(path):
            continue
        available = pl.scan_parquet(path).collect_schema().names()
        take = [c for c in columns if c in available]
        if id_column not in take:
            continue
        df = pl.read_parquet(path, columns=take)\
            .with_columns(pl.col(id_column).cast(pl.String).alias('id'))\
            .filter(pl.col('id').is_in(wanted) & ~pl.col('id').is_in(found))\
            .unique('id')
        if df.is_empty():
            continue
        found.update(df['id'].to_list())
        frames.append(df.select(['id'] + [c for c in take if c != id_column]))
        if len(found) == len(wanted):
            break
    if not frames:
        return None
    return pl.concat(frames, how='diagonal')


def _concat_source(document, caption, appended):
    """A caption with machine-read text appended: which of the two the model saw.

    Truncation in get_stance only ever shortens the document, so a document that is
    a prefix of the caption still came from the caption alone.
    """
    if not document:
        return 'unknown'
    if not caption:
        return appended
    if document == caption or caption.startswith(document):
        return 'caption'
    if document.startswith(caption):
        return f'caption+{appended}'
    return 'unknown'


def _coalesced_source(document, options):
    """Fields the document is one of, rather than a concatenation of."""
    if not document:
        return 'unknown'
    for label, value in options:
        if value and (document == value or value.startswith(document)):
            return label
    return 'unknown'


def _clean_text(value):
    return (value or '').strip()


def resolve_text_sources(post_df, raw_dir):
    """Which raw fields each classified document was assembled from.

    A tiktok document is the caption then whatever was said in the video, an
    instagram one the caption then whatever was read off the image, and a bluesky
    one the post's own text or, failing that, the text it reposted or the card it
    linked. Only the raw record says which of those a given document is, and the
    evaluation needs it to report transcript-derived posts as their own subgroup.
    """
    sources = {}
    for post_id, platform in post_df.filter(pl.col('platform') == 'twitter')\
            .select(['id', 'platform']).iter_rows():
        sources[(platform, post_id)] = 'written'

    remaining = post_df.filter(pl.col('platform').is_in(list(RAW_TEXT_COLUMNS)))
    if remaining.is_empty():
        return sources
    if not os.path.isdir(raw_dir):
        print(f"Warning: raw platform dir {raw_dir} not found, text_source will be "
              f"unknown for {remaining.height} posts", file=sys.stderr)
        return sources

    for platform, columns in RAW_TEXT_COLUMNS.items():
        posts = remaining.filter(pl.col('platform') == platform)
        if posts.is_empty():
            continue
        wanted = set(posts['id'].to_list())
        raw_df = _read_raw(raw_dir, platform, _raw_dates(post_df, platform),
                           columns, wanted)
        raw = {} if raw_df is None else {r['id']: r for r in raw_df.to_dicts()}
        for post_id, document in posts.select(['id', 'Document']).iter_rows():
            record = raw.get(post_id)
            document = _clean_text(document)
            if record is None:
                sources[(platform, post_id)] = 'unknown'
            elif platform == 'tiktok':
                sources[(platform, post_id)] = _concat_source(
                    document, _clean_text(record.get('desc')), 'transcript')
            elif platform == 'instagram':
                caption = record.get('caption')
                caption = caption.get('text') if isinstance(caption, dict) else caption
                sources[(platform, post_id)] = _concat_source(
                    document, _clean_text(caption or record.get('description')), 'ocr')
            else:
                card = ' '.join(part for part in
                                (_clean_text(record.get('embed_title')),
                                 _clean_text(record.get('embed_description'))) if part)
                sources[(platform, post_id)] = _coalesced_source(document, [
                    ('written', _clean_text(record.get('text'))),
                    ('repost text', _clean_text(record.get('original_post_text'))),
                    ('link card', card.strip()),
                ])
        unresolved = sum(1 for post_id in wanted
                         if sources.get((platform, post_id), 'unknown') == 'unknown')
        if unresolved:
            print(f"Warning: could not tell what {unresolved}/{len(wanted)} {platform} "
                  f"documents were built from", file=sys.stderr)
    return sources


def add_text_sources(post_df, raw_dir):
    sources = resolve_text_sources(post_df, raw_dir)
    return post_df.with_columns(
        pl.struct(['platform', 'id'])
        .map_elements(lambda r: sources.get((r['platform'], r['id']), 'unknown'),
                      return_dtype=pl.String)
        .alias('text_source'))


def add_post_urls(post_df, raw_dirs):
    """Add a browsable URL per post, plus whether it points at the post or the account."""
    uri_by_id = resolve_bluesky_uris(post_df, raw_dirs)

    urls = []
    kinds = []
    for post_id, platform, handle in post_df.select(['id', 'platform', 'handle']).iter_rows():
        url, kind = None, 'none'
        if platform == 'twitter':
            # x.com ignores the handle in a status URL, so a stale handle still resolves
            url = f"https://x.com/{handle or 'i'}/status/{post_id}"
            kind = 'post'
        elif platform == 'tiktok':
            if handle:
                url = f"https://www.tiktok.com/@{handle}/video/{post_id}"
                kind = 'post'
        elif platform == 'instagram':
            shortcode = instagram_shortcode(post_id)
            if shortcode:
                url = f"https://www.instagram.com/p/{shortcode}/"
                kind = 'post'
            elif handle:
                url = f"https://www.instagram.com/{handle}/"
                kind = 'profile'
        elif platform == 'bluesky':
            uri = uri_by_id.get(post_id)
            did, _ = bluesky_parts(post_id)
            if uri and uri.startswith('at://'):
                uri_did, _, rkey = uri[len('at://'):].partition('/app.bsky.feed.post/')
                if rkey:
                    url = f"https://bsky.app/profile/{uri_did}/post/{rkey}"
                    kind = 'post'
            if url is None and did:
                url = f"https://bsky.app/profile/{did}"
                kind = 'profile'
        urls.append(url)
        kinds.append(kind)

    return post_df.with_columns([
        pl.Series('post_url', urls, dtype=pl.String),
        pl.Series('url_kind', kinds, dtype=pl.String),
    ])


OUTPUT_COLUMNS = [
    'post_url', 'url_kind', 'platform', 'createtime', 'year', 'seed_name', 'handle',
    'main_type', 'sub_type', 'party', 'party_family', 'actor_group', 'province',
    'electoral_district', 'text_source', 'n_post_targets', 'n_sampled_targets',
    'pair_weight',
]


def sample_targets(post_df, max_targets, seed, keep_pairs=None):
    """Explode to one row per (post, target) pair, keeping at most `max_targets` per post.

    Pairs in `keep_pairs` are always kept, even where that exceeds `max_targets`,
    so no already-coded pair is dropped from a re-drawn sample.
    """
    long_df = post_df.explode(['Targets', 'Stances'])\
        .rename({'Targets': 'target', 'Stances': 'stance'})\
        .drop_nulls(['target', 'stance'])
    long_df = long_df.drop('must_keep', strict=False)
    long_df = _mark_keep(long_df, keep_pairs, ['platform', 'id', 'target'])

    # capping targets per post makes this a two-stage sample: a pair in a post with
    # many classified targets is less likely to be drawn than one in a post with
    # few. Record both counts so pair-level estimates can be weighted back up.
    long_df = long_df.with_columns(
        pl.len().over(['id', 'platform']).alias('n_post_targets'))

    if max_targets > 0:
        long_df = long_df.sample(fraction=1.0, shuffle=True, seed=seed)
        long_df = long_df.sort('must_keep', descending=True, maintain_order=True)
        long_df = long_df.with_columns(pl.int_range(pl.len()).over(['id', 'platform']).alias('target_rank'))
        long_df = long_df.filter(
            (pl.col('target_rank') < max_targets) | pl.col('must_keep')).drop('target_rank')

    return long_df.with_columns(
        pl.len().over(['id', 'platform']).alias('n_sampled_targets')
    ).with_columns(
        (pl.col('n_post_targets') / pl.col('n_sampled_targets')).alias('pair_weight')
    )


def add_class_probs(long_df, probs_dir, stance_files):
    """Join the classifier's per-class probabilities onto each sampled pair.

    The probabilities are written to their own weekly files, keyed by (id, platform)
    with one probability vector per extracted target.
    """
    frames = []
    for stance_file in sorted(stance_files):
        name = os.path.basename(stance_file).replace(STANCE_FILE_SUFFIX, PROBS_FILE_SUFFIX)
        path = os.path.join(probs_dir, name)
        if not os.path.exists(path):
            continue
        week = pl.read_parquet(path, columns=['id', 'platform', 'Targets', 'Probs'])\
            .explode(['Targets', 'Probs'])\
            .rename({'Targets': 'target'})\
            .drop_nulls(['target', 'Probs'])\
            .join(long_df.select(['id', 'platform', 'target']),
                  on=['id', 'platform', 'target'], how='semi')
        if week.height:
            frames.append(week)

    if not frames:
        print(f"Warning: no probabilities in {probs_dir} match the sample", file=sys.stderr)
        return long_df

    probs = pl.concat(frames).unique(['id', 'platform', 'target'])
    probs = probs.with_columns(
        [pl.col('Probs').list.get(i).alias(column)
         for i, column in enumerate(PROB_COLUMNS)]).drop('Probs')
    long_df = long_df.join(probs, on=['id', 'platform', 'target'], how='left')

    missing = long_df[PROB_COLUMNS[0]].null_count()
    if missing:
        print(f"Warning: {missing}/{long_df.height} sampled pairs have no class "
              f"probabilities", file=sys.stderr)
    return long_df


def check_carried_over(long_df, keep_pairs):
    """Confirm every already-coded pair made it into the new sample."""
    if not keep_pairs:
        return
    got = set(zip(long_df['platform'].to_list(), long_df['id'].to_list(),
                  long_df['target'].to_list()))
    missing = sorted(keep_pairs - got)
    kept = len(keep_pairs) - len(missing)
    print(f"\ncarried over {kept}/{len(keep_pairs)} previously sampled pairs, "
          f"{long_df.height - kept} new pairs to code")
    if missing:
        print(f"Warning: {len(missing)} previously sampled pairs are NOT in the new "
              f"sample -- their post or target is no longer in the stance data:",
              file=sys.stderr)
        for pair in missing[:5]:
            print(f"  {pair[0]} {pair[1]} :: {pair[2]}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more", file=sys.stderr)


def format_output(long_df, layout):
    """One row per (post, target) pair, or one row per post with the targets collapsed."""
    prob_columns = [c for c in PROB_COLUMNS if c in long_df.columns]
    if layout == 'long':
        return long_df.select(
            OUTPUT_COLUMNS
            + ['target', 'stance'] + prob_columns
            + [pl.col('Document').alias('post_text'), pl.col('ParentDocument').alias('parent_text'),
               pl.col('id').alias('post_id')]
        ).sort(['platform', 'createtime', 'post_id', 'target'])

    wide_df = long_df.group_by(['id', 'platform']).agg(
        [pl.col(c).first() for c in OUTPUT_COLUMNS if c != 'platform']
        + [
            pl.col('target').alias('targets'),
            pl.col('stance').alias('stances'),
            pl.format('{}: {}', pl.col('target'), pl.col('stance')).alias('target_stances'),
            pl.col('Document').first().alias('post_text'),
            pl.col('ParentDocument').first().alias('parent_text'),
        ]
    )
    return wide_df.with_columns([
        pl.col('targets').list.join(' | '),
        pl.col('stances').list.join(' | '),
        pl.col('target_stances').list.join(' | '),
    ]).select(
        OUTPUT_COLUMNS
        + ['target_stances', 'targets', 'stances', 'post_text', 'parent_text',
           pl.col('id').alias('post_id')]
    ).sort(['platform', 'createtime', 'post_id'])


def print_summary(post_df, long_df, output_df):
    print(f"\nSampled {post_df.height} posts, {long_df.height} (post, target) pairs, "
          f"{output_df.height} output rows, {post_df['seed_name'].n_unique()} distinct accounts, "
          f"{post_df['cell'].n_unique()} strata")

    targets_per_post = long_df.group_by(['id', 'platform']).len()
    print("\ntargets sampled per post:")
    for row in targets_per_post.group_by('len').len(name='posts').sort('len').to_dicts():
        print(f"  {row['len']}: {row['posts']}")
    print("\nstances:")
    for row in long_df.group_by('stance').len().sort('len', descending=True).to_dicts():
        print(f"  {row['stance']}: {row['len']}")

    # highly variable weights make pair-level estimates noisy, so show them
    weights = long_df['pair_weight']
    total = weights.sum()
    effective = total ** 2 / (weights ** 2).sum()
    print(f"\npair weights (classified targets per post / targets sampled): "
          f"min {weights.min():.2f}, median {weights.median():.2f}, "
          f"mean {weights.mean():.2f}, max {weights.max():.2f}")
    print(f"  weighted pair-level estimates have an effective n of "
          f"{effective:.0f} of {long_df.height} coded pairs")
    classified = long_df.group_by(['id', 'platform']).agg(pl.col('n_post_targets').first())
    print("  classified targets per sampled post:")
    for row in classified.group_by('n_post_targets').len(name='posts').sort('n_post_targets').to_dicts():
        print(f"    {row['n_post_targets']}: {row['posts']}")

    for column in ['platform', 'year', 'actor_group', 'url_kind', 'text_source']:
        counts = post_df.group_by(column).len().sort(column)
        print(f"\nposts per {column}:")
        for row in counts.to_dicts():
            print(f"  {row[column]}: {row['len']}")


def selftest():
    """Check the text_source rules, which the raw data this runs against is needed to
    exercise otherwise."""
    # tiktok and instagram append machine-read text to the caption
    assert _concat_source('a caption', 'a caption', 'transcript') == 'caption'
    assert _concat_source('a caption\nwhat was said', 'a caption', 'transcript') \
        == 'caption+transcript'
    assert _concat_source('what was said', '', 'transcript') == 'transcript'
    assert _concat_source('a cap', 'a caption', 'transcript') == 'caption'  # truncated
    assert _concat_source('something else', 'a caption', 'transcript') == 'unknown'
    assert _concat_source('', 'a caption', 'transcript') == 'unknown'
    assert _concat_source('a caption\nread off the image', 'a caption', 'ocr') \
        == 'caption+ocr'

    # bluesky documents stand in for one field or another
    options = [('written', 'my own words'), ('repost text', 'what they said'),
               ('link card', 'Headline A story')]
    assert _coalesced_source('my own words', options) == 'written'
    assert _coalesced_source('what they said', options) == 'repost text'
    assert _coalesced_source('Headline A story', options) == 'link card'
    assert _coalesced_source('my own', options) == 'written'          # truncated
    assert _coalesced_source('none of them', options) == 'unknown'
    assert _coalesced_source('what they said', [('written', ''),
                                                ('repost text', 'what they said')]) \
        == 'repost text'

    # both vintages of the bluesky id, and the ids of the platforms that have none
    assert bluesky_parts('did:plc:abc__bafyreidef') == ('did:plc:abc', 'bafyreidef')
    assert bluesky_parts('bafyreidef_did:plc:abc_repost') == ('did:plc:abc', 'bafyreidef')
    assert bluesky_parts('1897407239302078491') == (None, None)

    # a tweet needs no raw lookup, and a missing raw dir leaves the rest unknown
    post_df = pl.DataFrame({
        'id': ['1', '2'],
        'platform': ['twitter', 'tiktok'],
        'createtime': [datetime.datetime(2025, 1, 1), datetime.datetime(2025, 1, 1)],
        'Document': ['a tweet', 'a caption'],
    })
    out_df = add_text_sources(post_df, '/nonexistent-raw-dir')
    assert out_df.sort('id')['text_source'].to_list() == ['written', 'unknown']
    print('selftest: text_source rules pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', '--num-posts', type=int, default=100, help='number of posts to sample')
    parser.add_argument('-t', '--max-targets-per-post', type=int, default=2,
                        help='sample at most this many stance targets per post, 0 for all of them')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--include', action='append', default=[], metavar='CSV',
                        help='an earlier sample or coded CSV to keep inside this one, '
                             'so only the newly added pairs still need coding '
                             '(repeatable)')
    parser.add_argument('--stance-dir', default=DEFAULT_STANCE_DIR,
                        help='dir of *_doc_targets_with_stance.parquet.zstd files')
    parser.add_argument('--probs-dir',
                        help='dir of *_doc_targets_stance_probs.parquet.zstd files, whose '
                             'per-class probabilities are joined onto each sampled pair '
                             '(default: the stance dir with a _probs suffix, if it exists)')
    parser.add_argument('--raw-dir', default=DEFAULT_RAW_DIR,
                        help='dir of raw daily platform files, used to recover bluesky '
                             'post URLs and what each document was built from')
    parser.add_argument('--uri-dir', action='append', default=[], metavar='DIR',
                        help='another dir of raw daily files to search for bluesky at:// '
                             'uris, for vintages of the crawl whose current files no longer '
                             'carry them (repeatable)')
    parser.add_argument('--selftest', action='store_true',
                        help='verify the text_source rules and exit')
    parser.add_argument('--format', choices=['long', 'wide'], default='long',
                        help='one row per (post, target) pair, or one row per post')
    parser.add_argument('--output', default='./out/classified_post_sample.csv',
                        help='output path, .csv or .parquet')
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    keep_posts, keep_pairs = load_previous(args.include)
    if args.include:
        print(f"carrying over {len(keep_pairs)} pairs across {len(keep_posts)} posts "
              f"from {', '.join(args.include)}", file=sys.stderr)

    candidate_df = load_candidates(args.stance_dir)
    print(f"{candidate_df.height} classified posts available", file=sys.stderr)

    sample_df = choose_sample(candidate_df, args.num_posts, args.seed, keep_posts)
    post_df = load_sampled_posts(sample_df)
    post_df = add_post_urls(post_df, [args.raw_dir] + args.uri_dir)
    post_df = add_text_sources(post_df, args.raw_dir)
    long_df = sample_targets(post_df, args.max_targets_per_post, args.seed, keep_pairs)

    probs_dir = args.probs_dir or f'{args.stance_dir.rstrip("/")}_probs'
    if os.path.isdir(probs_dir):
        long_df = add_class_probs(long_df, probs_dir, sample_df['source_file'].unique().to_list())
    elif args.probs_dir:
        print(f"Warning: probs dir {probs_dir} not found", file=sys.stderr)

    output_df = format_output(long_df, args.format)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.output.endswith('.csv'):
        output_df.write_csv(args.output)
    else:
        output_df.write_parquet(args.output, compression='zstd')

    print_summary(post_df, long_df, output_df)
    check_carried_over(long_df, keep_pairs)
    print(f"\nWrote {args.output}")


if __name__ == '__main__':
    main()
