"""Find a trained landscape by the wandb run that produced it.

nn_potential writes run.json beside every checkpoint, naming the wandb run and
its objective, so a sweep trial's model is addressable without re-running it.
This reads those records back and emits the hydra overrides that resolve to the
same directory, which is everything downstream needs -- the eval and plot
scripts already locate a model through `run_dir(cfg)`.

The overrides cover exactly what keys `run_dir`: the latent config, the cell
aggregate, the split, and the landscape fields. Anything else a trial set is
left at the local default on purpose, so a stale key in config.yaml cannot
silently point the lookup at a different directory.

Only trials of the current era are offered: a prior that does not revert, on
the filtered state, trained after the fixes to the training loop. Nothing in a
record separates a fixed run from a broken one, so that last cut is by when the
trial finished. --where overrides any one field of the era and --any-era drops
it entirely.

    python scripts/flows/sweep_runs.py --best --where num_epochs=100
    python scripts/flows/sweep_runs.py --run-id yya9bnc2 --format shell
"""

import argparse
import glob
import json
import os

from latent_gp.latents import LatentConfig

# The trainer's own name for the checkpoint it scored. Older directories predate
# it and hold only per-epoch files, whose mtimes order by epoch.
BEST_STATE = 'model_best.pth'

RECORD = 'run.json'

# What a model has to be to stand behind a number in the paper: a prior that
# does not mean-revert, since the objective scores against a momentum rule that
# cannot predict reversion, and the filtered state, since a forecast cannot read
# observations after the origin.
ERA_WHERE = (('latents.fast_kind', 'wiener'),
             ('latents.slow_kind', 'const|wiener'),
             ('latents.causal_state', 'true'))

# Trials that finished before this predate the fixes to the training loop, and
# their objective is not comparable with a later one. A config value cannot tell
# them apart -- the fixes were to the code -- so the cut is the first sweep to
# run with them in place.
ERA_START = '2026-09-13T10:00:00'

# cfg paths for the LatentConfig fields that are not under `latents`
TOP_LEVEL = {'n_dims': 'n_dims', 'min_target_volume': 'min_target_volume'}

SPLIT_FIELDS = ('holdout_days', 'origin_offset_days', 'train_frac', 'val_frac',
                'seed')


def _cfg_paths():
    """Dotted config paths that, together, determine a run directory."""
    from nn_potential import LANDSCAPE_FIELDS

    paths = ['latents.method', 'latents.causal_state']
    for f in (f.name for f in LatentConfig.__dataclass_fields__.values()):
        paths.append(TOP_LEVEL.get(f, f'latents.{f}'))
    paths += [f'split.{f}' for f in SPLIT_FIELDS]
    paths += [f for f in LANDSCAPE_FIELDS if f not in paths]
    # dict.fromkeys rather than set(): the order is what a reader sees
    return list(dict.fromkeys(paths))


def _get(cfg, path):
    for part in path.split('.'):
        if not isinstance(cfg, dict) or part not in cfg:
            return KeyError
        cfg = cfg[part]
    return cfg


def _fmt(value):
    """A recorded config value as plain text, for display and for --where."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(_fmt(v) for v in value) + ']'
    return str(value)


def _value(value):
    """A recorded value as hydra override text, preserving its type.

    Strings are quoted because the tag hashes a value as it was recorded, and a
    sweep passes phi_hidden_dims as a string: composed back as a list it renders
    differently and keys a directory the run never wrote to.
    """
    if isinstance(value, str):
        return "'{}'".format(value.replace("'", r"\'"))
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(_value(v) for v in value) + ']'
    return _fmt(value)


def records(out_root):
    """Every (directory, record) pair under a trend's output root."""
    for path in sorted(glob.glob(os.path.join(out_root, '*', '*', RECORD))):
        try:
            with open(path) as fh:
                yield os.path.dirname(path), json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue


def has_state(run_path):
    """Whether a directory holds a checkpoint that can be loaded."""
    states = os.path.join(run_path, 'states')
    return os.path.isdir(states) and bool(os.listdir(states))


def find(out_root, run_id=None, sweep_id=None, where=(), require_state=True,
         since=None):
    """Matching records, best objective first.

    `where` is (dotted config path, string value) pairs, compared against the
    recorded config rendered the same way the overrides are, so that a filter
    reads like the override it would produce. A value may offer alternatives,
    `const|wiener`. `since` drops anything that finished before it.
    """
    out = []
    for run_path, rec in records(out_root):
        if run_id is not None and rec.get('run_id') != run_id:
            continue
        if sweep_id is not None and rec.get('sweep_id') != sweep_id:
            continue
        if require_state and not has_state(run_path):
            continue
        if since and (rec.get('finished') or '') < since:
            continue
        cfg = rec.get('config') or {}
        if any(_fmt(_get(cfg, k)) not in v.split('|') for k, v in where):
            continue
        out.append((run_path, rec))
    # an unscored trial sorts last rather than raising
    out.sort(key=lambda r: (r[1].get('objective') is not None,
                            r[1].get('objective') or 0.0), reverse=True)
    return out


def overrides(record):
    """Hydra overrides that reproduce this run's directory.

    A key the record does not carry is skipped: it predates that key, and
    LatentConfig defaults it the same way the trial did.
    """
    cfg = record.get('config') or {}
    out = []
    for path in _cfg_paths():
        value = _get(cfg, path)
        if value is KeyError:
            continue
        out.append(f'{path}={_value(value)}')
    return out


def state_path(run_path):
    """The checkpoint a run should be loaded from, or None.

    The scored model by name where the trainer wrote one. Directories that
    predate it fall back to the newest file, which is the epoch that tripped
    early stopping rather than the best one -- the trainer checkpoints every
    improvement and then the halting epoch too. Retrain rather than trust it
    for anything the numbers rest on.
    """
    states = os.path.join(run_path, 'states')
    best = os.path.join(states, BEST_STATE)
    if os.path.exists(best):
        return best
    files = [os.path.join(states, f) for f in os.listdir(states)] \
        if os.path.isdir(states) else []
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-root', default='./out/noun_phrase_bkrr_trends',
                    help="a trend's output root, holding one directory per fit")
    ap.add_argument('--run-id', help='wandb run id')
    ap.add_argument('--sweep-id', help='wandb sweep id')
    ap.add_argument('--where', action='append', default=[], metavar='PATH=VALUE',
                    help='require a recorded config value, e.g. num_epochs=100; '
                         'alternatives as slow_kind=const|wiener. Overrides the '
                         'era default for that path.')
    ap.add_argument('--any-era', action='store_true',
                    help='also offer trials that predate the training fixes, '
                         'or that revert, or that read the smoothed state')
    ap.add_argument('--best', action='store_true',
                    help='emit only the highest-objective match')
    ap.add_argument('--any-state', action='store_true',
                    help='include runs whose checkpoints did not survive')
    ap.add_argument('--format', choices=('table', 'overrides', 'shell'),
                    default='table')
    ap.add_argument('--name', default='BEST_OVERRIDES',
                    help='array name for --format shell')
    args = ap.parse_args()

    where = {} if args.any_era else dict(ERA_WHERE)
    for w in args.where:
        if '=' not in w:
            ap.error(f'--where needs PATH=VALUE, got {w!r}')
        k, v = w.split('=', 1)
        where[k] = v

    hits = find(args.out_root, run_id=args.run_id, sweep_id=args.sweep_id,
                where=where.items(), require_state=not args.any_state,
                since=None if args.any_era else ERA_START)
    if not hits:
        raise SystemExit('no run record matched'
                         + ('' if args.any_era else '; --any-era to look past '
                            f'the era beginning {ERA_START}'))
    if args.best or args.format != 'table':
        hits = hits[:1]

    if args.format == 'table':
        print(f'{"objective":>10}  {"run":10} {"sweep":10} {"state":14} dir')
        for run_path, rec in hits:
            obj = rec.get('objective')
            state = state_path(run_path)
            print(f'{obj if obj is not None else float("nan"):>10.5f}  '
                  f'{rec.get("run_id") or "-":10} {rec.get("sweep_id") or "-":10} '
                  f'{os.path.basename(state) if state else "-":14} {run_path}')
        return

    run_path, rec = hits[0]
    ovr = overrides(rec)
    if args.format == 'overrides':
        print('\n'.join(ovr))
        return

    print(f'# {rec.get("run_id")} of sweep {rec.get("sweep_id")}, '
          f'objective {rec.get("objective")}, finished {rec.get("finished")}')
    print(f'# model: {state_path(run_path)}')
    print(f'{args.name}=(')
    for o in ovr:
        print(f'  "{o}"')
    print(')')


if __name__ == '__main__':
    main()
