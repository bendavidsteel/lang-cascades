import json
import os

from hydra.core.override_parser.overrides_parser import OverridesParser

import sweep_runs


def write_run(tmp_path, run_id, objective, config, states=(), best=False,
              finished='2026-09-14T00:00:00'):
    """A run directory shaped the way nn_potential writes one."""
    d = tmp_path / f'gpfa{run_id}_data_split' / f'landscape{run_id}'
    (d / 'states').mkdir(parents=True)
    for name in states:
        (d / 'states' / name).write_bytes(b'')
    if best:
        (d / 'states' / sweep_runs.BEST_STATE).write_bytes(b'')
    (d / 'run.json').write_text(json.dumps(
        {'run_id': run_id, 'sweep_id': 'swp', 'objective': objective,
         'finished': finished, 'config': config}))
    return d


def parsed(override):
    """What hydra makes of one override string."""
    return OverridesParser.create().parse_overrides([override])[0].value()


def test_a_recorded_string_stays_a_string_through_an_override():
    # a sweep passes phi_hidden_dims as a string; composed back as a list it
    # hashes differently and keys a directory the run never wrote to
    assert parsed(f'phi_hidden_dims={sweep_runs._value("[128,128,128,128]")}') \
        == '[128,128,128,128]'
    assert parsed(f'phi_hidden_dims={sweep_runs._value([128, 128])}') == [128, 128]


def test_recorded_scalars_survive_the_round_trip():
    for value in (None, True, False, 0.0015263722849730675, 100, 'ou'):
        assert parsed(f'k={sweep_runs._value(value)}') == value


def test_overrides_cover_every_key_that_determines_the_directory():
    cfg = {'n_dims': 6, 'min_target_volume': 400, 'sigma': 0.2,
           'latents': {'method': 'gpfa', 'causal_state': False, 'n_fast': 2}}
    emitted = {o.split('=', 1)[0] for o in sweep_runs.overrides({'config': cfg})}
    assert {'n_dims', 'min_target_volume', 'sigma', 'latents.method',
            'latents.causal_state', 'latents.n_fast'} <= emitted
    # a key the record predates is skipped rather than guessed at
    assert not any(o.startswith('latents.w_ridge=') for o in
                   sweep_runs.overrides({'config': cfg}))


def test_the_scored_checkpoint_wins_over_a_newer_epoch_file(tmp_path):
    d = write_run(tmp_path, 'a', 0.1, {}, states=(), best=True)
    later = d / 'states' / 'model_42.pth'
    later.write_bytes(b'')
    os.utime(later, (2 ** 31, 2 ** 31))
    assert os.path.basename(sweep_runs.state_path(str(d))) == sweep_runs.BEST_STATE


def test_a_directory_without_a_scored_checkpoint_falls_back_to_the_newest(tmp_path):
    d = write_run(tmp_path, 'b', 0.1, {}, states=('model_1.pth', 'model_9.pth'))
    os.utime(d / 'states' / 'model_9.pth', (2 ** 31, 2 ** 31))
    assert os.path.basename(sweep_runs.state_path(str(d))) == 'model_9.pth'


def test_runs_are_ranked_by_objective_and_filtered_by_recorded_config(tmp_path):
    write_run(tmp_path, 'lo', 0.01, {'num_epochs': 100}, best=True)
    write_run(tmp_path, 'hi', 0.20, {'num_epochs': 100}, best=True)
    write_run(tmp_path, 'old', 0.30, {'num_epochs': 400}, best=True)

    hits = sweep_runs.find(str(tmp_path), where=[('num_epochs', '100')])
    assert [r['run_id'] for _, r in hits] == ['hi', 'lo']


def test_a_run_whose_checkpoints_did_not_survive_is_skipped(tmp_path):
    write_run(tmp_path, 'gone', 0.5, {})
    write_run(tmp_path, 'kept', 0.1, {}, best=True)

    assert [r['run_id'] for _, r in sweep_runs.find(str(tmp_path))] == ['kept']
    assert len(sweep_runs.find(str(tmp_path), require_state=False)) == 2


def test_a_trial_from_before_the_training_fixes_is_not_offered(tmp_path):
    write_run(tmp_path, 'old', 0.9, {}, best=True, finished='2026-09-01T00:00:00')
    write_run(tmp_path, 'new', 0.1, {}, best=True)

    hits = sweep_runs.find(str(tmp_path), since=sweep_runs.ERA_START)
    assert [r['run_id'] for _, r in hits] == ['new']


def test_a_filter_can_offer_alternatives(tmp_path):
    for run_id, kind in (('const', 'const'), ('wiener', 'wiener'), ('ou', 'ou')):
        write_run(tmp_path, run_id, 0.1, {'latents': {'slow_kind': kind}}, best=True)

    hits = sweep_runs.find(str(tmp_path),
                           where=[('latents.slow_kind', 'const|wiener')])
    assert sorted(r['run_id'] for _, r in hits) == ['const', 'wiener']
