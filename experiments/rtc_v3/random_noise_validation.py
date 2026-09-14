"""Random noise v2 execution and reproducible strength-freeze evidence."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "random_noise.trainable_rademacher_norm_matched.v2"
SCOPE = "trainable_parameters_only"
THRESHOLD = 0.03


def _truth(value):
    return str(value).lower() in {"true", "1", "1.0"}


def validate_client_evidence(spec, rounds, raw_dir, *, evidence_path=None):
    from experiments.periodic_attack import run_id
    if spec.get("attack_version") != VERSION:
        raise ValueError("Random noise execution requires v2")
    scale = float(spec["random_noise_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Formal Random noise scale must be finite and positive")
    c = pd.read_csv(evidence_path or Path(raw_dir) / f"{run_id(dict(spec))}_clients.csv")
    required = {"round", "cid", "attack_active", "is_malicious", "random_noise_version", "random_noise_scope",
                "random_noise_dimension", "random_noise_tensor_count", "random_noise_reference_norm",
                "random_noise_uploaded_norm", "random_noise_strength", "random_noise_coordinate_scale",
                "random_noise_buffers_preserved", "random_noise_output_finite", "random_noise_zero_reference"}
    if required.difference(c):
        raise ValueError("Missing per-client Random noise evidence: " + str(sorted(required.difference(c))))
    if c.duplicated(["round", "cid"]).any():
        raise ValueError("Duplicate Random noise client evidence")
    active = c[c.attack_active.map(_truth)].copy()
    if active.empty or not active.is_malicious.map(_truth).all():
        raise ValueError("Random noise active-client evidence is invalid")
    for _, row in rounds[rounds['round'] > 0].iterrows():
        clients = c[c['round'] == row['round']]
        planned = json.loads(str(row['fit_completed_partition_ids_json']))
        if set(clients.cid.astype(str)) != set(map(str, planned)):
            raise ValueError("Incomplete Random noise client evidence")
        expected = int(row['fit_selected_active_attackers'])
        actual = clients.attack_active.map(_truth)
        if int(actual.sum()) != expected or (expected and not _truth(row['planned_attack_active'])):
            raise ValueError("Random noise attack window/participant mismatch")
    for column in ('random_noise_dimension', 'random_noise_tensor_count', 'random_noise_reference_norm',
                   'random_noise_uploaded_norm', 'random_noise_strength', 'random_noise_coordinate_scale'):
        active[column] = pd.to_numeric(active[column], errors='coerce')
        if not np.isfinite(active[column]).all():
            raise ValueError(f"Nonfinite Random noise evidence: {column}")
    ref = active.random_noise_reference_norm
    actual = active.random_noise_uploaded_norm
    dim = active.random_noise_dimension
    if (not active.random_noise_version.eq(VERSION).all() or not active.random_noise_scope.eq(SCOPE).all()
            or not active.random_noise_buffers_preserved.map(_truth).all()
            or not active.random_noise_output_finite.map(_truth).all()
            or not np.allclose(active.random_noise_strength, scale, rtol=0, atol=1e-12)
            or not ((dim > 0) & (dim == np.floor(dim))).all()
            or dim.nunique() != 1 or not (active.random_noise_tensor_count > 0).all()
            or not (ref >= 0).all() or not (actual >= 0).all()
            or not np.allclose(actual, ref * scale, rtol=1e-4, atol=1e-10)
            or ((ref > 0) & (actual <= 0)).any()
            or not np.array_equal(active.random_noise_zero_reference.map(_truth), ref == 0)
            or not np.allclose(active.random_noise_coordinate_scale, scale * ref / np.sqrt(dim), rtol=1e-6, atol=1e-12)):
        raise ValueError("Random noise norm/scope/buffer contract violated")
    if not (ref > 0).any():
        raise ValueError("Random noise has no nonzero-reference attack evidence")


def _artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read_artifact(record, base):
    path = Path(record['path'])
    if not path.is_absolute():
        path = base / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError("Random noise screening evidence hash mismatch")
    return path


def validate_freeze_entry(entry, base):
    from attacks.spec import attack_source_hash
    evidence = entry.get('validation', {})
    scale = float(entry.get('parameters', {}).get('random_noise_scale', math.nan))
    if (evidence.get('schema_version') != 1 or evidence.get('attack_version') != VERSION
            or evidence.get('scope') != SCOPE or evidence.get('source_hash') != attack_source_hash()
            or entry.get('selection_metric') != 'accuracy_drop'
            or entry.get('selection_threshold') != THRESHOLD
            or not math.isfinite(scale) or scale <= 0
            or entry.get('parameters', {}).get('random_noise_distribution') != 'rademacher'):
        raise ValueError("Random noise v2 requires verified FedAvg screening evidence")
    seeds = set(); drops = []
    for run in evidence.get('runs', []):
        if run['seed'] in seeds:
            raise ValueError("Duplicate screening seed")
        seeds.add(run['seed'])
        frames = [pd.read_csv(_read_artifact(run[k], Path(base))).sort_values('round').reset_index(drop=True)
                  for k in ('attack_rounds', 'clean_rounds')]
        manifests = [json.loads(_read_artifact(run[k], Path(base)).read_text()) for k in ('attack_pairing', 'clean_pairing')]
        for key in ('trial_plan_hash', 'initial_model_sha256', 'data_manifest_sha256'):
            if not manifests[0].get(key) or manifests[0][key] != manifests[1].get(key):
                raise ValueError("Screening clean reference is not paired")
        for frame, attack in zip(frames, ('random_noise', 'none')):
            if (len(frame) != 61 or set(frame['round']) != set(range(61)) or frame['round'].duplicated().any()
                    or not np.isfinite(frame.server_loss).all() or not np.isfinite(frame.server_accuracy).all()
                    or not frame.attack.eq(attack).all()
                    or not frame.defense.eq('fedavg').all()
                    or not frame.seed.eq(run['seed']).all()):
                raise ValueError("Screening needs complete finite 60-round FedAvg runs")
            for flag in ('server_model_state_valid', 'server_logits_valid', 'server_loss_valid'):
                if flag not in frame or not frame[flag].map(_truth).all():
                    raise ValueError("Screening numerical validity evidence missing")
        attacked, clean = frames
        spec = run['attack_spec']
        if (spec.get('attack_version') != VERSION or spec.get('random_noise_scale') != scale
                or spec.get('trial_plan_hash') != manifests[0]['trial_plan_hash']
                or spec.get('seed') != run['seed'] or spec.get('defense') != 'fedavg'
                or spec.get('malicious_fraction') != 0.3 or spec.get('participation_rate') != 0.5
                or spec.get('partition') != 'iid'):
            raise ValueError('Screening specification differs from the formal protocol')
        from experiments.rtc_v3.byzantine import _runtime_attack_config
        from attacks.spec import attack_contract_payload
        runtime = _runtime_attack_config('random_noise', entry['parameters'], malicious_fraction=0.3,
                                         attack_start_round=11, attack_end_round=-1)
        implementation = attack_contract_payload('random_noise', vars(runtime))['implementation_hash']
        if (spec.get('attack_implementation_hash') != implementation
                or not attacked.attack_implementation_hash.eq(implementation).all()):
            raise ValueError('Screening attack implementation differs from current code')
        clients = _read_artifact(run['attack_clients'], Path(base))
        validate_client_evidence(spec, attacked, clients.parent, evidence_path=clients)
        if (not attacked.attack_version.eq(VERSION).all()
                or not attacked.random_noise_scale.eq(scale).all()
                or not attacked.loc[attacked['round'].between(11,60), 'planned_attack_active'].eq(1).all()
                or not attacked.loc[attacked['round'].between(0,10), 'planned_attack_active'].eq(0).all()
                or not clean.planned_attack_active.eq(0).all()
                or not attacked.fit_selected_partition_ids.fillna('').equals(clean.fit_selected_partition_ids.fillna(''))
                or not attacked.fit_fit_seed_digest.fillna('').equals(clean.fit_fit_seed_digest.fillna(''))):
            raise ValueError("Screening version, scale, window or random-stream mismatch")
        drop = float(clean.loc[clean['round']==60,'server_accuracy'].iloc[0] - attacked.loc[attacked['round']==60,'server_accuracy'].iloc[0])
        if drop < THRESHOLD or not math.isclose(drop, run['accuracy_drop'], abs_tol=1e-12):
            raise ValueError("Random noise effect threshold not met for every seed")
        drops.append(drop)
    if len(seeds) < 2 or not math.isclose(float(entry.get('selection_value', math.nan)), min(drops), abs_tol=1e-12):
        raise ValueError("Random noise freeze needs at least two verified seeds")


def attach_freeze_evidence(payload, summary, specs, output):
    from attacks.spec import attack_source_hash
    from experiments.periodic_attack import run_id
    entry = payload['attacks']['random_noise']
    scale = entry['parameters']['random_noise_scale']
    selected = summary[(summary.attack=='random_noise') & (summary.defense=='fedavg') & (summary.random_noise_scale==scale)]
    evidence=[]
    spec_by_id = {run_id(spec):spec for spec in specs}
    for _, row in selected.iterrows():
        clean = summary[(summary.attack=='none') & (summary.defense=='fedavg') & (summary.trial_plan_hash==row.trial_plan_hash)]
        if len(clean)!=1:
            raise ValueError('Missing unique plan-matched clean reference')
        other=clean.iloc[0]
        evidence.append({'seed':int(row.seed), 'accuracy_drop':float(other.final_accuracy-row.final_accuracy),
                         'attack_spec':spec_by_id[row.run_id],
                         'attack_clients':_artifact(output/'raw'/f'{row.run_id}_clients.csv'),
                         'attack_rounds':_artifact(output/'rounds'/f'{row.run_id}.csv'),
                         'clean_rounds':_artifact(output/'rounds'/f'{other.run_id}.csv'),
                         'attack_pairing':_artifact(output/'raw'/f'{row.run_id}_pairing_manifest.json'),
                         'clean_pairing':_artifact(output/'raw'/f'{other.run_id}_pairing_manifest.json')})
    entry.update(selection_metric='accuracy_drop', selection_threshold=THRESHOLD,
                 selection_value=min((r['accuracy_drop'] for r in evidence), default=math.nan),
                 validation={'schema_version':1, 'scope':SCOPE, 'attack_version':VERSION,
                             'source_hash':attack_source_hash(), 'runs':evidence})
    validate_freeze_entry(entry, output)
