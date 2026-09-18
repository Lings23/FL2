"""Old-trajectory G1 repair diagnosis; no threshold fitting or training."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from defenses.rtc.filtered_lower_tail import FilteredLowerTailEvidence
from defenses.rtc.lower_tail import load_calibration
from experiments.rtc_r0b_observation import truth


def main():
    batch=ROOT/'logs/rtc_r3_lower_tail'
    saved=ROOT/'analysis/rtc_retained_candidates/M2-G1'
    receipt=json.loads((saved/'candidate.json').read_text())
    for name,expected in receipt['snapshot_sha256'].items():
        assert hashlib.sha256((saved/name).read_bytes()).hexdigest()==expected,name
    evidence=json.loads((saved/'review.json').read_text())['evidence']
    cal=load_calibration(saved/'rtc_r3_lower_tail_calibration.json')
    reports=[]
    for path in sorted(batch.glob('*/raw/*_clients.csv')):
        if 'lower_' not in path.name:continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert evidence[path.relative_to(batch).as_posix()]==digest
        rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        scorer=FilteredLowerTailEvidence(cal)
        counts=dict(old_good=0,old_bad=0,filtered_good=0,filtered_bad=0,abstained_records=0)
        for rnd in range(1,61):
            clients=[r for r in rows if int(r['round'])==rnd]
            scores,state=scorer.prepare([float(r['rtc_r3l_residual_norm']) for r in clients],
                [r['principal_id'] for r in clients],
                [float(r['rtc_r3l_nominal_mass']) for r in clients],'observe',
                [truth(r['rtc_r2_flagged']) for r in clients])
            scorer.commit(state)
            for row,score in zip(clients,scores):
                counts['abstained_records']+=int(not score['rtc_r3l_valid'])
                if rnd<=10 and not path.parent.parent.name.startswith('none_'):continue
                label='bad' if truth(row['attack_active']) else 'good'
                counts['old_'+label]+=int(truth(row['rtc_r3l_flagged']))
                counts['filtered_'+label]+=int(score['rtc_r3l_flagged'])
        reports.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest,**counts))
    assert len(reports)==16
    result=dict(rule='Exclude current R2 rejected principals; >=9 other eligible peers; invalid resets; two visits; original threshold unchanged',
        threshold=cal['ratio_threshold'],closed_loop=False,threshold_fitted=False,runs=reports)
    (Path(__file__).parent/'g1_filter_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
