"""Replay archived scalar cumulative evidence, without training or candidate tuning."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def replay(clients, calibration):
    state={}; evidence=[]; error=0.; eps=2.220446049250313e-16
    for rd in range(1,61):
        cs=[c for c in clients if int(c['round'])==rd]
        assert len(cs)==len({c['principal_id'] for c in cs})==10
        mass=[float(c['nominal_mass']) for c in cs]
        values=[float(c['rtc_v3_residual_norm'])*w/(w+eps) for c,w in zip(cs,mass)]
        indices=sorted(range(10),key=lambda i:(values[i],cs[i]['principal_id'],cs[i]['cid']))
        total=sum(mass[i] for i in indices); partial=0.
        for i in indices:
            partial+=mass[i]
            if partial>=.5*total: center=values[i];break
        scale=min(calibration['scale_upper'],max(calibration['scale_lower'],center))
        for c,v in zip(cs,values):
            pid=c['principal_id'];z=v/(scale+eps)
            s=max(0.,state.get(pid,0.)+z-calibration['kappa']);state[pid]=s
            q=1. if s<=calibration['threshold'] else max(calibration['q_min'],math.exp(-calibration['eta']*(s-calibration['threshold'])))
            error=max(error,abs(q-float(c['rtc_v3_cumulative_q_full'])))
            evidence.append({'round':rd,'cid':c['cid'],'attack_active':c['attack_active']=='True','z':z,'statistic':s,'q':q})
    assert error<1e-9,error
    return evidence,error


def main():
    reports=[]; records=[]
    for run in rows(ROOT/'analysis/rtc_v1_multiseed/runs.csv'):
        if run['defense']!='rtc_cumulative_q_cap_accepted_anchor' or run['condition'] not in ('clean','lie_z05','lie_z025','dba'):continue
        source=Path(run['source_directory']);rid=run['run_id'];p=source/'raw'/(rid+'_clients.csv')
        cs=rows(p)
        status=json.loads((source/'status'/(rid+'.json')).read_text())
        assert status['state'] in ('completed','completed_cached') and status['exit_code']==0 and status['last_round']==60
        cfg=json.loads((source/'raw'/(rid+'_config.json')).read_text())
        custom=cfg['security']['defense']['custom_params']
        calibration_path=custom.get('calibration_manifest',custom.get('calibration_path'))
        cal=json.loads((ROOT/calibration_path).read_text()) if isinstance(calibration_path,str) else calibration_path
        assert cal['resolutions']==['full']
        evidence,error=replay(cs,cal['cumulative']['resolutions']['full'])
        for c in evidence: records.append({'condition':run['condition'],'seed':int(run['seed']),**c})
        start=1 if run['condition']=='clean' else 11
        window=[c for c in cs if int(c['round'])>=start]
        report={'condition':run['condition'],'seed':int(run['seed']),'source':p.relative_to(ROOT).as_posix(),
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'replay_max_q_error':error}
        for label,active in [('good','False'),('bad','True')]:
            group=[c for c in window if c['attack_active']==active];q=[float(c['rtc_v3_cumulative_q_full']) for c in group]
            report[label]={'n':len(group),'q_below1':sum(v<1-1e-12 for v in q),'q_at_floor':sum(v<=.500000000001 for v in q),
                'mean_q':st.mean(q) if q else None,'weight_per_round':sum(float(c['aggregation_weight']) for c in group)/(61-start) if group else None,
                'weight_before_cumulative_trigger_per_round':sum(float(c['aggregation_weight']) for c in group if float(c['rtc_v3_cumulative_q_full'])>=1-1e-12)/(61-start) if group else None,
                'weight_exceeding_q_squared_per_round':sum(max(0,float(c['aggregation_weight'])-float(c['nominal_mass'])*float(c['rtc_v3_cumulative_q_full'])**2) for c in group)/(61-start) if group else None,
                'first_q_lt1':min((int(c['round']) for c in group if float(c['rtc_v3_cumulative_q_full'])<1-1e-12),default=None)}
        reports.append(report)
    (HERE/'cumulative_inventory.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    (HERE/'cumulative_replay.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps({'runs':len(reports),'records':len(records),'max_q_error':max(r['replay_max_q_error'] for r in reports)},indent=2))


if __name__=='__main__':main()
