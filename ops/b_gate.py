"""Pure validation rules for the one authorized B control."""
import hashlib,json,math
from pathlib import Path


def completed_100k(status, evaluations):
    return (status.get('phase')=='completed' and status.get('step')==100000
            and bool(evaluations) and evaluations[-1].get('training_step')==100000
            and evaluations[-1].get('episodes')==10
            and len(evaluations[-1].get('returns',[]))==10
            and all(math.isfinite(x) for x in evaluations[-1]['returns']))


def short_training_gate(run, reference, gpu_result):
    status=json.loads((run/'status.json').read_text())
    assert status['phase']=='completed' and status['step']==1800
    assert gpu_result['ok'] and gpu_result['world_model_parameters']==7161606
    assert gpu_result['cache_bytes_per_row_float32']==164864
    assert gpu_result['identical_base_initialization'] and gpu_result['scalar_optimizer_coverage']
    evaluations=[json.loads(line) for line in (run/'evaluations.jsonl').read_text().splitlines()]
    assert len(evaluations)==1 and evaluations[0]['training_step']==1800 and evaluations[0]['episodes']==2
    times={};scales={};metric_count=0
    for line in (run/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line);assert math.isfinite(float(row['value'])),row
        metric_count+=1
        if row['tag']=='WorldModel/total_loss' and 1200<=row['step']<=1700:times[row['step']]=row['time']
        if row['tag'].startswith('WorldModel/Harmony/') and row['tag'].endswith('_log_scale'):
            scales[row['tag']]=row['value']
    assert len(times)>400 and len(scales)==3 and all(abs(v)>1e-8 for v in scales.values())
    assert any(k.startswith('ActorCritic/') for k in status['latest'])
    first,last=min(times),max(times);seconds=(times[last]-times[first])/(last-first)
    assert reference['profile']=='lightweight' and reference['phase']=='completed' and reference['step']==1800
    assert reference['seconds_per_interaction']>0 and seconds>0
    ratio=seconds/reference['seconds_per_interaction']
    result=dict(passed=ratio<=1.15,window=[first,last],seconds_per_interaction=seconds,
                time_ratio_vs_A=ratio,threshold=1.15,finite_metric_rows=metric_count,
                final_log_scales=scales,step=1800,scope='Engineering gate, not RL performance')
    return result
