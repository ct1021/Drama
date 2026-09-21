"""Three fixed 1800-interaction engineering tests, never a 100K run queue."""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for profile in ('lightweight','lightweight-readout','lightweight-routed'):
        run=args.output/profile;run.mkdir()
        print(json.dumps({'starting':profile,'target_interactions':1800,'integration_only':True}),flush=True)
        cmd=[sys.executable,'-u',str(ROOT/'ops/run_atari100k.py'),'--profile',profile,
             '--run-dir',str(run),'--game','Boxing','--seed','3710','--steps','1800',
             '--eval-episodes','2','--max-hours','0.15']
        with (run/'console.log').open('xb') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,timeout=600)
        status=json.loads((run/'status.json').read_text())
        assert result.returncode==0 and status['phase']=='completed' and status['step']==1800,status
        assert all(math.isfinite(v) for v in status['latest'].values() if isinstance(v,(int,float)))
        assert any(k.startswith('ActorCritic/') for k in status['latest'])
        ev=[json.loads(x) for x in (run/'evaluations.jsonl').read_text().splitlines()]
        assert len(ev)==1 and ev[0]['episodes']==2 and len(ev[0]['returns'])==2
        times={}
        for line in (run/'metrics.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['tag']=='WorldModel/total_loss' and 1200<=row['step']<=1700:
                times[row['step']]=row['time']
        assert len(times)>400,len(times)
        first,last=min(times),max(times)
        seconds_per_step=(times[last]-times[first])/(last-first)
        record=dict(profile=profile,phase=status['phase'],step=status['step'],
                    pid=status['pid'],elapsed_seconds=status['elapsed_hours']*3600,
                    window=[first,last],seconds_per_interaction=seconds_per_step,
                    peak_gpu_GiB=status['peak_gpu_GiB'],evaluations=ev,
                    code_commit=json.loads((run/'manifest.json').read_text())['code_commit'])
        results.append(record)
        (args.output/'integration-summary.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(record),flush=True)
    baseline=results[0]['seconds_per_interaction']
    for row in results:row['time_ratio_vs_A']=row['seconds_per_interaction']/baseline
    (args.output/'integration-summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'completed':True,'results':results,'scope':'short real training and timing, not RL benchmark evidence'}),flush=True)


if __name__=='__main__':main()
