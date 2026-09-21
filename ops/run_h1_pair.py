"""Execute only the specified C/D pair after its integration gate passes.

No retries, additional seeds, installations, instance actions or old queues.
The caller must explicitly launch this program; importing it does nothing.
"""
import argparse,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path


def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',required=True,type=Path)
    parser.add_argument('--gate',required=True,type=Path)
    parser.add_argument('--runs',required=True,type=Path)
    args=parser.parse_args()
    code=args.code.resolve();runs=args.runs.resolve()
    gates=json.loads(args.gate.read_text())
    assert [x['profile'] for x in gates]==['lightweight','lightweight-readout','lightweight-routed']
    assert all(x['phase']=='completed' and x['step']==1800 for x in gates)
    assert gates[-1]['time_ratio_vs_A']<=1.15,'Efficiency gate did not pass'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    assert all(x['code_commit']==commit for x in gates),'Must use tested training commit'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip()
    jobs=[('lightweight-readout','boxing-readout-control-s3710-100k-20260921'),
          ('lightweight-routed','boxing-h1-routed-s3710-100k-20260921')]
    assert all(not (runs/name).exists() for _,name in jobs),'Refusing duplicate or resumed run'
    pair=runs/'h1-pair-s3710-20260921';pair.mkdir(exist_ok=False)
    state={'phase':'starting','pid':os.getpid(),'started_unix':time.time(),
           'code_commit':commit,'gate':str(args.gate),'scope':'exactly two authorized C/D experiments',
           'runs':[{'profile':p,'run_dir':str(runs/n),'phase':'pending'} for p,n in jobs],
           'automatic_restart':False,'additional_experiments':False,'per_run_cap_hours':7}
    atomic(pair/'pair-status.json',state)
    active=None
    def interrupted(signum,frame):raise InterruptedError('Pair interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        for index,(profile,name) in enumerate(jobs):
            assert shutil.disk_usage(runs).free>5*2**30,'Insufficient data disk space'
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied by another process'
            run=runs/name;run.mkdir(exist_ok=False)
            cmd=['timeout','--signal=TERM','--kill-after=60s','7h',sys.executable,'-u',
                 str(code/'ops/run_atari100k.py'),'--run-dir',str(run),'--profile',profile,
                 '--game','Boxing','--seed','3710','--steps','100000','--eval-episodes','10','--max-hours','6.9']
            with (run/'console.log').open('xb') as log:
                active=subprocess.Popen(cmd,cwd=code,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                (run/'launch.pid').write_text(str(active.pid)+'\n')
                state.update(phase='running',current=index,updated_unix=time.time())
                state['runs'][index].update(phase='running',supervisor_pid=active.pid)
                atomic(pair/'pair-status.json',state)
                rc=active.wait();active=None
            s=json.loads((run/'status.json').read_text())
            if rc or s['phase']!='completed' or s['step']!=100000:
                state['runs'][index].update(phase='failed',step=s.get('step'),returncode=rc)
                raise RuntimeError('Run did not reach successful 100K; remaining job will not start')
            state['runs'][index].update(phase='completed',step=100000)
            atomic(pair/'pair-status.json',state)
        state.update(phase='completed',updated_unix=time.time())
        atomic(pair/'pair-status.json',state)
    except BaseException as error:
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=65)
            except subprocess.TimeoutExpired:active.kill();active.wait()
        state.update(phase='failed',error=repr(error),updated_unix=time.time())
        atomic(pair/'pair-status.json',state)
        raise


if __name__=='__main__':main()
