"""Wait for the fixed D, then run exactly one B gate and one B 100K.

No restart, extra seed, package installation, instance or old-queue actions.
"""
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from b_gate import completed_100k
ROOT=Path(__file__).resolve().parents[1]
D_NAME='boxing-h1-routed-s3710-100k-20260921'
B_NAME='boxing-optimization-control-s3710-100k-20260922'
CONTROL='b-after-d-s3710-20260921'


def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def active_run(run,status):
    p=Path('/proc')/str(status.get('pid',0))/'cmdline'
    command=p.read_bytes() if p.exists() else b''
    return b'ops/run_atari100k.py' in command and str(run).encode() in command


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True);args=parser.parse_args()
    runs=args.runs.resolve();d=runs/D_NAME;b=runs/B_NAME;control=runs/CONTROL
    assert d.exists() and not b.exists() and not control.exists(),'Missing D or duplicate B'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    d_manifest=json.loads((d/'manifest.json').read_text())
    assert d_manifest['code_commit']=='5829fc831febdbd844506f9d87906e6c0b348cbb'
    assert d_manifest['profile']=='lightweight-routed' and d_manifest['seed']==3710
    assert d_manifest['target_interactions']==100000 and d_manifest['env']=='ALE/Boxing-v5'
    start_status=json.loads((d/'status.json').read_text())
    wait_deadline=start_status['started_unix']+7*3600+120
    control.mkdir()
    state=dict(phase='waiting_for_D',pid=os.getpid(),started_unix=time.time(),
               code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               predecessor=str(d),run_dir=str(b),wait_deadline_unix=wait_deadline,
               preflight_limit_seconds=900,formal_limit_seconds=25200,
               automatic_restart=False,additional_runs=False)
    active=None
    def save(**changes):
        state.update(changes);state['updated_unix']=time.time();atomic(control/'status.json',state)
    def stop(signum,frame):raise InterruptedError('B coordinator signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def execute(argv,cap,log):
        nonlocal active
        with log.open('xb') as stream:
            active=subprocess.Popen(['timeout','--signal=TERM','--kill-after=15s',str(cap)+'s',
                sys.executable,'-u',*argv],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            save(child_supervisor_pid=active.pid)
            rc=active.wait();active=None
        assert rc==0,f'B stage exited {rc}; see {log}'
    try:
        save()
        while True:
            status=json.loads((d/'status.json').read_text())
            pair=json.loads((runs/'h1-pair-s3710-20260921/pair-status.json').read_text())
            if status['phase']=='failed' or pair['phase']=='failed':raise RuntimeError('D/pair failed; B will not start')
            if time.time()>wait_deadline:raise TimeoutError('D did not finish before its bounded deadline')
            if status['phase']=='completed':
                ev=[json.loads(x) for x in (d/'evaluations.jsonl').read_text().splitlines()]
                assert completed_100k(status,ev),'D is not a complete successful 100K'
                if not active_run(d,status) and pair['phase']=='completed':break
            elif not active_run(d,status):raise RuntimeError('D exited without successful completion')
            save(predecessor_step=status['step'],predecessor_phase=status['phase'])
            time.sleep(20)
        assert shutil.disk_usage(runs).free>5*2**30
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
        records=[]
        for name in ('status.json','manifest.json','config.resolved.json','evaluations.jsonl',
                     'metrics.jsonl','console.log','ckpt/world_model-final.pth','ckpt/agent-final.pth'):
            p=d/name;assert p.is_file() and p.stat().st_size>0
            h=hashlib.sha256()
            with p.open('rb') as stream:
                while block:=stream.read(1024*1024):h.update(block)
            records.append(dict(path=name,bytes=p.stat().st_size,sha256=h.hexdigest()))
        atomic(control/'D-completed-evidence.json',dict(status=status,evaluations=ev,files=records))
        gate=control/'preflight';save(phase='preflight',preflight_started_unix=time.time())
        execute(['ops/validate_b_stage.py','--output',str(gate),'--reference',str(args.reference.resolve())],900,control/'preflight.log')
        verdict=json.loads((gate/'gate.json').read_text());assert verdict['passed']
        assert verdict['code_commit']==state['code_commit']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
        assert shutil.disk_usage(runs).free>5*2**30
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        b.mkdir(exist_ok=False);save(phase='training',formal_started_unix=time.time(),gate=verdict)
        execute(['ops/run_atari100k.py','--profile','lightweight-harmonized','--run-dir',str(b),
                 '--game','Boxing','--seed','3710','--steps','100000','--eval-episodes','10','--max-hours','6.9'],25200,b/'console.log')
        final=json.loads((b/'status.json').read_text())
        ev=[json.loads(x) for x in (b/'evaluations.jsonl').read_text().splitlines()]
        assert completed_100k(final,ev),'B did not complete 100K'
        save(phase='completed',step=100000,final_mean=ev[-1]['mean'])
    except BaseException as error:
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=20)
            except subprocess.TimeoutExpired:active.kill();active.wait()
        save(phase='failed',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
