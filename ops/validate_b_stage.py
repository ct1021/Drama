"""One B GPU/1800-step gate, under an external 15-minute hard limit."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from b_gate import short_training_gate
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False)
    deadline=time.monotonic()+890
    def execute(argv,name):
        remaining=deadline-time.monotonic();assert remaining>0,'B preflight deadline reached'
        with (out/name).open('xb') as log:
            subprocess.run([sys.executable,'-u',*argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,timeout=remaining,check=True)
    for script in ('test_loss_harmonization.py','test_experiment_profiles.py','test_training_protocol.py'):
        execute(['ops/'+script],script+'.log')
    execute(['ops/verify_b_control.py','--output',str(out/'gpu.json')],'gpu.log')
    run=out/'training';run.mkdir()
    execute(['ops/run_atari100k.py','--profile','lightweight-harmonized','--run-dir',str(run),
             '--game','Boxing','--seed','3710','--steps','1800','--eval-episodes','2',
             '--max-hours',str(max(.01,(deadline-time.monotonic()-10)/3600))],'integration.log')
    reference=json.loads(args.reference.read_text())[0]
    result=short_training_gate(run,reference,json.loads((out/'gpu.json').read_text()))
    # Load the actual training checkpoint; ensure the learned scalars survived.
    import torch
    state=torch.load(run/'ckpt/world_model-final.pth',map_location='cpu',weights_only=True)
    scalar=state['loss_harmonizer.log_scales']
    assert scalar.shape==(3,) and torch.isfinite(scalar).all() and (scalar.abs()>1e-8).all()
    result.update(checkpoint_log_scales=scalar.tolist(),code_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),reference_file=str(args.reference))
    (out/'gate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    assert result['passed'],'B failed the fixed 15% timing gate; formal run will not start'


if __name__=='__main__':main()
