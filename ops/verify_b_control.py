"""GPU checks for B, invoked only after D has exited. No optimizer steps."""
import argparse,contextlib,json,sys
from pathlib import Path
import torch,yaml
from verify_task_routing import run,ROOT
from experiment_profiles import apply_profile
from train import DotDict,build_world_model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        states=[]
        for profile in ('lightweight','lightweight-harmonized'):
            torch.manual_seed(3710)
            raw=apply_profile(yaml.safe_load((ROOT/'config_files/configure.yaml').read_text()),profile)
            raw['Models']['WorldModel']['dtype']=torch.float32
            wm=build_world_model(DotDict(raw),18,'cuda:0')
            state={k:v.detach().cpu().clone() for k,v in wm.state_dict().items()}
            if profile=='lightweight-harmonized':
                assert sum(p.numel() for p in wm.parameters())==7161606
                groups=wm.optimizer.param_groups
                scalar=wm.loss_harmonizer.log_scales
                scalar_group=[g for g in groups if any(p is scalar for p in g['params'])]
                assert len(scalar_group)==1 and scalar_group[0]['weight_decay']==0
                assert len({g['lr'] for g in groups})==1
                assert wm.routing_metadata is None
            states.append(state);del wm
        only=set(states[1])-set(states[0]);assert only=={'loss_harmonizer.log_scales'},only
        assert not (set(states[0])-set(states[1]))
        for key,value in states[0].items():torch.testing.assert_close(value,states[1][key],rtol=0,atol=0)
        del states;torch.cuda.empty_cache()
        result=run('lightweight-harmonized')
        assert result['world_model_parameters']==7161606
    result.update(ok=True,identical_base_initialization=True,extra_training_scalars=3,
                  scalar_optimizer_coverage=True,scalar_weight_decay=0,
                  scope='B GPU correctness; not RL performance')
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
