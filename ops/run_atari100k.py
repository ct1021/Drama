"""One bounded online Drama baseline run; no experiment queue or cloud uploads."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['WANDB_MODE'] = 'disabled'
import numpy as np
import torch
import yaml
import train
from eval import build_single_env
from replay_buffer import ReplayBuffer
from envs.my_atari import Atari
from utils import seed_np_torch


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    temp.replace(path)


class LocalLogger:
    enable_video = False

    def __init__(self, directory, limit_hours):
        self.directory = directory
        self.started = time.time()
        self.limit = limit_hours * 3600
        self.last_step = 0
        self.latest = {}
        self.metrics = (directory / 'metrics.jsonl').open('a', encoding='utf-8')
        self.status('starting')

    def log(self, tag, value, global_step):
        if any(kind in tag for kind in ('video', 'images', 'hist')):
            return
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError(f'Non-finite {tag} at step {global_step}')
        self.latest[tag] = value
        self.metrics.write(json.dumps(dict(step=int(global_step), tag=tag, value=value,
                                          time=time.time())) + '\n')

    def status(self, phase, **extra):
        self.metrics.flush()
        atomic_json(self.directory / 'status.json', dict(
            phase=phase, pid=os.getpid(), step=self.last_step,
            started_unix=self.started, updated_unix=time.time(),
            elapsed_hours=(time.time()-self.started)/3600,
            peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
            latest=self.latest, **extra))

    def on_step(self, step):
        self.last_step = step
        if step % 100 == 0 or step <= 2:
            self.status('training')
        if time.time() - self.started > self.limit:
            raise TimeoutError('Configured run wall-time limit reached')


@torch.no_grad()
def evaluate_fixed_episodes(config, world_model, agent, logger, global_step=None):
    """Exactly one complete episode per fixed seed; completed slots never restart."""
    count = config.Evaluate.EpisodeNum
    seeds = [200000 + config.BasicSettings.Seed * 10 + i for i in range(count)]
    envs = []
    was_training = (world_model.training, agent.training)
    np_state, py_state = np.random.get_state(), random.getstate()
    logger.last_step = global_step
    logger.status('evaluating')
    started = time.time()
    scores = np.zeros(count)
    lengths = np.zeros(count, dtype=int)
    done = np.zeros(count, dtype=bool)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            seed_np_torch(900000 + config.BasicSettings.Seed)
            world_model.eval()
            agent.eval()
            obs = []
            for seed in seeds:
                env = build_single_env(config.BasicSettings.Env_name, config.BasicSettings.ImageSize)
                envs.append(env)
                env.action_space.seed(seed)
                obs.append(env.reset(seed=seed)[0])
            obs = np.stack(obs)
            context_obs = deque(maxlen=config.JointTrainAgent.RealityContextLength)
            context_actions = deque(maxlen=config.JointTrainAgent.RealityContextLength)
            while not done.all():
                if time.time() - logger.started > logger.limit:
                    raise TimeoutError('Run wall-time limit reached during evaluation')
                if not context_actions:
                    actions = np.array([env.action_space.sample() for env in envs])
                else:
                    latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    history_actions = torch.as_tensor(np.stack(context_actions, axis=1),
                                                      device=world_model.device, dtype=torch.float32)
                    prior, feature = world_model.calc_last_dist_feat(latent, history_actions)
                    actions = agent.sample_as_env_action(torch.cat([prior, feature], dim=-1), greedy=True)
                context_obs.append(torch.as_tensor(obs, device=world_model.device,
                                                   dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(1)/255)
                context_actions.append(actions)
                for i, env in enumerate(envs):
                    if done[i]:
                        continue
                    next_obs, reward, terminated, truncated, _ = env.step(int(actions[i]))
                    obs[i] = next_obs
                    scores[i] += reward
                    lengths[i] += 1
                    done[i] = terminated or truncated
                    if lengths[i] > 27000:
                        raise RuntimeError('Evaluation exceeded Atari 108000-frame episode cap')
            record = dict(training_step=global_step, seeds=seeds, returns=scores.tolist(),
                          action_counts=lengths.tolist(), mean=float(scores.mean()),
                          seconds=time.time()-started, policy='greedy', episodes=count)
            with (logger.directory / 'evaluations.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            logger.log('evaluate/score', scores.mean(), global_step)
            print('EVALUATION ' + json.dumps(record), flush=True)
            return record
    finally:
        for env in envs:
            env.close()
        np.random.set_state(np_state)
        random.setstate(py_state)
        world_model.train(was_training[0])
        agent.train(was_training[1])
        logger.status('training')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--game', choices=['Boxing', 'Krull', 'Breakout'], default='Boxing')
    parser.add_argument('--seed', type=int, default=3710)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--max-hours', type=float, default=11.9)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / 'status.json').exists():
        raise RuntimeError('Run already exists; refusing to overwrite or silently resume')
    os.chdir(ROOT)
    raw = yaml.safe_load((ROOT / 'config_files/configure.yaml').read_text())
    raw['BasicSettings'].update(Env_name=f'ALE/{args.game}-v5', Seed=args.seed,
                                Device='cuda:0', Compile=False)
    raw['Models']['WorldModel']['dtype'] = torch.float32
    raw['Models']['Agent']['dtype'] = torch.float32
    raw['JointTrainAgent'].update(SampleMaxSteps=args.steps,
                                  FreezeWorldModelAfterSteps=args.steps,
                                  FreezeBehaviourAfterSteps=args.steps,
                                  SaveEverySteps=2000)
    raw['Evaluate'].update(EpisodeNum=args.eval_episodes, NumEnvs=args.eval_episodes,
                           DuringTraining=True, EverySteps=10000)
    config = train.DotDict(raw)
    seed_np_torch(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger = LocalLogger(args.run_dir, args.max_hours)
    world_model = agent = None
    try:
        dummy = Atari(config.BasicSettings.Env_name, seed=args.seed)
        action_dim = dummy.action_space.n
        dummy.close()
        world_model = train.build_world_model(config, action_dim, 'cuda:0')
        agent = train.build_agent(config, action_dim, 'cuda:0')
        train.update_model_parameters(config, world_model, agent)
        atomic_json(args.run_dir / 'config.resolved.json', config)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        atomic_json(args.run_dir / 'manifest.json', dict(
            code_commit=commit, branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
            command=sys.argv, scope='baseline' if args.steps == 100000 else 'integration_only',
            env=config.BasicSettings.Env_name, seed=args.seed, target_interactions=args.steps,
            eval_interval=10000, eval_episodes=args.eval_episodes,
            compile=False, amp=config.BasicSettings.Use_amp, cuda_graph=config.BasicSettings.Use_cg,
            architecture='upstream YAML four-stage encoder; not paper-exact DramaXS',
            wall_time_limit_hours=args.max_hours, automatic_next_run=False,
            checkpoint_scope='weights only; not exact resumable training state'))
        replay = ReplayBuffer(config, device='cuda:0', action_dim=action_dim, is_discrete=True)
        train.eval_episodes = evaluate_fixed_episodes
        train.joint_train_world_model_agent(config, str(args.run_dir), replay, world_model, agent, logger)
        torch.save(world_model.state_dict(), args.run_dir / 'ckpt/world_model-final.pth')
        torch.save(agent.state_dict(), args.run_dir / 'ckpt/agent-final.pth')
        logger.status('completed', target_interactions=args.steps)
    except BaseException as exc:
        # Preserve weights for diagnosis, but do not claim exact resumability.
        if world_model is not None and agent is not None:
            (args.run_dir / 'ckpt').mkdir(exist_ok=True)
            torch.save(world_model.state_dict(), args.run_dir / 'ckpt/world_model-interrupted.pth')
            torch.save(agent.state_dict(), args.run_dir / 'ckpt/agent-interrupted.pth')
        logger.status('failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        logger.metrics.close()


if __name__ == '__main__':
    main()
