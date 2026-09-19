"""Check actual interaction accounting and episode reset before paid training."""
from pathlib import Path
import sys
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train


def test_count_reset_and_final_evaluation():
    config = train.DotDict(dict(
        BasicSettings=dict(Env_name='ALE/Boxing-v5', ImageSize=[64, 64], Seed=3710),
        JointTrainAgent=dict(RealityContextLength=16, SampleMaxSteps=3, NumEnvs=1,
                             SaveModels=True, SaveEverySteps=2),
        Evaluate=dict(DuringTraining=True, EverySteps=2)))
    env = Mock()
    env.action_space.n = 18
    env.action_space.sample.return_value = 0
    env.reset.side_effect = [(10, {}), (20, {}), (30, {}), (40, {})]
    env.step.return_value = (99, 1.0, True, dict(is_terminal=True, episode_frame_number=4))
    replay = Mock()
    replay.ready.return_value = False
    logger, world, agent = Mock(), Mock(), Mock()
    with patch.object(train, 'Atari', return_value=env), \
         patch.object(train.os, 'makedirs'), \
         patch.object(train.torch, 'save'), \
         patch.object(train, 'tqdm', side_effect=lambda values, **kwargs: values), \
         patch.object(train, 'eval_episodes') as evaluate:
        train.joint_train_world_model_agent(config, '/unused', replay, world, agent, logger)
    assert [call.args[0] for call in replay.append.call_args_list] == [10, 20, 30]
    assert [call.args[0] for call in logger.on_step.call_args_list] == [1, 2, 3]
    assert [call.args[-1] for call in evaluate.call_args_list] == [2, 3]
    assert env.step.call_count == 3
    env.action_space.seed.assert_called_once_with(3710)
    env.close.assert_called_once()


if __name__ == '__main__':
    test_count_reset_and_final_evaluation()
    print('PASS: exact interaction counts, reset observation, final evaluation, action seed')
