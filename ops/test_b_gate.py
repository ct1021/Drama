import json,tempfile,unittest
from pathlib import Path
from b_gate import completed_100k,short_training_gate


class BGateTests(unittest.TestCase):
    def test_only_complete_100k_ten_episodes_is_accepted(self):
        s=dict(phase='completed',step=100000)
        e=[dict(training_step=100000,episodes=10,returns=[1.]*10)]
        self.assertTrue(completed_100k(s,e))
        for bad in [dict(phase='training',step=100000),dict(phase='failed',step=100000),dict(phase='completed',step=99999)]:
            self.assertFalse(completed_100k(bad,e))
        self.assertFalse(completed_100k(s,[]))
        e[0]['returns'][0]=float('nan');self.assertFalse(completed_100k(s,e))

    def setup_run(self,root,speed):
        (root/'status.json').write_text(json.dumps(dict(phase='completed',step=1800,latest={'ActorCritic/total_loss':1})))
        (root/'evaluations.jsonl').write_text(json.dumps(dict(training_step=1800,episodes=2))+'\n')
        rows=[dict(tag='WorldModel/total_loss',step=i,time=i*speed,value=1.) for i in range(1200,1701)]
        rows += [dict(tag='WorldModel/Harmony/'+name+'_log_scale',step=1700,value=.1) for name in ('image','reward','kl')]
        (root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return dict(profile='lightweight',phase='completed',step=1800,seconds_per_interaction=.2),dict(ok=True,world_model_parameters=7161606,cache_bytes_per_row_float32=164864,identical_base_initialization=True,scalar_optimizer_coverage=True)

    def test_efficiency_gate_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as path:
            p=Path(path);reference,gpu=self.setup_run(p,.22)
            self.assertTrue(short_training_gate(p,reference,gpu)['passed'])
            self.setup_run(p,.24)
            self.assertFalse(short_training_gate(p,reference,gpu)['passed'])

    def test_nonfinite_and_missing_scalar_update_fail(self):
        with tempfile.TemporaryDirectory() as path:
            p=Path(path);reference,gpu=self.setup_run(p,.2)
            with (p/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(tag='bad',step=1800,value=float('inf')))+'\n')
            with self.assertRaises(AssertionError):short_training_gate(p,reference,gpu)
            self.setup_run(p,.2)
            text=(p/'metrics.jsonl').read_text().replace('"value": 0.1','"value": 0.0')
            (p/'metrics.jsonl').write_text(text)
            with self.assertRaises(AssertionError):short_training_gate(p,reference,gpu)


if __name__=='__main__':unittest.main()
