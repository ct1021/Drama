"""Ensure lightweight selection cannot silently alter unrelated training settings."""
import unittest
from experiment_profiles import apply_profile


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.raw = dict(Models=dict(WorldModel=dict(
            Encoder=dict(Mults=[1, 2, 4, 8], Depth=16),
            Decoder=dict(Mults=[1, 2, 4, 8], Depth=16),
            Mamba=dict(n_layer=2, ssm_cfg=dict(d_state=16))), Agent=dict(hidden=256)),
            JointTrainAgent=dict(BatchSize=16, BatchLength=128, ImagineBatchSize=1024))

    def test_public_unchanged_and_independent(self):
        result = apply_profile(self.raw, 'public')
        self.assertEqual(result, self.raw)
        result['Models']['WorldModel']['Encoder']['Mults'][0] = 999
        self.assertEqual(self.raw['Models']['WorldModel']['Encoder']['Mults'][0], 1)

    def test_lightweight_only_changes_two_mult_lists(self):
        result = apply_profile(self.raw, 'lightweight')
        for component in ('Encoder', 'Decoder'):
            self.assertEqual(result['Models']['WorldModel'][component]['Mults'], [1, 2, 3, 4, 4])
            result['Models']['WorldModel'][component]['Mults'] = [1, 2, 4, 8]
        self.assertEqual(result, self.raw)

    def test_unknown_profile_fails(self):
        with self.assertRaises(ValueError):
            apply_profile(self.raw, 'typo')


if __name__ == '__main__':
    unittest.main()
