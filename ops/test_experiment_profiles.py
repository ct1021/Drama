"""Ensure lightweight selection cannot silently alter unrelated training settings."""
import unittest
from experiment_profiles import apply_profile


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.raw = dict(Models=dict(WorldModel=dict(
            Encoder=dict(Mults=[1, 2, 4, 8], Depth=16),
            Decoder=dict(Mults=[1, 2, 4, 8], Depth=16),
            Mamba=dict(n_layer=2, ssm_cfg=dict(d_state=16)),
            HiddenStateDim=512),
            Agent=dict(hidden=256, AC=dict(Actor=dict(HiddenUnits=256), Critic=dict(HiddenUnits=512)))),
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

    def test_routing_profiles_only_add_routing_to_lightweight(self):
        for profile, mode in [('lightweight-readout', 'readout'), ('lightweight-routed', 'routed')]:
            result = apply_profile(self.raw, profile)
            self.assertEqual(result['Models']['WorldModel'].pop('TaskRouting'), mode)
            self.assertEqual(result, apply_profile(self.raw, 'lightweight'))

    def test_parameter_matched_profile_sets_only_explicit_width_and_route(self):
        result = apply_profile(self.raw, 'lightweight-routed-budgetmatched')
        self.assertEqual(result['Models']['WorldModel']['TaskRouting'], 'routed')
        self.assertEqual(result['Models']['WorldModel']['HiddenStateDim'], 640)
        self.assertEqual(result['Models']['Agent']['AC']['Actor']['HiddenUnits'], 192)
        self.assertEqual(result['Models']['WorldModel']['Encoder']['Mults'], [1, 2, 3, 4, 4])
        self.assertEqual(result['Models']['WorldModel']['Decoder']['Mults'], [1, 2, 3, 4, 4])
        self.assertEqual(self.raw['Models']['WorldModel'].get('HiddenStateDim'), 512)
        self.assertEqual(self.raw['Models']['Agent']['AC']['Actor']['HiddenUnits'], 256)

    def test_B_only_adds_loss_harmonization(self):
        result = apply_profile(self.raw, 'lightweight-harmonized')
        self.assertEqual(result['Models']['WorldModel'].pop('LossHarmonization'), 'rectified')
        self.assertEqual(result, apply_profile(self.raw, 'lightweight'))


if __name__ == '__main__':
    unittest.main()
