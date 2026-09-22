"""Explicit architecture profiles; no implicit change to the public baseline."""
import copy

PROFILES = {
    'public': 'upstream YAML four-stage encoder; not paper-exact DramaXS',
    'lightweight': 'five-stage AE [1,2,3,4,4]; original Mamba2/DFS/AC; head_dim64; not paper-exact DramaXS',
    'lightweight-readout': 'C: lightweight AE; shared Mamba2 recurrence; task-restricted prediction readouts',
    'lightweight-routed': 'D: lightweight AE; H1-R1 block-triangular recurrence and partition norms; same readouts as C',
    'lightweight-routed-budgetmatched': 'H1-R1 budget-matched candidate: HiddenStateDim=640 and actor width 192; total model budget within 24 parameters of A',
    'lightweight-harmonized': 'B: original lightweight architecture; three rectified loss weights; HarmonyDream adaptation',
}


def apply_profile(raw, profile):
    if profile not in PROFILES:
        raise ValueError('Unknown experiment profile: ' + profile)
    result = copy.deepcopy(raw)
    if profile.startswith('lightweight'):
        for component in ('Encoder', 'Decoder'):
            result['Models']['WorldModel'][component]['Mults'] = [1, 2, 3, 4, 4]
    if profile in ('lightweight-readout', 'lightweight-routed', 'lightweight-routed-budgetmatched'):
        result['Models']['WorldModel']['TaskRouting'] = 'routed' if profile.endswith('budgetmatched') else profile.removeprefix('lightweight-')
    if profile == 'lightweight-routed-budgetmatched':
        result['Models']['WorldModel']['HiddenStateDim'] = 640
        result['Models']['Agent']['AC']['Actor']['HiddenUnits'] = 192
    if profile == 'lightweight-harmonized':
        result['Models']['WorldModel']['LossHarmonization'] = 'rectified'
    return result
