"""Explicit architecture profiles; no implicit change to the public baseline."""
import copy

PROFILES = {
    'public': 'upstream YAML four-stage encoder; not paper-exact DramaXS',
    'lightweight': 'five-stage AE [1,2,3,4,4]; original Mamba2/DFS/AC; head_dim64; not paper-exact DramaXS',
}


def apply_profile(raw, profile):
    if profile not in PROFILES:
        raise ValueError('Unknown experiment profile: ' + profile)
    result = copy.deepcopy(raw)
    if profile == 'lightweight':
        for component in ('Encoder', 'Decoder'):
            result['Models']['WorldModel'][component]['Mults'] = [1, 2, 3, 4, 4]
    return result
