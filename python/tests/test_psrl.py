import sys
sys.path.append('..')
import umbrella_mesh
import elastic_rods
import numpy as np
import pytest
import finite_diff
import numpy.linalg as la
from test_constructor import construct_umbrella

def test_psrl_unit_umbrella():
    curr_um = construct_umbrella('../../data/sphere_cap_0.3.json.gz')
    
    for i in range(curr_um.numSegments()):
        rl = curr_um.segment(i).rod.restLengths()
        assert(np.isclose(np.min(rl), np.max(rl)))

if __name__ == "__main__":
    test_psrl_unit_umbrella()