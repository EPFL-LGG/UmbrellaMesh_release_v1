import sys
sys.path.append('..')
import umbrella_mesh
import elastic_rods
import numpy as np
import pytest
import finite_diff
import numpy.linalg as la
from test_constructor import construct_umbrella


def test_sphere_cap_umbrella():
	input_path = '../../data/sphere_cap_0.3.json.gz'
	curr_um = construct_umbrella(input_path)
	
	perturbation = np.random.uniform(-1e-3, 1e-3, curr_um.numDoF())
	dof = curr_um.getDoFs()
	curr_um.setDoFs(dof + perturbation)
	
	curr_um.updateSourceFrame()
	variableDesignParameters = True
	hessian = curr_um.hessian(variableDesignParameters = variableDesignParameters)
	hessian.reflectUpperTriangle()
	hessian = hessian.compressedColumn()
	n_dof = curr_um.numExtendedDoF() if variableDesignParameters else curr_um.numDoF()
	perturb = np.random.uniform(0, 1e-3, n_dof)
	input_vector = perturb
	# input_vector[var_indices[vj]] = perturb[var_indices[vj]]
	code_output = curr_um.applyHessian(input_vector, variableDesignParameters)
	matrix_output = hessian * input_vector
	# code_output = code_output[var_indices[vi]]
	# matrix_output = matrix_output[var_indices[vi]]
	error = la.norm(code_output - matrix_output) / la.norm(code_output)
	assert error < 1e-8
	
if __name__ == "__main__":
    test_sphere_cap_umbrella()