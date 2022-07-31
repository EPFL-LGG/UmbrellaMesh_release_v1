import sys
sys.path.append('..')
import umbrella_mesh
import elastic_rods
import numpy as np
import pytest
import finite_diff
import numpy.linalg as la


from load_jsondata import read_data
from test_constructor import construct_umbrella

def test_randomly_perturb_linkage():
	curr_um = construct_umbrella('../../data/sphere_cap_0.3.json.gz')
	
	curr_um.uniformDeploymentEnergyWeight = 1e-3
	curr_um.repulsionEnergyWeight = 1e-3


	curr_um.setDoFs(curr_um.getDoFs() + np.random.uniform(-1e-3, 1e-3, curr_um.numDoF()))
	curr_um.updateSourceFrame()

	(_, errors, _) = finite_diff.gradient_convergence(curr_um)
	assert np.min(errors) < 1e-7

def test_randomly_perturb_linkage_constant_forces():
	curr_um = construct_umbrella('../../data/sphere_cap_0.3.json.gz')
	
	curr_um.uniformDeploymentEnergyWeight = 1e-3
	curr_um.deploymentForceType = umbrella_mesh.DeploymentForceType.Constant
	curr_um.repulsionEnergyWeight = 1e-3


	curr_um.setDoFs(curr_um.getDoFs() + np.random.uniform(-1e-3, 1e-3, curr_um.numDoF()))
	curr_um.updateSourceFrame()

	(_, errors, _) = finite_diff.gradient_convergence(curr_um)
	assert np.min(errors) < 1e-7

def test_randomly_perturb_linkage_attraction():
	curr_um = construct_umbrella('../../data/sphere_cap_0.3.json.gz')
	
	# curr_um.uniformDeploymentEnergyWeight = 1e-3
	curr_um.attractionWeight = 1e-3
	curr_um.scaleInputPosWeights(0.1)
	# curr_um.repulsionEnergyWeight = 1e-3
	curr_um.setHoldClosestPointsFixed(True)


	curr_um.setDoFs(curr_um.getDoFs() + np.random.uniform(-1e-6, 1e-6, curr_um.numDoF()))
	curr_um.updateSourceFrame()

	(_, errors, _) = finite_diff.gradient_convergence(curr_um)
	assert np.min(errors) < 1e-6

	curr_um.setHoldClosestPointsFixed(False)
	curr_um.updateSourceFrame()

	(_, errors, _) = finite_diff.gradient_convergence(curr_um)
	assert np.min(errors) < 1e-6
