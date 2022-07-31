import sys
sys.path.append('..')
import umbrella_mesh
import elastic_rods
import numpy as np
import pytest
import finite_diff
from load_jsondata import read_data
from test_constructor import construct_umbrella
def fd_hessian(curr_um):
	
	curr_um.uniformDeploymentEnergyWeight = 0.01
	curr_um.repulsionEnergyWeight = 0.01
	curr_um.attractionWeight = 0.01

	perturbation = np.random.uniform(-1e-5, 1e-5, curr_um.numDoF())
	dof = curr_um.getDoFs()
	curr_um.setDoFs(dof + perturbation)
	curr_um.updateSourceFrame()
	(_, errors) = finite_diff.hessian_convergence(curr_um)
	assert np.min(errors) < 1e-8

def hessian_test(input_path, deploymentForceType=umbrella_mesh.DeploymentForceType.Spring):
	curr_um = construct_umbrella(input_path)
	curr_um.deploymentForceType = deploymentForceType
	curr_um.setHoldClosestPointsFixed(False)
	fd_hessian(curr_um)	
	curr_um.setHoldClosestPointsFixed(True)
	fd_hessian(curr_um)	

	
def test_hessian():
	hessian_test('../../data/sphere_cap_0.3.json.gz')
	hessian_test('../../data/sphere_cap_0.3.json.gz', deploymentForceType=umbrella_mesh.DeploymentForceType.Constant)
	hessian_test('../../data/hemisphere_5t.json.gz')