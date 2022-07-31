import sys
sys.path.append('..')
import umbrella_mesh
import elastic_rods
import linkage_vis
from umbrella_mesh import UmbrellaEnergyType

import numpy as np
from test_constructor import construct_umbrella

import py_newton_optimizer
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-10
OPTS.verbose = 1
OPTS.beta = 1e-6
OPTS.niter = 200
OPTS.verboseNonPosDef = False

def deploy_umbrella(input_path):
	curr_um = construct_umbrella(input_path)

	dof = curr_um.getDoFs()
	for i in range(curr_um.numJoints()):
	    if (curr_um.joint(i).jointType() == umbrella_mesh.JointType.X):
	        dof[curr_um.dofOffsetForJoint(i) + 6] = 1e-6
			
	curr_um.setDoFs(dof)
	curr_um.uniformDeploymentEnergyWeight = 1e-3
	curr_um.repulsionEnergyWeight = 0
	curr_um.attractionWeight = 0.001
	curr_um.setHoldClosestPointsFixed(False)
	curr_um.scaleInputPosWeights(0.5)
	curr_um.deploymentForceType = umbrella_mesh.DeploymentForceType.LinearActuator
	fixedVars = curr_um.rigidJointAngleDoFIndices()

	results = umbrella_mesh.compute_equilibrium(curr_um, callback = None, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)
	assert results.success

def test_one_umbrella():
	deploy_umbrella('../../data/sphere_cap_0.3.json.gz')

