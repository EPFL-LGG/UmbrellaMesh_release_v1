import sys
sys.path.append('../..')
import umbrella_mesh
import elastic_rods
import linkage_vis
from umbrella_mesh import UmbrellaEnergyType
from bending_validation import suppress_stdout as so
from visualization_helper import *

import numpy as np

def get_deployed_umbrella_mesh(input_path, target_mesh_path):

    from load_jsondata import read_data
    input_data, io = read_data(filepath = input_path)
    width = 2*input_data['arm_plate_edge_offset']
    thickness = width * 0.5 # 1.5 mm # FIX from mm to meters everywhere
    cross_section = [thickness, width]

    ### Initialization

    # curr_um = umbrella_mesh.UmbrellaMesh(io)
    curr_um = umbrella_mesh.UmbrellaMesh(target_mesh_path, io)
    curr_um.setMaterial(elastic_rods.RodMaterial('rectangle', 2000, 0.3, cross_section, stiffAxis=elastic_rods.StiffAxis.D1))
    curr_um.energy(UmbrellaEnergyType.Full)
    ### Pin Rigid Motion
    use_pin = False

    driver = curr_um.centralJoint()
    jdo = curr_um.dofOffsetForJoint(driver)
    fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()

    ### Equilibrium solve

    import py_newton_optimizer
    OPTS = py_newton_optimizer.NewtonOptimizerOptions()
    OPTS.gradTol = 1e-2
    OPTS.verbose = 1
    OPTS.beta = 1e-6
    OPTS.niter = 100
    OPTS.verboseNonPosDef = False

    dof = curr_um.getDoFs()
    for i in range(curr_um.numJoints()):
        if (curr_um.joint(i).jointType() == umbrella_mesh.JointType.X):
            dof[curr_um.dofOffsetForJoint(i) + 6] = 1e-3
    curr_um.setDoFs(dof)

    curr_um.uniformDeploymentEnergyWeight = 0.1
    # curr_um.deploymentForceType = umbrella_mesh.DeploymentForceType.Constant
    curr_um.targetDeploymentHeight = thickness * 2
    curr_um.repulsionEnergyWeight = 0
    curr_um.attractionWeight = 1000
    curr_um.setHoldClosestPointsFixed(False)
    curr_um.scaleInputPosWeights(0.5)

    curr_um.energyElastic(), curr_um.energyDeployment(), curr_um.energyRepulsion(), curr_um.energyAttraction()

    results = umbrella_mesh.compute_equilibrium(curr_um, callback = None, options = OPTS, fixedVars = fixedVars)
    return curr_um