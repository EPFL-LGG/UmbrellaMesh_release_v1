import umbrella_mesh
from bending_validation import suppress_stdout as so
from load_jsondata import read_data
import numpy as np

def parse_input(input_path, handleBoundary = False, resolution = 10, handlePivots = True):
    input_data, io = read_data(filepath = input_path, handleBoundary = handleBoundary, handlePivots = handlePivots)
    import mesh
    target_mesh = mesh.Mesh(input_data['target_v'], input_data['target_f'])

    curr_um = umbrella_mesh.UmbrellaMesh(io, resolution)
    thickness = io.material_params[6]
    target_height_multiplier = input_data['target_spacing_factor']
    return io, input_data, target_mesh, curr_um, thickness, target_height_multiplier

def configure_umbrella_pre_deployment(curr_um, thickness, target_height_multiplier):
    curr_um.deploymentForceType = umbrella_mesh.DeploymentForceType.LinearActuator
    curr_um.targetDeploymentHeight = thickness * target_height_multiplier
    curr_um.repulsionEnergyWeight = 0
    curr_um.attractionWeight = 1e-1
    curr_um.setHoldClosestPointsFixed(False)
    curr_um.scaleInputPosWeights(0.99999)

    curr_um.angleBoundEnforcement = umbrella_mesh.AngleBoundEnforcement.Penalty

def break_input_angle_symmetry(curr_um):
    dof = curr_um.getDoFs()
    for i in range(curr_um.numJoints()):
        # if (curr_um.joint(i).jointType() == umbrella_mesh.JointType.X):
        dof[curr_um.dofOffsetForJoint(i) + 6] = 1e-3
    curr_um.setDoFs(dof)

def insert_randomness(curr_um, zPerturbationEpsilon = 1e-4):
    dof = np.array(curr_um.getDoFs())
    zCoordDoFs = np.array(curr_um.jointPositionDoFIndices())[2::3]
    dof[zCoordDoFs] += 2 * zPerturbationEpsilon * (np.random.random_sample(len(zCoordDoFs)) - 0.5)
    curr_um.setDoFs(dof)

def staged_deployment(curr_um, weights, eqm_callback, OPTS, fixedVars, elasticEnergyIncreaseFactorLimit = 2.5):
    for weight in weights:
        curr_um.uniformDeploymentEnergyWeight = weight
        with so(): results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=elasticEnergyIncreaseFactorLimit)
    return results

def configure_umbrella_optimization(curr_um, bdryMultiplier = 1.0):
    # ### Initialize Design Optimization
    curr_um.uniformDeploymentEnergyWeight = 1e0
    curr_um.repulsionEnergyWeight = 0
    curr_um.attractionWeight = 1e-3
    curr_um.setHoldClosestPointsFixed(True) # Don't let closest point's drift away from reasonable values with the low attraction weight.
    curr_um.scaleInputPosWeights(0.1, bdryMultiplier)


def configure_umbrella_true_equlibrium(curr_um, thickness, target_height_multiplier):
    curr_um.uniformDeploymentEnergyWeight = 1e0
    curr_um.targetDeploymentHeight = thickness * target_height_multiplier 
    curr_um.attractionWeight = 0

def configure_umbrella_undeployment_step_one(curr_um, thickness, target_height_multiplier, undeployment_multiplier = 10):
    curr_um.uniformDeploymentEnergyWeight = 1e0
    curr_um.targetDeploymentHeight = thickness * target_height_multiplier * undeployment_multiplier
    curr_um.attractionWeight = 0

def configure_umbrella_undeployment_step_two(curr_um):
    curr_um.uniformDeploymentEnergyWeight = 0
    curr_um.attractionWeight = 0

def configure_design_optimization_umbrella(uo):
    uo.equilibriumOptimizer.options.verbose = 1
    #uo.equilibriumOptimizer.options.verboseWorkingSet = True
    uo.equilibriumOptimizer.options.gradTol = 1e-10
    # Hold the closest points fixed in the target-attraction term of the equilibrium solve:
    # this seems to make the design optimization much more robust.
    uo.setHoldClosestPointsFixed(True, False)