import sys
sys.path.append('..')
import umbrella_mesh
import elastic_rods
import linkage_vis
from umbrella_mesh import UmbrellaEnergyType
from bending_validation import suppress_stdout as so
from visualization_helper import *

import numpy as np

### Initial Deployment

name = 'sphere_cap_0.3_one_ring'

input_path = '../../data/{}.json'.format(name)
target_mesh_path = '../../data/target_meshes/{}.obj'.format('sphere_cap_0.3')

from helpers.deployment_helper import get_deployed_umbrella_mesh

from load_jsondata import read_data
input_data, io = read_data(filepath = input_path)
width = 2*input_data['arm_plate_edge_offset']
thickness = width * 0.5 # 1.5 mm # FIX from mm to meters everywhere
cross_section = [thickness, width]

curr_um = get_deployed_umbrella_mesh(input_path, target_mesh_path)

curr_um.energy(UmbrellaEnergyType.Full)

#### Pin Rigid Motion



use_pin = False

driver = curr_um.centralJoint()
jdo = curr_um.dofOffsetForJoint(driver)
fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()

import py_newton_optimizer
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-1
OPTS.verbose = 1
OPTS.beta = 1e-6
OPTS.niter = 100
OPTS.verboseNonPosDef = False

rod_colors = []
for ri in range(curr_um.numSegments()):
    rod_colors.append(np.ones(curr_um.segment(ri).rod.numVertices()) * ri)

import mesh
target_mesh = mesh.Mesh(target_mesh_path)
view = linkage_vis.LinkageViewerWithSurface(curr_um, target_mesh, width=1024, height=600)
# view = linkage_vis.LinkageViewer(curr_um, width=1024, height=600)
# view.update(scalarField = rod_colors)
set_surface_view_options(view, color = 'green', surface_color = 'gray', umbrella_transparent = False, surface_transparent = True)
view.averagedMaterialFrames = True

angles = []
def eqm_callback(prob, i):
    angles.append(curr_um.getDoFs()[curr_um.jointAngleDoFIndices()])
    if (i % 10 == 0):
        view.update()
        view.showScalarField(rod_colors)

curr_um.uniformDeploymentEnergyWeight = 0.1
# curr_um.deploymentForceType = umbrella_mesh.DeploymentForceType.Constant
curr_um.targetDeploymentHeight = thickness * 1
curr_um.repulsionEnergyWeight = 0
curr_um.attractionWeight = 100000
curr_um.setHoldClosestPointsFixed(True)
curr_um.scaleInputPosWeights(0.1)

curr_um.energyElastic(), curr_um.energyDeployment(), curr_um.energyRepulsion(), curr_um.energyAttraction()

angles = []
def eqm_callback(prob, i):
    angles.append(curr_um.getDoFs()[curr_um.jointAngleDoFIndices()])
    if (i % 10 == 0):
        view.update()
        view.showScalarField(rod_colors)

results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars)

results.success

### Initialize Design Optimization

import umbrella_optimization
import umbrella_optimization_finite_diff

optimizer = umbrella_optimization.UmbrellaOptimization(curr_um, target_mesh_path, OPTS, -1, False, fixedVars)

class UmbrellaOptimizationCallback:
    def __init__(self, optimizer, umbrella, view, update_color = False, no_surface = False, callback_freq = 1):
        self.optimizer     = optimizer
        self.umbrella       = umbrella
        self.view  = view
        self.update_color = update_color
        self.no_surface    = no_surface
        self.callback_freq = callback_freq
        self.iterateData = []

    def __call__(self):
        global prev_vars
        global prev_time_stamp
        #print('running callback', flush=True)

        if (self.no_surface):
            self.view.update()
            return
        #print('here1', flush=True)

        curr_vars = self.umbrella.getExtendedDoFsPARL()
        # Record values of all objective terms, plus timestamp and variables.
        idata = {t.name: t.term.value() for t in self.optimizer.objective.terms}
        idata.update({'iteration_time':   time.time() - prev_time_stamp,
                      'extendedDoFsPARL': curr_vars}) 
        idata.update({'{}_grad_norm'.format(t.name): get_component_gradient_norm(self.optimizer, t.type) for t in self.optimizer.objective.terms})
        self.iterateData.append(idata)
        prev_time_stamp = time.time()
        
        if self.linkage_view and (len(self.iterateData) % self.callback_freq == 0):
            if self.update_color:
                bottomColor =[79/255., 158/255., 246/255.]
                topColor =[0.5, 0.5, 0.5]
                heights = self.linkage.visualizationGeometryHeightColors()
                colors = np.take(np.array([bottomColor, topColor]), heights < heights.mean(), axis=0)
                self.view.showScalarField(rod_colors)
            else:
                pass
                self.view.update()

    def numIterations(self): return len(self.iterateData)

algorithm = umbrella_optimization.OptAlgorithm.NEWTON_CG
optimizer.set_holdClosestPointsFixed(False)
cb = UmbrellaOptimizationCallback(optimizer, curr_um, view, True, False, 1)

optimizer.getAttractionWeight()

solverStatus = optimizer.optimize(algorithm, 100, 1.0, 1e-2, cb, -1)


