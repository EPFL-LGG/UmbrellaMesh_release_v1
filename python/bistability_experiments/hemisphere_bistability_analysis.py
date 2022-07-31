#!/usr/bin/env python
# coding: utf-

#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('..')
import umbrella_mesh
import elastic_rods
import linkage_vis
from umbrella_mesh import UmbrellaEnergyType
from bending_validation import suppress_stdout as so
from visualization_helper import *

import pipeline_helper, importlib, design_optimization_analysis
with so(): importlib.reload(pipeline_helper)
with so(): importlib.reload(design_optimization_analysis)

from pipeline_helper import UmbrellaOptimizationCallback, allEnergies, allGradientNorms, allDesignObjectives, allDesignGradientNorms, set_joint_vector_field, show_center_joint_normal, show_joint_normal

from design_optimization_analysis import DesignOptimizationAnalysis

import umbrella_optimization
import umbrella_optimization_finite_diff
from umbrella_optimization import OptEnergyType

import numpy as np
import numpy.linalg as la

import pickle, gzip

from configuration import *

# ### Initializatio

name = 'hemisphere_5t'
input_path = '../../data/{}.json.gz'.format(name)

io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path)
# target_height_multiplier = 1

# curr_um = pickle.load(gzip.open('../../output/saddle_5t_optimized_equilibrium_2022_01_20_15_01_target_height_factor_5.0.pkl.gz', 'r'))


# #### Pin Rigid Motio

use_pin = False

driver = curr_um.centralJoint()
jdo = curr_um.dofOffsetForJoint(driver)
fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()

curr_um = pickle.load(gzip.open('output/hemisphere_5t/2022_01_21_16_12/equilibrium_at_step_0.47500000000000003_target_height_factor_5.0.pkl.gz'))

import py_newton_optimizer
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1
OPTS.beta = 1e-6
OPTS.niter = 300
OPTS.verboseNonPosDef = False

rod_colors = get_color_field(curr_um, input_data)

# lview = linkage_vis.LinkageViewer(curr_um, width=1024, height=600)
# lview.update(scalarField = rod_colors)
# lview.show()

import mesh
view = linkage_vis.LinkageViewerWithSurface(curr_um, target_mesh, width=1024, height=600)
set_surface_view_options(view, color = 'green', surface_color = 'gray', umbrella_transparent = False, surface_transparent = True)
view.averagedMaterialFrames = True
view.showScalarField(rod_colors)
view.show()

view.getCameraParams()

view.getSize()

from equilibrium_solve_analysis import EquilibriumSolveAnalysis
eqays = EquilibriumSolveAnalysis(curr_um)
def eqm_callback(prob, i):
    eqays.record(prob)
    if (i % 2 == 0):
        view.showScalarField(rod_colors)

configure_umbrella_pre_deployment(curr_um, thickness, target_height_multiplier)

allGradientNorms(curr_um)

break_input_angle_symmetry(curr_um)

view.showScalarField(rod_colors)

results = staged_deployment(curr_um, np.logspace(-3, 0, 4), eqm_callback, OPTS, fixedVars)

results.success

eqays.plot()

# curr_um = pickle.load(gzip.open('../../output/hemisphere_5t_optimized_equilibrium_2022_01_21_15_32_target_height_factor_5.0.pkl.gz'))


# ### Initialize Design Optimizatio

configure_umbrella_optimization(curr_um)

import py_newton_optimizer
opt_opts = py_newton_optimizer.NewtonOptimizerOptions()
opt_opts.gradTol = 1e-8
opt_opts.verbose = 10
opt_opts.beta = 1e-6
opt_opts.niter = 600
opt_opts.verboseNonPosDef = False

results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = opt_opts, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

opt_opts.niter = 50

results.success

optimizer = umbrella_optimization.UmbrellaOptimization(curr_um, opt_opts, 2.5, -1, False, fixedVars)

optimizer.beta = 1 * 1e6
optimizer.gamma = 1
optimizer.eta = 0
optimizer.zeta = 0# 1e1
optimizer.iota = 1 * 1e10

allDesignObjectives(optimizer)

allDesignGradientNorms(optimizer)

optimizer.objective.terms[-1].term.normalActivationThreshold = -2e-5

optimizer.objective.terms[-1].term.normalWeight = 1
optimizer.objective.terms[-1].term.tangentialWeight = 0
optimizer.objective.terms[-1].term.torqueWeight = 0


# ### Force Analysi

import force_analysis
with so(): importlib.reload(force_analysis)

force_analysis.UmbrellaForceAnalysis(curr_um)
v1 = force_analysis.UmbrellaForceFieldVisualization(curr_um)
v1.show()


# ### Run Optimizatio

rest_height_optimizer = umbrella_optimization.UmbrellaRestHeightsOptimization(optimizer)
single_rest_height_optimizer = umbrella_optimization.UmbrellaSingleRestHeightOptimization(rest_height_optimizer)

rest_height_optimizer.newPt(rest_height_optimizer.params())

original_design_parameters = rest_height_optimizer.params()

doptays = DesignOptimizationAnalysis(rest_height_optimizer)
def eqm_callback(prob, i):
    eqays.record(prob)
    if (i % 2 == 0):
        view.showScalarField(rod_colors)

import time
pipeline_helper.prev_time_stamp = time.time()

uo = rest_height_optimizer.get_parent_opt()

uo.equilibriumOptimizer.options.verbose = 1
#uo.equilibriumOptimizer.options.verboseWorkingSet = True
uo.equilibriumOptimizer.options.gradTol = 1e-10
# Hold the closest points fixed in the target-attraction term of the equilibrium solve:
# this seems to make the design optimization much more robust.
uo.setHoldClosestPointsFixed(True, False)

tfview = pipeline_helper.TargetFittingVisualization(curr_um, uo.target_surface_fitter, view)
cb = pipeline_helper.UmbrellaOptimizationCallback(rest_height_optimizer, view, True, False, 1, rod_colors, doptays.record, tfview=tfview)

algorithm = umbrella_optimization.OptAlgorithm.NEWTON_CG
#algorithm = umbrella_optimization.OptAlgorithm.BFGS
solverStatus = umbrella_optimization.optimize(rest_height_optimizer, algorithm, 1000, 0.005, 1e-5, cb, input_data["plate_edge_length"] / 30 * 32)

rest_height_optimizer.beta, rest_height_optimizer.gamma, rest_height_optimizer.eta

import time

from matplotlib import pyplot as plt
doptays.plot()

force_analysis.UmbrellaForceAnalysis(curr_um)
v2 = force_analysis.UmbrellaForceFieldVisualization(curr_um)
v2.show()


# ### Get true equilibrium stat

use_pin = True

driver = curr_um.centralJoint()
jdo = curr_um.dofOffsetForJoint(driver)
fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()

configure_umbrella_true_equlibrium(curr_um, thickness, target_height_multiplier)

allEnergies(curr_um)

OPTS.niter = 600

results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

results.success

import pickle 
import gzip
import time
pickle.dump(curr_um, gzip.open('../../output/{}_optimized_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(name, time.strftime("%Y_%m_%d_%H_%M"), target_height_multiplier), 'w'))
# load_um = pickle.load(gzip.open('test_pickle_um.pkl.gz', 'r'))

# from load_jsondata import update_optimized_json
# update_optimized_json(input_path, rest_height_optimizer.params(), output_json_path = '../../output/{}_optimized_params_{}.json'.format(name, time.strftime("%Y_%m_%d_%H_%M")), optim_spacing_factor = target_height_multiplier)



# from load_jsondata import write_deformed_config

# write_deformed_config(curr_um, input_path, output_path = '../../output/{}_optimized_rendering_output_{}.json.gz'.format(name, time.strftime("%Y_%m_%d_%H_%M"), write_stress = False, is_rest_state = False))

deployed_heights = curr_um.getUmbrellaHeights()


# ### Undeploymen

import configuration
importlib.reload(configuration)
from configuration import *

configure_umbrella_undeployment_step_one(curr_um, thickness, target_height_multiplier, 5)

allEnergies(curr_um)

OPTS.niter = 30

results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)
results.success

configure_umbrella_undeployment_step_two(curr_um)

allEnergies(curr_um)

OPTS.niter = 500

results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)
results.success

initial_heights = curr_um.getUmbrellaHeights()

import pickle 
import gzip

pickle.dump(curr_um, gzip.open('../../output/{}_optimized_rest_state_equilibrium_{}_target_height_factor_{}.pkl.gz'.format(name, time.strftime("%Y_%m_%d_%H_%M"), target_height_multiplier), 'w'))

# write_deformed_config(curr_um, input_path, output_path = '../../output/{}_optimized_rest_state_rendering_output_{}.json.gz'.format(name, time.strftime("%Y_%m_%d_%H_%M")), write_stress = False, is_rest_state = True)


# ### Deploymen

use_pin = False

driver = curr_um.centralJoint()
jdo = curr_um.dofOffsetForJoint(driver)
fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()
configure_umbrella_pre_deployment(curr_um, thickness, target_height_multiplier)

break_input_angle_symmetry(curr_um)

view.showScalarField(rod_colors)

results = staged_deployment(curr_um, np.logspace(-3, 0, 4), eqm_callback, OPTS, fixedVars)

results.success


# ### Bistability Analysi

# curr_um = pickle.load(gzip.open("../../output/hemisphere_5t_optimized_equilibrium_2022_01_20_20_05_target_height_factor_5.0.pkl.gz"))

pos_steps = np.linspace(0, 1, 41)

pos_steps

neg_steps = pos_steps[1:4] * -1

steps =  list(np.flip(neg_steps)) + list(pos_steps)

initial_heights *  -0.075 + deployed_heights * (1 -  -0.075) - thickness

thickness

evals = []

curr_um.attractionWeight = 0

curr_um.uniformDeploymentEnergyWeight = 1e0

use_pin = True

driver = curr_um.centralJoint()
jdo = curr_um.dofOffsetForJoint(driver)
fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()

from load_jsondata import update_optimized_json
from load_jsondata import write_deformed_config

import os

time_stamp = time.strftime("%Y_%m_%d_%H_%M")
if not os.path.exists('{}/{}/{}'.format('output', name, time_stamp)):
    os.makedirs('{}/{}/{}'.format('output', name, time_stamp))  

def write_data(step):
    pickle.dump(curr_um, gzip.open('output/{}/{}/equilibrium_at_step_{}_target_height_factor_{}.pkl.gz'.format(name, time_stamp, step, target_height_multiplier), 'w'))
# load_um = pickle.load(gzip.open('test_pickle_um.pkl.gz', 'r'))
    write_deformed_config(curr_um, input_path, output_path = 'output/{}/{}/equilibrium_at_step_{}_rendering_output.json.gz'.format(name, time_stamp, step), write_stress = False, is_rest_state = False)

for i, step in enumerate(steps):
    heights = initial_heights * step + deployed_heights * (1 - step)
    curr_um.targetDeploymentHeightVector = heights
    with so(): results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)
#     if i % 8 == 0:
    write_data(step)
    evals.append(curr_um.energyElastic())
    print("the equilibrium solve for step {} is {}. The elastic energy is {}".format(step, 'successful' if results.success else 'not successful', curr_um.energyElastic()))

import matplotlib.pyplot as plt

evals.reverse()

from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(40, evals[40], color = cm.tab10(1), s= 50)

plt.scatter(0, evals[0], color = cm.tab10(6), s= 50)
plt.scatter(4, evals[4], color = cm.tab10(6), s= 50)
plt.scatter(21, evals[21], color = cm.tab10(6), s= 50)
plt.scatter(32, evals[32], color = cm.tab10(6), s= 50)



plt.plot(evals, '-o', zorder = 0)
plt.savefig('{}/bistability_analysis.svg'.format('{}/{}/{}'.format('output', name, time_stamp)))

np.save('{}/bistability_analysis_energy'.format('{}/{}/{}'.format('output', name, time_stamp)), evals)

selected_index = [-1, -5, -22, -33, -41]

selected_steps = ['1.0', '0.9', '0.47500000000000003', '0.2', '0.0']

curr_um = pickle.load(gzip.open('output/hemisphere_5t/2022_01_21_16_12/equilibrium_at_step_{}_target_height_factor_5.0.pkl.gz'.format(step)))

for step in selected_steps:
    curr_um = pickle.load(gzip.open('output/hemisphere_5t/2022_01_21_16_12/equilibrium_at_step_{}_target_height_factor_5.0.pkl.gz'.format(step)))
    with open('umbrella_connectivity_{}.obj'.format(step), 'w') as f:
        for i in range(curr_um.numJoints()):
            v = curr_um.joint(i).position
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for i in range(curr_um.numSegments()):
            s = curr_um.segment(i)
            f.write('l {} {}\n'.format(s.startJoint + 1, s.endJoint + 1))

