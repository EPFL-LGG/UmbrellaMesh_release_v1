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

import pickle, gzip

from configuration import *

import py_newton_optimizer
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1
OPTS.beta = 1e-6
OPTS.niter = 300
OPTS.verboseNonPosDef = False

def get_vibrational_mode_views():

	name = 'hemisphere_5t'
	input_path = '../../data/{}.json.gz'.format(name)

	io, input_data, target_mesh, curr_um, thickness, target_height_multiplier = parse_input(input_path, handleBoundary=False, handlePivots = True)
	rod_colors = get_color_field(curr_um, input_data)

	def get_saddle_view(umbrella):
	    import mesh
	    view = linkage_vis.LinkageViewerWithSurface(umbrella, target_mesh, width=1024, height=600)
	    set_surface_view_options(view, color = 'green', surface_color = 'gray', umbrella_transparent = False, surface_transparent = True)
	    view.averagedMaterialFrames = True
	    view.setCameraParams(((-11.144512556513881, -11.281582930201559, 4.961758097391448),
	                             (0.2578692282155091, 0.4699669630554463, 0.8441768267229215),
	                             (4.75386274113031, -3.697285366936124, -4.116992725883979)))
	    view.showScalarField(rod_colors)
	    return view

	curr_um = pickle.load(gzip.open("../../output/hemisphere_5t_optimized_rest_state_equilibrium_2022_04_25_11_08_target_height_factor_5.0.pkl.gz", 'r'))

	use_pin = False

	driver = curr_um.centralJoint()
	jdo = curr_um.dofOffsetForJoint(driver)
	fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()
	view = get_saddle_view(curr_um)
	view.show()
	break_input_angle_symmetry(curr_um)

	def eqm_callback(prob, i):
	    if (i % 1 == 0):
	        view.showScalarField(rod_colors)

	configure_umbrella_pre_deployment(curr_um, thickness, target_height_multiplier)

	results = staged_deployment(curr_um, np.logspace(-4, 0, 5), eqm_callback, OPTS, fixedVars, elasticEnergyIncreaseFactorLimit = 1.5)


	import compute_vibrational_modes
	class ModalAnalysisWrapper:
	    def __init__(self, um):
	        self.um = um
	    def hessian(self):
	        return self.um.hessian(umbrellaEnergyType=umbrella_mesh.UmbrellaEnergyType.Full)
	    def massMatrix(self): return self.um.massMatrix()
	    def lumpedMassMatrix(self): return self.um.lumpedMassMatrix()


	configure_umbrella_pre_deployment(curr_um, thickness, 5)
	# curr_um.attractionWeight = 1e-3
	curr_um.attractionWeight = 0
	OPTS.niter = 500

	with so(): results = umbrella_mesh.compute_equilibrium(curr_um, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

	results.success

	lambdas, modes = compute_vibrational_modes.compute_vibrational_modes(ModalAnalysisWrapper(curr_um), fixedVars=curr_um.rigidJointAngleDoFIndices(), mtype=compute_vibrational_modes.MassMatrixType.FULL, n=16, sigma=-1e-6)

	import mode_viewer, importlib
	importlib.reload(mode_viewer);
	mview_5 = mode_viewer.ModeViewer(curr_um, modes, lambdas, amplitude=0.5 / lambdas[6])
	# mview_5.showScalarField(rod_colors)


	# Target separation = 2 * thickness

	import copy 
	um_2 = copy.deepcopy(curr_um)


	configure_umbrella_pre_deployment(um_2, thickness, 2)
	# um_2.attractionWeight = 1e-3
	um_2.attractionWeight = 0
	OPTS.niter = 500

	with so(): results = umbrella_mesh.compute_equilibrium(um_2, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

	results.success

	lambdas, modes = compute_vibrational_modes.compute_vibrational_modes(ModalAnalysisWrapper(um_2), fixedVars=um_2.rigidJointAngleDoFIndices(), mtype=compute_vibrational_modes.MassMatrixType.FULL, n=16, sigma=-1e-6)

	import mode_viewer, importlib
	importlib.reload(mode_viewer);
	mview_2 = mode_viewer.ModeViewer(um_2, modes, lambdas, amplitude=0.5 / lambdas[6])
	# mview_2.showScalarField(rod_colors)


	# Target separation = 10 * thickness

	um_10 = copy.deepcopy(curr_um)


	configure_umbrella_pre_deployment(um_10, thickness, 10)
	# um_10.attractionWeight = 1e-3
	um_10.attractionWeight = 0
	OPTS.niter = 500

	with so(): results = umbrella_mesh.compute_equilibrium(um_10, callback = eqm_callback, options = OPTS, fixedVars = fixedVars, elasticEnergyIncreaseFactorLimit=2.5)

	results.success

	lambdas, modes = compute_vibrational_modes.compute_vibrational_modes(ModalAnalysisWrapper(um_10), fixedVars=um_10.rigidJointAngleDoFIndices(), mtype=compute_vibrational_modes.MassMatrixType.FULL, n=16, sigma=-1e-6)

	import mode_viewer, importlib
	importlib.reload(mode_viewer);
	mview_10 = mode_viewer.ModeViewer(um_10, modes, lambdas, amplitude=0.5 / lambdas[6])
	# mview_10.showScalarField(rod_colors)

	mview_5.selectMode(6, play=True)
	mview_2.selectMode(6, play=True)
	mview_10.selectMode(6, play=True)
	mview_2.mode_selector.value = mview_2.mode_selector.options[6]
	mview_5.mode_selector.value = mview_5.mode_selector.options[6]
	mview_10.mode_selector.value = mview_10.mode_selector.options[6]
	return mview_5, mview_2, mview_10