import sys
sys.path.append('..')
import umbrella_mesh
from test_constructor import construct_umbrella
import umbrella_optimization
def test_umbrella_optimization_constructor():
    input_path = '../../data/sphere_cap_0.3.json.gz'
    curr_um = construct_umbrella(input_path)

    use_pin = False

    driver = curr_um.centralJoint()
    jdo = curr_um.dofOffsetForJoint(driver)
    fixedVars = (list(range(jdo, jdo + 6)) if use_pin else []) + curr_um.rigidJointAngleDoFIndices()

    import py_newton_optimizer
    OPTS = py_newton_optimizer.NewtonOptimizerOptions()
    OPTS.gradTol = 1e-2
    OPTS.verbose = 1
    OPTS.beta = 1e-6
    OPTS.niter = 1
    OPTS.verboseNonPosDef = False

    optimizer = umbrella_optimization.UmbrellaOptimization(curr_um, OPTS, 2.5, -1, False, fixedVars)
    rest_height_optimizer = umbrella_optimization.UmbrellaRestHeightsOptimization(optimizer)
    _ = umbrella_optimization.UmbrellaSingleRestHeightOptimization(rest_height_optimizer)

if __name__ == "__main__":
	test_umbrella_optimization_constructor()