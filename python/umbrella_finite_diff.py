from elastic_rods import EnergyType
from umbrella_mesh import UmbrellaEnergyType
from MeshFEM import sparse_matrices
import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt
import numpy.linalg as la
import inspect

class DesignOptimizationTermFDWrapper:
    def __init__(self, term, umbrella):
        self.term = term
        self.umbrella = umbrella
        self.term.update()

    def setDoFs(self, v):
        self.umbrella.setExtendedDoFsPARL(v)
        self.term.update()
        #self.umbrella.updateSourceFrame()

    def getDoFs(self):
        return self.umbrella.getExtendedDoFsPARL()

    def numDoF(self):          return self.umbrella.numExtendedDoFPARL()
    def energy(self):          return self.term.value()
    def gradient(self):        return self.term.grad()
    # def gradient(self):        return self.term.computeGrad()
    def applyHessian(self, v): return self.term.computeDeltaGrad(v)

# All functions here work with both Elastic Rod object and Rod umbrella object despite the parameter names.
def getVars(l, variableDesignParameters=False, perArmRestLen=False):
    if perArmRestLen:
        return l.getExtendedDoFsPARL()
    if (variableDesignParameters):
        return l.getExtendedDoFs()
    return l.getDoFs()

def setVars(l, dof, variableDesignParameters=False, perArmRestLen=False):
    if perArmRestLen:
        return l.setExtendedDoFsPARL(dof)
    if (variableDesignParameters):
        return l.setExtendedDoFs(dof)
    return l.setDoFs(dof)

def energyAt(l, dof, umbrellaEnergyType, etype = EnergyType.Full, variableDesignParameters=False, perArmRestLen=False, restoreDoF=True):
    if restoreDoF: prevDoF = getVars(l, variableDesignParameters, perArmRestLen)
    setVars(l, dof, variableDesignParameters, perArmRestLen)
    energy = guardedEval(l.energy, umbrellaEnergyType = umbrellaEnergyType, energyType = etype)
    if restoreDoF: setVars(l, prevDoF, variableDesignParameters, perArmRestLen)
    return energy

# Pybind11 methods/funcs apparently don't support `inspect.signature`,
# but at least their arg names are guaranteed to appear in the docstring... :(
def hasArg(func, argName):
    if (func.__doc__ is not None):
        return argName in func.__doc__
    return argName in inspect.signature(func).parameters

def guardedEval(func, *args, **kwargs):
    '''
    Evaluate `func`, on the passed arguments, filtering out any unrecognized keyword arguments.
    '''
    return func(*args, **{k: v for k, v in kwargs.items() if hasArg(func, k)})

def gradientAt(l, dof, umbrellaEnergyType, etype = EnergyType.Full, variableDesignParameters=False, perArmRestLen=False, updatedSource=False, restoreDoF=True):
    if restoreDoF: prevDoF = getVars(l, variableDesignParameters, perArmRestLen)
    setVars(l, dof, variableDesignParameters, perArmRestLen)
    geval = l.gradientPerArmRestlen if perArmRestLen else l.gradient
    g = guardedEval(geval, updatedSource=updatedSource, umbrellaEnergyType=umbrellaEnergyType, energyType=etype, variableDesignParameters=variableDesignParameters)
    if restoreDoF: setVars(l, prevDoF, variableDesignParameters, perArmRestLen)
    return g

def fd_gradient_test(obj, stepSize, umbrellaEnergyType, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perArmRestLen=False, precomputedAnalyticalGradient=None, x=None, restoreDoF=True, updatedSource=False):
    if (x is None):
        x = getVars(obj, variableDesignParameters, perArmRestLen)
    if (direction is None): direction = np.random.uniform(-1, 1, x.shape)
    step = stepSize * direction

    an = precomputedAnalyticalGradient
    if (an is None):
        if perArmRestLen:
            grad = guardedEval(obj.gradientPerArmRestlen, updatedSource=updatedSource, umbrellaEnergyType = umbrellaEnergyType, energyType=etype, variableDesignParameters=variableDesignParameters)
        else:
            grad = guardedEval(obj.gradient, updatedSource=updatedSource, umbrellaEnergyType = umbrellaEnergyType, energyType=etype, variableDesignParameters=variableDesignParameters)
        an = np.dot(direction, grad)

    energyPlus  = energyAt(obj, x + step, umbrellaEnergyType, etype, variableDesignParameters, perArmRestLen, restoreDoF=False)
    energyMinus = energyAt(obj, x - step, umbrellaEnergyType, etype, variableDesignParameters, perArmRestLen, restoreDoF=False)
    if restoreDoF:
        setVars(obj, x, variableDesignParameters, perArmRestLen)
    # print("energy plus minus ", energyPlus, energyMinus)
    return [(energyPlus - energyMinus) / (2 * stepSize), an]

def gradient_convergence(umbrella, minStepSize=1e-12, maxStepSize=1e-2, umbrellaEnergyType = UmbrellaEnergyType.Full, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perArmRestLen=False, updatedSource=False):
    origDoF = getVars(umbrella, variableDesignParameters, perArmRestLen)

    if (direction is None): direction = np.random.uniform(-1, 1, origDoF.shape)

    epsilons = np.logspace(np.log10(minStepSize), np.log10(maxStepSize), 100)
    errors = []

    an = None
    for eps in epsilons:
        fd, an = fd_gradient_test(umbrella, eps, umbrellaEnergyType = umbrellaEnergyType, etype=etype, direction=direction, variableDesignParameters = variableDesignParameters, perArmRestLen=perArmRestLen, precomputedAnalyticalGradient=an, restoreDoF=False, x=origDoF, updatedSource=updatedSource)
        err = np.abs((an - fd) / an)
        # print("fd validation ", an, fd)
        errors.append(err)

    setVars(umbrella, origDoF, variableDesignParameters, perArmRestLen)

    return (epsilons, errors, an)

def gradient_convergence_plot(umbrella, minStepSize=1e-12, maxStepSize=1e-2, umbrellaEnergyType = UmbrellaEnergyType.Full, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perArmRestLen=False, plot_name=None, updatedSource=False):
    eps, errors, ignore = gradient_convergence(umbrella, minStepSize, maxStepSize, umbrellaEnergyType, etype, direction, variableDesignParameters, perArmRestLen, updatedSource)
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()
    if (plot_name is not None): plt.savefig(plot_name, dpi = 300)


def fd_hessian_test(obj, stepSize, umbrellaEnergyType, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perArmRestLen=False, infinitesimalTransportGradient=False, hessianVectorProduct=False, x=None, precomputedAnalyticalHessVec=None, restoreDoF=True):
    if (x is None):
        x = getVars(obj, variableDesignParameters, perArmRestLen)
    if (direction is None): direction = np.random.uniform(-1, 1, x.shape)

    an = precomputedAnalyticalHessVec
    if (an is None):
        # Use hessian-vector product if requested, or if it's all we have.
        if hessianVectorProduct or not callable(getattr(obj, 'hessian', None)):
            an = guardedEval(obj.applyHessian, v=direction, energyType=etype, variableDesignParameters=variableDesignParameters)
        else:
            hessEval = obj.hessianPerArmRestlen if perArmRestLen else obj.hessian
            h = guardedEval(hessEval, umbrellaEnergyType = umbrellaEnergyType, energyType=etype, variableDesignParameters=variableDesignParameters)
            h.reflectUpperTriangle()
            H = csc_matrix(h.compressedColumn())
            an = H * direction

    gradPlus  = gradientAt(obj, x + stepSize * direction, umbrellaEnergyType = umbrellaEnergyType, etype = etype, variableDesignParameters=variableDesignParameters, perArmRestLen=perArmRestLen,  updatedSource=infinitesimalTransportGradient, restoreDoF=False)
    gradMinus = gradientAt(obj, x - stepSize * direction, umbrellaEnergyType = umbrellaEnergyType, etype = etype, variableDesignParameters=variableDesignParameters, perArmRestLen=perArmRestLen,  updatedSource=infinitesimalTransportGradient, restoreDoF=False)

    if restoreDoF:
        setVars(obj, x, variableDesignParameters, perArmRestLen)

    return [(gradPlus - gradMinus) / (2 * stepSize), an]

def fd_hessian_test_relerror_max(umbrella, stepSize, umbrellaEnergyType, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perArmRestLen=False, hessianVectorProduct=False):
    fd, an = fd_hessian_test(umbrella, stepSize, umbrellaEnergyType, etype, direction, variableDesignParameters, perArmRestLen, hessianVectorProduct = hessianVectorProduct)
    relErrors = np.nan_to_num(np.abs((fd - an) / an), 0.0)
    idx = np.argmax(relErrors)
    return (idx, relErrors[idx], fd[idx], an[idx])

def fd_hessian_test_relerror_norm(umbrella, stepSize, umbrellaEnergyType, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perArmRestLen=False, infinitesimalTransportGradient=False, hessianVectorProduct=False, x=None, precomputedAnalyticalHessVec=None, restoreDoF=True):
    fd, an = fd_hessian_test(umbrella, stepSize, umbrellaEnergyType, etype, direction, variableDesignParameters, perArmRestLen, infinitesimalTransportGradient=infinitesimalTransportGradient, hessianVectorProduct = hessianVectorProduct, x=x, precomputedAnalyticalHessVec=precomputedAnalyticalHessVec, restoreDoF=restoreDoF)
    return [norm(fd - an) / norm(an), an]

def hessian_convergence(umbrella, minStepSize=1e-12, maxStepSize=1e-2, umbrellaEnergyType = UmbrellaEnergyType.Full, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perArmRestLen=False, infinitesimalTransportGradient=False, hessianVectorProduct=False, nsteps=40):
    origDoF = getVars(umbrella, variableDesignParameters, perArmRestLen)
    if (direction is None): direction = np.random.uniform(-1, 1, origDoF.shape)

    epsilons = np.logspace(np.log10(minStepSize), np.log10(maxStepSize), nsteps)

    an = None
    errors = []
    for eps in epsilons:
        err, an = fd_hessian_test_relerror_norm(umbrella, eps, umbrellaEnergyType = umbrellaEnergyType, etype=etype, direction=direction, variableDesignParameters=variableDesignParameters, perArmRestLen=perArmRestLen, infinitesimalTransportGradient=infinitesimalTransportGradient, hessianVectorProduct=hessianVectorProduct, x=origDoF, precomputedAnalyticalHessVec=an, restoreDoF=False)
        errors.append(err)

    setVars(umbrella, origDoF, variableDesignParameters, perArmRestLen)

    return (epsilons, errors)

def hessian_convergence_plot(umbrella, minStepSize=1e-12, maxStepSize=1e-2, umbrellaEnergyType = UmbrellaEnergyType.Full, etype=EnergyType.Full, direction=None, variableDesignParameters=False, perArmRestLen=False, infinitesimalTransportGradient=False, plot_name='hessian_validation.png', hessianVectorProduct=False, nsteps=40):
    from matplotlib import pyplot as plt
    eps, errors = hessian_convergence(umbrella, minStepSize, maxStepSize, umbrellaEnergyType, etype, direction, variableDesignParameters, perArmRestLen=perArmRestLen, infinitesimalTransportGradient=infinitesimalTransportGradient, hessianVectorProduct=hessianVectorProduct, nsteps=nsteps)
    plt.title('Directional derivative fd test for hessian')
    plt.ylabel('Relative error')
    plt.xlabel('Step size')
    plt.loglog(eps, errors)
    plt.grid()
    if (plot_name is not None): plt.savefig(plot_name, dpi = 300)

def block_error(umbrella, var_indices, va, vb, grad, umbrellaEnergyType, etype=EnergyType.Full, eps=1e-6, perturb=None, variableDesignParameters=False, perArmRestLen=False, hessianVectorProduct=False):
    '''
    Report the error in the (va, vb) block of the Hessian, where
    va and vb are members of the `var_types` array.
    '''
    if (perturb is None):
        perturb = np.random.normal(0, 1, len(grad))
    block_perturb = np.zeros_like(perturb)
    block_perturb[var_indices[vb]] = perturb[var_indices[vb]]
    fd_delta_grad, an_delta_grad = fd_hessian_test(umbrella, eps, umbrellaEnergyType = umbrellaEnergyType, etype=etype, direction=block_perturb, variableDesignParameters=variableDesignParameters, perArmRestLen=perArmRestLen, hessianVectorProduct=hessianVectorProduct)
    fd_delta_grad = fd_delta_grad[var_indices[va]]
    an_delta_grad = an_delta_grad[var_indices[va]]
    return (la.norm(an_delta_grad - fd_delta_grad) / la.norm(an_delta_grad),
            fd_delta_grad, an_delta_grad)

def hessian_convergence_block_plot(umbrella, var_types, var_indices, umbrellaEnergyType, etype=EnergyType.Full, variableDesignParameters=False, perArmRestLen=False, plot_name='rod_umbrella_hessian_validation.png', hessianVectorProduct=False, perturb = None):
    # The perArmRestLen flag should take priority over the variableDesignParameter since the perArmRestLen automatically assume design parameter exists (in particular the rest length exists).
    geval = umbrella.gradientPerArmRestlen if perArmRestLen else umbrella.gradient
    grad = guardedEval(geval, updatedSource=True, umbrellaEnergyType=umbrellaEnergyType, energyType=etype, variableDesignParameters=variableDesignParameters)
    if (perturb is None):
        perturb = np.random.normal(0, 1, len(grad))
    numVarTypes = len(var_types) - 1
    epsilons = np.logspace(np.log10(1e-12), np.log10(1e2), 50)
    fig = plt.figure(figsize=(16, 12))
    for i, vi in enumerate(var_types[1:]):
        for j, vj in enumerate(var_types[1:]):
            plt.subplot(numVarTypes, numVarTypes, i * numVarTypes + j + 1)
            errors = [block_error(umbrella, var_indices, vi, vj, grad, umbrellaEnergyType, etype, eps, perturb, variableDesignParameters, perArmRestLen, hessianVectorProduct=hessianVectorProduct)[0] for eps in epsilons]
            plt.loglog(epsilons, errors)
            plt.title(f'({vi}, {vj}) block')
            plt.grid()
            plt.tight_layout()
    plt.savefig(plot_name, dpi = 300)
    plt.show()
