import umbrella_mesh
import numpy as np
from matplotlib import pyplot as plt

def freeComponentNorm(g, fv):
    g[fv] = 0
    return np.linalg.norm(g)

class EquilibriumSolveAnalysis:
    def __init__(self, um):
        self.um = um
        self.uets = list(umbrella_mesh.UmbrellaEnergyType.__members__.items())
        self.energies = {name: [] for name, uet in self.uets}
        self.gradientNorms = {name: [] for name, uet in self.uets}
        self.angles = []
        self.plateHeights = []

    def record(self, prob):
        self.angles.append(self.um.getDoFs()[self.um.jointAngleDoFIndices()])
        self.plateHeights.append(self.um.plateHeights)
        fv = prob.fixedVars() + [b.idx for b in prob.activeBoundConstraints(prob.getVars(), prob.gradient())]
        for name, uet in self.uets:
            self.energies[name].append(self.um.energy(uet))
            self.gradientNorms[name].append(freeComponentNorm(self.um.gradient(False, uet), fv))

    def plotEnergies(self):
        for name, _ in self.uets:
            e = self.energies[name]
            if np.linalg.norm(e) == 0: continue
            plt.semilogy(e, label=name)
        x = np.arange(10)
        #plt.semilogy(x, self.energies['Elastic'][0] * 2.5**x, label='rate limit')
        plt.title('Potential Energy')
        plt.xlabel('Iterate')
        plt.legend()
        plt.grid()
        plt.tight_layout()

    def plotGradients(self):
        for name, _ in self.uets:
            g = self.gradientNorms[name]
            if np.linalg.norm(g) == 0: continue
            plt.semilogy(g, label=name)
        plt.title('Forces')
        plt.ylabel('Free Gradient Component')
        plt.xlabel('Iterate')
        plt.legend()
        plt.grid()
        plt.tight_layout()

    def plotAngles(self):
        plt.xlabel('Iterate')
        for i in range(len(self.angles[0])):
            plt.plot([a[i] / np.pi for a in self.angles])
        plt.title('Joint Angles')
        plt.xlabel('Iterate')
        plt.ylabel('Pi Radians')
        plt.grid()
        plt.tight_layout()

    def plotPlateHeights(self):
        plt.xlabel('Iterate')
        for i in range(len(self.plateHeights[0])):
            plt.plot([h[i] for h in self.plateHeights])
        plt.title('Plate Height')
        plt.xlabel('Iterate')
        plt.ylabel('Height')
        plt.grid()
        plt.tight_layout()

    def plot(self):
        plt.figure(figsize=(16,6))
        plt.subplot(1, 4, 1); self.plotEnergies()
        plt.subplot(1, 4, 2); self.plotGradients()
        plt.subplot(1, 4, 3); self.plotAngles()
        plt.subplot(1, 4, 4); self.plotPlateHeights()
        plt.tight_layout()
