import umbrella_optimization
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la

class DesignOptimizationAnalysis:
    def __init__(self, opt):
        self.opt = opt
        self.uets = list(umbrella_optimization.OptEnergyType.__members__.items())
        self.energies = {name: [] for name, uet in self.uets}
        self.unweightedEnergies = {name: [] for name, val in self.opt.objective.values().items()}
        self.gradientNorms = {name: [] for name, uet in self.uets}
        self.angles = []
        self.plateHeights = []
        self.armLengths = []

    def record(self):
        um = self.opt.committedObject
        self.angles.append(um.getDoFs()[self.opt.committedObject.jointAngleDoFIndices()])
        self.plateHeights.append(um.plateHeights)
        self.armLengths.append(um.getPerArmRestLength())
        for name, uet in self.uets:
            self.energies[name].append(self.opt.J(self.opt.params(), uet))
            self.gradientNorms[name].append(la.norm(self.opt.gradp_J(self.opt.params(), uet)))
        for name, val in self.opt.objective.values().items():
            self.unweightedEnergies[name].append(val)

    def plotEnergies(self, weighted=True):
        data = self.energies if weighted else self.unweightedEnergies
        for name, e in data.items():
            if np.linalg.norm(e) == 0: continue
            plt.semilogy(e, label=name)
        x = np.arange(10)
        #plt.semilogy(x, self.energies['Elastic'][0] * 2.5**x, label='rate limit')
        plt.title('Objective Value')
        plt.xlabel('Iterate')
        plt.legend()
        plt.grid()
        plt.tight_layout()

    def plotGradients(self):
        for name, _ in self.uets:
            g = self.gradientNorms[name]
            if np.linalg.norm(g) == 0: continue
            plt.semilogy(g, label=name)
        plt.title('Gradient Norms')
        plt.ylabel('Norms')
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
