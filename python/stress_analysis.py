import linkage_vis, vis
import mesh
from matplotlib import cm
from visualization_helper import *
from ipywidgets import HBox
import matplotlib.pyplot as plt
class stressFieldVisualization():
    def __init__(self, linkage, target_mesh):
        self.initial_stresses = linkage.maxVonMisesStresses()
        vals = np.array(self.initial_stresses).flatten()
        self.vmin, self.vmax = vals.min(), vals.max()
        self.target_mesh = target_mesh
        self.initialField = vis.fields.ScalarField(linkage, self.initial_stresses, colormap = cm.plasma, vmin= self.vmin, vmax = self.vmax)
        self.initialView = linkage_vis.LinkageViewerWithSurface(linkage, self.target_mesh, scalarField=self.initialField)
        set_surface_view_options(self.initialView, surface_color = 'gray', surface_transparent = True)
        self.initialView.averagedMaterialFrames = True
    
    def showInitial(self):
        return self.initialView.show()
    
    def show(self):
        return HBox([self.initialView.show(), self.optimView.show()])
        
        
    def getView(self, linkage):
        stresses = linkage.maxVonMisesStresses()
        stressField = vis.fields.ScalarField(linkage, stresses, colormap = cm.plasma, vmin= self.vmin, vmax = self.vmax)
        self.optimView = linkage_vis.LinkageViewerWithSurface(linkage, self.target_mesh, scalarField=stressField)
        set_surface_view_options(self.optimView, surface_color = 'gray', surface_transparent = True)
        self.optimView.averagedMaterialFrames = True
        return self.optimView

def plotStressDistributions(linkages, names, vmin, vmax):
    fig, axs = plt.subplots(1, len(linkages), figsize = (len(linkages)*10, 10))
    for i, linkage in enumerate(linkages):
        axs[i].hist(np.array(linkage.maxVonMisesStresses()).flatten(), 20, range = [vmin, vmax])
        axs[i].set_title(names[i], fontsize = 20.0)
    plt.show()