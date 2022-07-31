from types import new_class
from matplotlib import cm
import vis
import numpy as np
from PIL import ImageColor
import umbrella_mesh

def get_color_field(obj, input_data):
    def new_colors(self):
        return self.data
    
    rod_colors = []
    for seg_id in range(obj.numSegments()):
        seg = obj.segment(seg_id)
        start, end = obj.joint(seg.startJoint), obj.joint(seg.endJoint)
        uid = None
        color = np.array((ImageColor.getcolor('#416788', "RGB")))/255.0
        for j, jid in zip([start, end], [seg.startJoint, seg.endJoint]):
            if j.jointPosType() != umbrella_mesh.JointPosType.Arm:
                uid = input_data['uid'][jid][0]
                break
        if input_data['color'][uid]: color = np.array((ImageColor.getcolor('#81D2C7', "RGB")))/255.0
        rod_colors.append(np.array([color]*obj.segment(seg_id).rod.numVertices()).astype(np.float64))
    sf = vis.fields.ScalarField(obj, rod_colors)
    sf.colors = new_colors.__get__(sf, vis.fields.ScalarField) 
    return sf

def get_rest_state_color_field(obj, input_data):
    def new_colors(self):
        return self.data
    
    rod_colors = []
    for seg_id in range(obj.numSegments()):
        seg = obj.segment(seg_id)
        start, end = obj.joint(seg.startJoint), obj.joint(seg.endJoint)
        uid = None
        color = np.array((ImageColor.getcolor('#416788', "RGB")))/255.0
        for j, jid in zip([start, end], [seg.startJoint, seg.endJoint]):
            if j.jointPosType() != umbrella_mesh.JointPosType.Arm:
                uid = input_data['uid'][jid][0]
                break
        if input_data['color'][uid]: color = np.array((ImageColor.getcolor('#416788', "RGB")))/255.0
        rod_colors.append(np.array([color]*obj.segment(seg_id).rod.numVertices()).astype(np.float64))
    sf = vis.fields.ScalarField(obj, rod_colors)
    sf.colors = new_colors.__get__(sf, vis.fields.ScalarField) 
    return sf

def get_scalar_field(obj, content_per_segment, rangeMin = None, rangeMax = None):
    if rangeMin is not None:
        rangeMin = np.stack(content_per_segment).min()
    if rangeMax is not None:
        rangeMax = np.stack(content_per_segment).max()
    sf = vis.fields.ScalarField(obj, content_per_segment, colormap=cm.plasma, vmin=rangeMin, vmax=rangeMax)
    return sf

def set_surface_view_options(view, color = None, surface_color = 'gray', umbrella_transparent = None, surface_transparent = True):
    view.viewOptions = {view.ViewType.LINKAGE: view.ViewOption(umbrella_transparent, color),
                        view.ViewType.SURFACE: view.ViewOption(surface_transparent, surface_color)}
    view.applyViewOptions()

def set_joint_vector_field(linkage, linkage_view, joint_vector_field):
    vector_field = [np.zeros((s.rod.numVertices(), 3)) for s in linkage.segments()]
    for ji in range(linkage.numJoints()):
        # seg_index = linkage.joint(ji).segment[0]
        seg_index = linkage.joint(ji).getSegmentAt(0)
        if linkage.segment(seg_index).startJoint == ji:
            vx_index = 0
        else:
            vx_index = -1
        vector_field[seg_index][vx_index] = joint_vector_field[ji]
    
    linkage_view.update(vectorField = vector_field)
def set_joint_scalar_field(linkage, linkage_view, joint_scalar_field):
    scalar_field = [np.zeros(s.rod.numVertices()) for s in linkage.segments()]
    for ji in range(linkage.numJoints()):
        seg_index = linkage.joint(ji).getSegmentAt(0)
        if linkage.joint(ji).getIsStartAt(0):
            vx_index = 0
        else:
            vx_index = -1
        scalar_field[seg_index][vx_index] = joint_scalar_field[ji]
    linkage_view.showScalarField(scalar_field)

def renderToFile(path, view, renderCam = None):
    orender = view.offscreenRenderer(width=2048, height=1200)
    if renderCam is not None:
        orender.setCameraParams(renderCam)
    orender.render()
    orender.save(path)