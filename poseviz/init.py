import vtk
from mayavi import mlab
from tvtk.api import tvtk


def initialize(offscreen=False):
    mlab.options.offscreen = offscreen
    global figure
    fig = mlab.figure('PoseViz', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1280, 720))
    fig.scene.renderer.render_window.set(alpha_bit_planes=1, multi_samples=0, full_screen=False)
    # fig.scene.renderer.render_window.set_
    fig.scene.renderer.set(
        use_depth_peeling=True, maximum_number_of_peels=5, occlusion_ratio=0.01)
    fig.scene._tool_bar.setVisible(False)
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    output = vtk.vtkFileOutputWindow()
    output.SetFileName('/dev/null')
    vtk.vtkOutputWindow().SetInstance(output)
    vtk.vtkObject.GlobalWarningDisplayOff()
    fig.scene.camera.clipping_range = [0.1, 1000]
    fig.scene.anti_aliasing_frames = 0
    return fig
