
import os

from reprep.plot_utils.axes import turn_all_axes_off

from complete_image_pipeline.image_simulation import SimulationData
# XXX: should not include tests
from complete_image_pipeline_tests.synthetic import simulate_image
import duckietown_utils as dtu
from duckietown_utils.cli import D8App
from easy_algo import get_easy_algo_db
from ground_projection.ground_projection_interface import get_ground_projection
import numpy as np

from .maps import FAMILY_SEGMAPS, _plot_map_segments


class DisplayTileAndMaps(D8App):
    """ 
        Displays the segment maps.
    """
    usage = """
    
Use as follows:

    $ rosrun complete_image_pipeline display_segmaps [pattern]

For example:

    $ rosrun complete_image_pipeline display_segmaps 'DT17*'
    
    """

    def define_program_options(self, params):
        params.accept_extra()
    
    def go(self):
        out = 'out-maps'
        extra = self.options.get_extra()
        
        if len(extra) == 0:
            query = '*' 
        else:
            query = extra

        db = get_easy_algo_db()
        maps = list(db.query_and_instance(FAMILY_SEGMAPS, query))

        self.debug('maps: %s' % maps)
        for id_map in maps:
            display_map(id_map, out)
            
def display_map(id_map, out):
    dtu.logger.info('id_map == %s' % id_map)
    db = get_easy_algo_db()
    smap = db.create_instance(FAMILY_SEGMAPS, id_map)
    texture_png = get_texture(smap, dpi=600)
    fn = os.path.join(out, '%s-texture.png' % (id_map))
    dtu.write_data_to_file(texture_png, fn)
    
    simdata = simulate_camera_view(smap)
    
    fn = os.path.join(out, '%s-view.jpg' % (id_map))
    dtu.write_bgr_to_file_as_jpg(simdata.rectified_synthetic_bgr, fn)
    fn = os.path.join(out, '%s-segments.jpg' % (id_map))
    dtu.write_bgr_to_file_as_jpg(simdata.rectified_segments_bgr, fn)

@dtu.contract(returns='str', dpi=int)
def get_texture(smap, dpi):
    figure_args=dict(figsize=(2,2), facecolor='green')
    a = dtu.CreateImageFromPylab(dpi=dpi, figure_args=figure_args)
    frames = list(set(_.id_frame for _ in smap.points.values()))
    id_frame = frames[0]
#     print('frames: %s choose %s' % (frames, id_frame))
    with a as pylab:
        _plot_map_segments(smap, pylab, id_frame, plot_ref_segments=False)
        pylab.axis('equal')
        turn_all_axes_off(pylab)
        pylab.tight_layout()
    png = a.get_png()
    return png

@dtu.contract(returns=SimulationData)
def simulate_camera_view(sm, robot_name = 'shamrock'):
    gp = get_ground_projection(robot_name)
    # GroundProjectionGeometry
    gpg = gp.get_ground_projection_geometry() 
    
    pose = dtu.geo.SE2_from_translation_angle([0,-0.05],-np.deg2rad(-5))
    res = simulate_image(sm, pose, gpg, blur_sigma=0.3)
    return res

        
    
