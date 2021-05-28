import mayavi.mlab as mlab
import numpy as np


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig

def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig



def draw_points_with_labels(pts, labels):

	fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0), engine=None, size=(600, 600))
	mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.squeeze(labels/np.max(labels)), mode='sphere',
												   					  scale_factor=0.03, scale_mode='none',
												   					  figure=fig)
	
	fig = draw_multi_grid_range(fig, bv_range=(-40, -40, 80, 40))
	mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
	mlab.show(stop=False)


def draw_points_without_labels(pts):

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0), engine=None, size=(600, 600))
    mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                                                                      scale_factor=1, 
                                                                      figure=fig)
    
    fig = draw_multi_grid_range(fig, bv_range=(-40, -40, 80, 40))
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    mlab.show(stop=False)
