"""
Prof. Francisco Hernán Ortega Culaciati
ortega.francisco@u.uchile.cl
Departamento de Geofísica - FCFM
Universidad de Chile

April 07, 2018


"""
import numpy as NP
from matplotlib.patches import Ellipse


def pyvelo(ax, x, y, vx, vy, svx=None, svy=None, rhoxy=None, Nsigma=1,
           scale=1, arrow_prop={}):
    """
    plots a vector field with error ellipses in the matplolib axis ax.
    WARNING: USING THIS FUNCTION IS OK FOR CARTESIAN COORDINATES OR GEODESIC COORDINATES
             WITH A PLOT USING PlateCarree()
    scale: factor to convert vx, vy units into the units of x, and y)
           if scale = 1, vx, vy are drawn in x,y units.

    rhoxy: indicates correlation between errors svx and svy, for the moment is a dummy
           parameter. TODO: calculate angle of rotation and semiaxis of error ellipses in
                            the principal reference frame.

    Arrow properties are defined as:
        arrow_width = arrow_prop['width']
        arrow_head_width =  arrow_prop['head_width_factor'] * arrow_width
        # the following factor must be between 0 and 1
        arrow_head_length_factor =  arrow_prop['head_length_factor']
        (arrow head length is a factor of the arrow length, default = 0.2)
        arrow_face_color = arrow_prop['face_color']
        arrow_edge_color = arrow_prop['edge_color']
        ellipse_color = arrow_prop['ellipse_color']
        ellipse_width = arrow_prop['ellipse_width']
    """
    # make sure we receive arrays
    x = NP.atleast_1d(x)
    y = NP.atleast_1d(y)
    vx = NP.atleast_1d(vx)
    vy = NP.atleast_1d(vy)
    if svx is not None:
        svx = NP.atleast_1d(svx)
    if svy is not None:
        svy = NP.atleast_1d(svy)
    if rhoxy is not None:
        rhoxy = NP.atleast_1d(rhoxy)

    # DEFINE ARROW GEOMETRIC PROPERTIES AND COLOR PROPERTIES
    # ARROW WIDTH
    if 'width' not in arrow_prop.keys():
        arrow_width = 0.01
    else:
        arrow_width = arrow_prop['width']

    # ARROW HEAD WIDTH
    if 'head_width_factor' not in arrow_prop.keys():
        head_width_factor = 10
        arrow_head_width = head_width_factor * arrow_width

    else:
        head_width_factor = arrow_prop['head_width_factor']
        arrow_head_width = head_width_factor * arrow_width

    # ARROW HEAD LENGTH FACTOR
    # (ARROW HEAD LENGTH IS DEFINED AS A FRACTION OF THE ARROW LENGTH)
    # this factor must be between 0 and 1
    if 'head_length_factor' not in arrow_prop.keys():
        arrow_head_length = 1.5 * arrow_head_width
    else:
        arrow_head_length = arrow_prop['head_length_factor'] * arrow_head_width

    # ARROW COLOR
    # FACE
    if 'face_color' not in arrow_prop.keys():
        arrow_face_color = 'k'
    else:
        arrow_face_color = arrow_prop['face_color']
    # EDGE
    if 'edge_color' not in arrow_prop.keys():
        arrow_edge_color = arrow_face_color
    else:
        arrow_edge_color = arrow_prop['edge_color']

    # ELLIPSE PROPERTIES
    # ELLIPSE COLOR
    if 'ellipse_color' not in arrow_prop.keys():
        ellipse_color = arrow_edge_color
    else:
        ellipse_color = arrow_prop['ellipse_color']
    # ELLIPSE LINE WIDTH
    if 'ellipse_width' not in arrow_prop.keys():
        ellipse_width = 0.3
    else:
        ellipse_width = arrow_prop['ellipse_width']

    # in order to avoid errors when plotting, i need to avoid plotting
    # arrows with zero length. (this is because the length_includes_head = True option)
    # so I only plot arrows whose norm is larger than a minimum value
    V_plot_norm = scale * NP.sqrt(vx ** 2 + vy ** 2)
    # get the indices in which V_plot_norm is larger than a threshold
    threshold = NP.finfo(float).eps * 1E6
    I_points2plot = NP.where(V_plot_norm > threshold)[0]

    Scale_HEAD = NP.power( V_plot_norm / NP.max(V_plot_norm), 0.35)


    # NOW WE HAVE DEFINED THE properties of the arrows, do the plots.
    for k in I_points2plot:
        ax.arrow(x[k], y[k], vx[k] * scale, vy[k] * scale,
                 width=arrow_width * Scale_HEAD[k],
                 head_width=head_width_factor * arrow_width * Scale_HEAD[k], #head_width=arrow_head_width * Scale_HEAD[k]
                 overhang=0.1,  # define the shape of the tip
                 shape='full',
                 head_length=arrow_head_length * Scale_HEAD[k],
                 facecolor=arrow_face_color,
                 edgecolor=arrow_edge_color,
                 head_starts_at_zero=False,
                 length_includes_head=True)

    # FIRST PLOT ALL THE ELLIPSES, SO THESE ARE BENEATH THE ARROWS
    if (svx is not None) and (svy is not None):
        for k in I_points2plot:
            ## TODO: Need to add option for correlated errors !!!!
            # HERE ASSUMING rho = 0, so errors are in principal coordinates
            psv1 = svx
            psv2 = svy
            angle_to_principal_axis = 0  # TODO: determine this from Rho if available.
            # setup the ellipse
            xy = NP.array([x[k] + vx[k] * scale, y[k] + vy[k] * scale])
            e = Ellipse(xy=xy,
                        width=2 * Nsigma * psv1[k] * scale,
                        height=2 * Nsigma * psv2[k] * scale,
                        angle=angle_to_principal_axis)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_facecolor('none')
            e.set_edgecolor(ellipse_color)
            e.set_linewidth(ellipse_width)

