"""
This module contains code used for visualizing the geometries
used by the simulator. Each geometry is a seperate hdf5 file.
"""

import h5py
import sys
import os
import numpy as np
import vtk

class GeometryVisualizer(object):
    """Class for visualizing a 3D geometry stored as a h5 file.

    The geometry is read from a single h5 file, vtk-compatible 
    subfiles are then generated for each spatial domain and boundary.
    Users can then specify which domains and boundaries to show in 
    the visualization.

    The spatial domains are designated by the following identificators
        0: cytosol
        1: cleft
        2: junctional sr 
        3: non-junctional sr
        4: t-tubule
    """
    def __init__(self, geometry_file, style=None, vizpath='.visualize/'):
        """Take a .h5 geometry file and load domain and boundary data.

        :param geometry_file: path to the h5 geometry file to visualize 
        :param vizpath: where to save domain/boundary subfiles
        """
        self.vizpath = vizpath
        self.style = style if style else self.default_style()
        self.imgpath = None

        if not os.path.isfile(geometry_file):
            raise IOError("File '{}' not found.".format(geometry_file))
        try:
            data = h5py.File(geometry_file)
        except IOError:
            e = "Unable to open file '{}', is it a h5 file?"
            raise IOError(e.format(geometry_file))
        try:
            self.voxels = data['domains']['voxels'].value
            self.boundaries = data['boundaries']
        except KeyError:
            error = ("Geometry file '{}' not set up correctly. Domains and/or"
                     " boundary missing".format(geometry_file))
            raise KeyError(error)

    def render(self, azimuth=210, roll=0, elevation=20):
        """Create vtk-compatible files and vtk objects."""
        # Set up domain and boundary subfiles
        self.domain_ids = np.unique(self.voxels)
        for domain_id in self.domain_ids:
            if domain_id in self.style['domains']:
                    self.write_domain_file(domain_id)
        for key in self.boundaries.keys():
            if key in self.style['boundaries']:
                self.write_boundary_file(key)

        # Set up all vtk objects
        self.create_renderer()
        self.create_domain_actors()
        self.create_boundary_actors()
        if self.style['bbox']:
            self.create_bbox()
        
        # Prettify 
        self.renderer.SetAmbient(0.0,1.0,1.0)
        self.renderer.SetBackground(1, 1, 1)  # Background color white
        self.renderWindow.SetSize(1700, 1800)
        self.renderWindow.Render()
        cam = self.renderer.GetActiveCamera()
        cam.Azimuth(azimuth)
        cam.Roll(roll)
        cam.Elevation(elevation)
        cam.Zoom(1.1)
        self.renderWindow.Render()

        # Save a png of the geometry if wanted
        if self.imgpath is not None:
            w2i = vtk.vtkWindowToImageFilter()
            writer = vtk.vtkPNGWriter()
            w2i.SetInput(self.renderWindow)
            w2i.Update()
            writer.SetInputConnection(w2i.GetOutputPort())

            if len(self.imgpath) < 4 or self.imgpath[-4:] != '.png':
                self.imgpath += '.png'
            writer.SetFileName(self.imgpath)
            self.renderWindow.Render()
            writer.Write()
        
        self.renderWindowInteractor.Start()

    def default_style(self):
        """Default style to be used if none is given by user.

        Only jSR, nSR and RyR are shown by default.
        """
        domains = [2, 3]
        boundaries = ['ryr']
        labels = ['ryr']
        colors = {0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (0,0.5,0.5), 4:(0,0,1),
                  'serca': (1,1,0), 'sr_cleft': (0,1,1), 'ryr': (1,0,1),
                  'lcc': (0,0,0)}
        opacities = {0: 0.01, 1: 0.1, 2: 1.0, 3: 1.0, 4: 1.0}
        return {'domains': domains,
                'boundaries': boundaries,
                'colors': colors,
                'opacities': opacities,
                'labels': labels,
                'bbox': False}

    def create_renderer(self):
        """Set up a vtk renderer."""
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        
    def create_domain_actors(self):
        """Set up the actors of the domains that are to be visualized
        and add them to the renderer.
        """
        for d in self.domain_ids:
            if d in self.style['domains']:
                self.create_domain_actor(d)

    def create_boundary_actors(self):
        for key in self.boundaries.keys():
            if key in self.style['boundaries']:
                self.create_boundary_actor(key)
        
    def write_domain_file(self, domain_id):
        """Creates a .vtk file for the given spatial domain.

        Each point for the given spatial domain is written into a 
        .vtk file along with metadata.

        :param domain_id: integer in the range [0, 4].
        """
        with open(self.vizpath+'type{}.vtk'.format(domain_id), 'w') as outfile:
            outfile.write('# vtk DataFile Version 3.0\n' \
                          'vtk ouput\nASCII\nDATASET POLYDATA\n')
            idx = np.nonzero(self.voxels == domain_id)
            N = idx[0].shape[0] # Number of points

            outfile.write('POINTS %d float\n' % N)
            # Write coordinates to file
            for i in range(N):
                outfile.write('%f %f %f\n' % (idx[0][i]+0.5, idx[1][i]+0.5, idx[2][i]+0.5))

    def write_boundary_file(self, key):
        """Write information about the given boundary to a .bnd file.

        The boundary information in the h5 is specified by point-pairs
        (p1, p2), where the flux points from p1 to p2. We calculate the 
        centerpoint between these and save it to the boundary file 
        alongside which direction the flux points.

        :param key: indentifier of the boundary, which is also its key
                    in the h5 file.
        """
        boundaries = self.boundaries[key].value
        
        with open(self.vizpath + key + '.bnd', 'w') as outfile:
            for i in range(boundaries.shape[0]):
                p1 = boundaries[i, :3] + 0.5
                p2 = boundaries[i, 3:] + 0.5
                p = (p1 + p2)/2. # Center point
                d = abs(p2 - p1).argmax() # Dimension of change
                outfile.write('{p[0]:f} {p[1]:f} {p[2]:f} '\
                              '{d:d}\n'.format(p=p, d=d))


    def create_domain_actor(self, domain_id):
        """Initialize a vtk actor for given spatial domain.

        :param domain_id: which domain to visualize
        """
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.vizpath+'type{}.vtk'.format(domain_id))
        reader.Update()

        vertexFilter = vtk.vtkGlyph3D()
        box = vtk.vtkCubeSource()
        vertexFilter.SetSourceConnection(box.GetOutputPort())
        vertexFilter.SetSourceConnection(box.GetOutputPort())
        vertexFilter.SetInputConnection(reader.GetOutputPort())
        vertexFilter.Update()

        # Create a mapper and actor for smoothed dataset
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertexFilter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        actor.GetProperty().SetColor(self.style['colors'][domain_id])
        if domain_id in self.style['opacities']:
            actor.GetProperty().SetOpacity(self.style['opacities'][domain_id])
        self.renderer.AddActor(actor)

    def create_boundary_actor(self, key):
        """Create a boundary actor for the given boundary.

        :param key: name of boundary
        :param color: which color to use 
        :param labal: boolean, label each 'voxel' of the boundary?
        """
        boundaries = np.loadtxt(self.vizpath+key+'.bnd')
        if boundaries.shape[0] == 0:
            return
        # check shape
        if len(boundaries.shape) == 1:
            boundaries = boundaries.reshape(1,4)
    
        centers = boundaries[:,:3]
        direction = boundaries[:,-1]
        N = direction.shape[0]

        for i in range(N):
            box = vtk.vtkCubeSource()
            box.SetCenter(centers[i,:])
            dim = int(direction[i])
            f = 0.1
            pos = centers[i,:]
            pos[0] += 0.6
            pos[1] -= 0.2
            h = -0.1

            # Increment the correct dimension
            [box.SetXLength, box.SetYLength, box.SetZLength][dim](f)
            pos[dim] += h

            # Set up mapper and actor objects
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(box.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.style['colors'][key])
            if key in self.style['opacities']:
                actor.GetProperty().SetOpacity(self.style['opacities'][key])
            self.renderer.AddActor(actor)

            if key in self.style['labels']:
                atext = vtk.vtkVectorText()
                atext.SetText("%d" % i)
                textMapper = vtk.vtkPolyDataMapper()
                textMapper.SetInputConnection(atext.GetOutputPort())
                textActor = vtk.vtkActor()
                textActor.SetOrientation(180,0,180)
                textActor.SetMapper(textMapper)
                f = 0.7
                textActor.SetScale(f,f,f)
                textActor.AddPosition(centers[i,:]) 
                self.renderer.AddActor(textActor)
    
    def create_bbox(self):
        """Add a bounding box of the entire geometry to the visualization."""
        bb = vtk.vtkOutlineSource()
        Nx, Ny, Nz = self.voxels.shape
        bb.SetBounds(0, Nx, 0, Ny, 0, Nz);

        mapper = vtk.vtkPolyDataMapper();
        mapper.SetInputConnection(bb.GetOutputPort());

        bbox_actor = vtk.vtkActor()
        bbox_actor.SetMapper(mapper);
        bbox_actor.GetProperty().SetColor((0,0,0))
        self.renderer.AddActor(bbox_actor) # add boundingbox



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize a 3D geometry.',
                                     usage='%(prog)s geometry_file [options]'
                                     '\nUse flag --help for info.')
    parser.add_argument('geometry_file', type=str, 
                        help='h5 geometry file to visualize')
    parser.add_argument('--domains', nargs='*', metavar='d', type=int, 
                        default=[2, 3],
                        help='sequence of visible domains, use ids from [1, 4]')
    parser.add_argument('--boundaries', nargs='*', metavar='b', default=['ryr'], 
                        help='sequence of visibile boundaries, use names'
                             '(ryr, serca, sr_cleft')
    parser.add_argument('--labels', nargs='*', metavar='b', default=['ryr'], 
                        help='sequence of visible labels, use names'
                             '(ryr')
    parser.add_argument('--frame', dest='bbox', action='store_true',
                        help='Draw a bounding box of entire geometry')
    parser.add_argument('--no-frame', dest='bbox', action='store_false',
                        help='Do not draw bounding box for entire geometry')
    parser.add_argument('--saveimg', type=str, default=None, 
                        help='If provided, save a png image to the given path')
    parser.add_argument('--azimuth', type=int, default=210,
                        help='If provided, sets the inital rotation of the camera')
    parser.add_argument('--roll', type=int, default=0,
                        help='If provided, sets the inital rotation of the camera')
    parser.add_argument('--elevation', type=int, default=20,
                        help='If provided, sets the inital rotation of the camera')

    args = parser.parse_args()
    geomviz = GeometryVisualizer(args.geometry_file)
    
    geomviz.style['domains'] = args.domains
    geomviz.style['boundaries'] = args.boundaries
    geomviz.style['labels'] = args.labels
    if args.bbox is not None:
        geomviz.style['bbox'] = args.bbox
    geomviz.imgpath = args.saveimg
    geomviz.render(azimuth=args.azimuth, roll=args.roll, elevation=args.elevation)



