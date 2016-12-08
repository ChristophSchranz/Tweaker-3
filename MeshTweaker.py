## Author: Christoph Schranz, Salzburg Research, 2015/16
## Contact: christoph.schranz@salzburgresearch.at
## Runs on: Python 2.7 and 3.5

#import sys
import math
from time import time, sleep
from collections import Counter
# upgrade numpy with: "pip install numpy --upgrade"
import numpy as np

# Constants used:
VECTOR_TOL = 0.001  # To remove alignment duplicates, the vector tolerance is 
                    # used to distinguish whether two vectors are the same.
PLAFOND_ADV = 0.2   # Printing a plafond is known to be more effective than
                    # very step overhangs. This value sets the advantage in %.
FIRST_LAY_H = 0.1   # Since the initial layer of a print has a higher altitude
                    # >= 0, bottom layer and very bottom-near overhangs can be
                    # handled as similar.
NEGL_FACE_SIZE = 1  # The fast operation mode neglects facet sizes smaller than
                    # this value (in mm^2) for a better performance
ABSOLUTE_F = 100    # These values scale the the parameters bottom size,
RELATIVE_F = 1      # overhang size, and bottom contour lenght to get a robust
CONTOUR_F = 0.5     # value for the Unprintability


class Tweak:    
    """ The Tweaker is an auto rotate class for 3D objects.

    The critical angle CA is a variable that can be set by the operator as
    it may depend on multiple factors such as material used, printing
     temperature, printing speed, etc.

    Following attributes of the class are supported:
    The tweaked z-axis'.
    Euler coords .v and .phi, where v is orthogonal to both z and z' and phi
     the angle between z and z' in rad.
    The rotational matrix .Matrix, the new mesh is created by multiplying each
     vector with R.
    And the relative unprintability of the tweaked object. If this value is
     greater than 10, a support structure is suggested.
        """
    def __init__(self, content, extended_mode=False, verbose=True):

        self.extended_mode = extended_mode
        n = -np.array([0,0,1], dtype=np.float64)
        orientations = [[list(n), 0.0]]
        
        ## Preprocess the input mesh format.
        t_start = time()
        mesh = self.preprocess(content)
        t_pre = time()
        
        ## Searching promising orientations: 
        orientations += self.area_cumulation(mesh, n)

        t_areacum = time()
        if extended_mode:
            dialg_time = time()
            orientations += self.death_star(mesh, 8)
            dialg_time = time() - dialg_time   
            
        t_ds = time()
        
        orientations = self.remove_duplicates(orientations)
        
        if verbose:
            print("Examine {} orientations:".format(len(orientations)))
            print("  %-26s %-10s%-10s%-10s%-10s " %("Alignment:", 
            "Bottom:", "Overhang:", "Contour:", "Unpr.:"))
        
        # Calculate the unprintability for each orientation
        results = np.array([None,None,None,None,np.inf])
        for side in orientations:
            orientation =np.array([float("{:6f}".format(-i)) for i in side[0]])
            mesh = self.project_verteces(mesh, orientation)
            bottom, overhang, contour = self.lithograph(mesh, orientation)
            Unprintability = self.target_function(bottom, overhang, contour)
            results = np.vstack((results, [orientation, bottom,
                            overhang, contour, Unprintability]))                        
            if verbose:
                print("  %-26s %-10s%-10s%-10s%-10s " 
                %(str(np.around(orientation, decimals = 4)), 
                round(bottom, 3), round(overhang,3), round(contour,3), 
                round(Unprintability,2)))
        t_lit = time()               
               
        # Best alignment
        best_alignment = results[np.argmin(results[:, 4])]
            
           
        if verbose:
            print("""
Time-stats of algorithm:
  Preprocessing:    \t{pre:2f} s
  Area Cumulation:  \t{ac:2f} s
  Death Star:       \t{ds:2f} s
  Lithography Time:  \t{lt:2f} s  
  Total Time:        \t{tot:2f} s
""".format(pre=t_pre-t_start, ac=t_areacum-t_pre, 
           ds=t_ds-t_areacum, lt=t_lit-t_ds, tot=t_lit-t_start))  
           
        if len(best_alignment) > 0:
            [v, phi, Matrix] = self.euler(best_alignment)
            self.Euler = [[v[0],v[1],v[2]], phi]
            self.Matrix = Matrix
            
            self.Alignment=best_alignment[0]
            self.BottomArea = best_alignment[1]
            self.Overhang = best_alignment[2]
            self.Contour = best_alignment[3]
            self.Unprintability = best_alignment[4]
            
        return None


    def target_function(self, bottom, overhang, contour):
        '''This function returns the Unprintability for a given set of bottom
        overhang area and bottom contour lenght, based on an ordinal scale.'''
        Unprintability =( overhang/ABSOLUTE_F
                + (overhang + 1) / (1 + CONTOUR_F*contour + bottom) /RELATIVE_F)
                
        return np.around(Unprintability, 6)
        
        
    def preprocess(self, content):
        '''The Mesh format gets preprocessed for a better performance.'''
        mesh = np.array(content, dtype=np.float64)
        
        # prefix area vector, if not already done (e.g. in STL format)
        if len(mesh[0]) == 3:
            row_number = int(len(content)/3)
            mesh = mesh.reshape(row_number,3,3)
            v0=mesh[:,0,:]
            v1=mesh[:,1,:]
            v2=mesh[:,2,:]
            normals = np.cross( np.subtract(v1,v0), np.subtract(v2,v0)
                                                    ).reshape(row_number,1,3)
            mesh = np.hstack((normals,mesh))
        
        face_count = len(mesh)
        
        # append columns with a_min, area_size
        addendum = np.zeros((face_count, 2, 3))
        addendum[:,0,0] = mesh[:,1,2]
        addendum[:,0,1] = mesh[:,2,2]
        addendum[:,0,2] = mesh[:,3,2]
        
        # calc area size
        addendum[:,1,0] = (np.sum(np.abs(mesh[:,0,:])**2, axis=-1)**0.5).reshape(face_count)
        addendum[:,1,1] = np.max(mesh[:,1:4,2], axis=1)
        addendum[:,1,2] = np.median(mesh[:,1:4,2], axis=1)
        mesh = np.hstack((mesh, addendum))
        
        # filter faces without area
        mesh[mesh[:,5,0]!=0]

        # normalise area vector and correct area size
        mesh[:,0,:] = mesh[:,0,:]/mesh[:,5,0].reshape(face_count, 1)
        mesh[:,5,0] = mesh[:,5,0]/2
        
        # remove small facets (these are essential for countour calculation)
        if not self.extended_mode: ## TODO remove facets smaller than a 
                                #relative proportion of the total dimension
            filtered_mesh = mesh[mesh[:,5,0] > NEGL_FACE_SIZE]
            if len(filtered_mesh) > 100:
                mesh = filtered_mesh
            
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return mesh


    def area_cumulation(self, mesh, n):
        '''Gathering the most auspicious alignments by cumulating the 
        magnitude of parallel area vectors.'''
        if self.extended_mode: best_n = 10
        else: best_n = 7
        orient = Counter()
        
        align = mesh[:,0,:]
        for index in range(len(mesh)):       # Cumulate areavectors
            orient[tuple(align[index])] += mesh[index, 5, 0]

        top_n = orient.most_common(best_n)
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return [[list(el[0]), float("{:2f}".format(el[1]))] for el in top_n]
       
       
    def death_star(self, mesh, best_n):
        '''Searching normals or random edges with one vertice'''
        vcount = len(mesh)
        # Small files need more calculations
        if vcount < 1000: it = 30
        elif vcount < 2000: it = 15
        elif vcount < 5000: it = 5
        elif vcount < 10000: it = 3
        elif vcount < 20000: it = 2
        else: it = 1     
        
        vertexes = mesh[:vcount,1:4,:]
        v0u1 = vertexes[:,np.random.choice(3, 2, replace=False)]
        v0 = v0u1[:,0,:]
        v1 = v0u1[:,1,:]
        v2 = vertexes[:,np.random.choice(3,1, replace=False)].reshape(vcount,3)
        
        lst = list()
        for i in range(it):
            v2 = v2[np.random.choice(vcount, vcount),:]
            normals = np.cross( np.subtract(v2,v0), np.subtract(v1,v0))

            # normalise area vector
            area_size = (np.sum(np.abs(normals)**2, axis=-1)**0.5).reshape(vcount,1)
            nset = np.hstack((normals, area_size))
            
            nset = np.array([n for n in nset if n[3]!=0])  
            if nset.size == 0:
                continue
            
            normals = np.around(nset[:,0:3]/nset[:,3].reshape(len(nset),1), 
                                decimals=6)

            lst += [tuple(face) for face in normals]
            sleep(0)  # Yield, so other threads get a bit of breathing space.
        
        orient = Counter(lst)
        top_n = orient.most_common(best_n)
        top_n = list(filter(lambda x: x[1] > 2, top_n))
        
        # add antiparallel orientations
        top_n = [[list(v[0]),v[1]] for v in top_n]
        top_n += [[list((-v[0][0], -v[0][1], -v[0][2] )), v[1]] 
                        for v in top_n]
        return top_n


    def remove_duplicates(self, o):
        '''Removing duplicates in orientation'''
        orientations = list()
        for i in o:
            duplicate = None
            for j in orientations:
                if np.allclose(i[0],j[0], atol = 0.0001):
                    duplicate = True
                    break
            if duplicate is None:
                orientations.append(i)
        return orientations


    def project_verteces(self, mesh, orientation):
        '''Returning the "lowest" point vector regarding a vector n for
        each vertex.'''
        mesh[:,4,0] = np.inner(mesh[:,1,:], orientation)
        mesh[:,4,1] = np.inner(mesh[:,2,:], orientation)
        mesh[:,4,2] = np.inner(mesh[:,3,:], orientation)
               
        mesh[:,5,1] = np.max(mesh[:,4,:], axis=1)
        mesh[:,5,2] = np.median(mesh[:,4,:], axis=1)
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return mesh
        
        
    def lithograph(self, mesh, orientation):
        '''Calculating bottom and overhang area for a mesh regarding 
        the vector n.'''
        overhang = 0
        bottom = 0
        
        ascent = np.cos((120)*np.pi/180)
        anti_orient = -np.array(orientation)
        total_min = np.amin(mesh[:,4,:])

        # filter bottom area        
        bottoms = np.array([face for face in mesh
                if face[5,1] < total_min + FIRST_LAY_H])
        if len(bottoms) > 0:
            bottom = np.sum(bottoms[:,5,0]) 
        else: bottom = 0
        
        # filter overhangs
        overhangs = mesh[np.inner(mesh[:,0,:], orientation) < ascent]
        overhangs = overhangs[overhangs[:,5,1] > (total_min + FIRST_LAY_H)]
                    
        if self.extended_mode:
            plafonds = overhangs[(overhangs[:,0,:]==anti_orient).all(axis=1)]
            if len(plafonds) > 0:
                plafond = np.sum(plafonds[:,5,0]) 
            else: plafond = 0
        else: plafond = 0
        if len(overhangs) > 0:
            overhang = np.sum(overhangs[:,5,0] * 2
                *(np.amax((np.zeros(len(overhangs))+0.5,
                           -np.inner(overhangs[:,0,:], orientation))
                           ,axis=0) -0.5 )**2)
            overhang = overhang - PLAFOND_ADV * plafond  
        else: overhang = 0
        

        # filter the total length of the bottom area's contour
        if self.extended_mode:
            contours = mesh[total_min+FIRST_LAY_H < mesh[:,5,1]]
            contours = mesh[mesh[:,5,2] < total_min+FIRST_LAY_H]
            
            if len(contours) > 0:
                conlen = np.arange(len(contours))
                sortsc0 = np.argsort(contours[:,4,:], axis=1)[:,0]
                sortsc1 = np.argsort(contours[:,4,:], axis=1)[:,1]

                con = np.array([np.subtract(
                    contours[conlen,1+sortsc0,:],
                    contours[conlen,1+sortsc1,:])])
                    
                contours = np.sum(np.abs(con)**2, axis=-1)**0.5
                contour = np.sum(contours)     
            else:
                contour = 0
        else: # consider the bottom area as square, bottom=a**2 ^ contour=4*a
            contour = 4*np.sqrt(bottom)
        
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return bottom, overhang, contour


    def euler(self, bestside):
        '''Calculating euler rotation parameters and rotation matrix'''
        if np.allclose(bestside[0], np.array([0, 0, -1]), atol = VECTOR_TOL):
            v = [1, 0, 0]
            phi = np.pi
        elif np.allclose(bestside[0], np.array([0, 0, 1]), atol = VECTOR_TOL):
            v = [1, 0, 0]
            phi = 0
        else:
            phi = float("{:2f}".format(np.pi - np.arccos( -bestside[0][2] )))
            v = [-bestside[0][1] , bestside[0][0], 0]
            v = [i / np.sum(np.abs(v)**2, axis=-1)**0.5 for i in v]
            v = np.array([float("{:2f}".format(i)) for i in v])
            
        R = [[v[0] * v[0] * (1 - math.cos(phi)) + math.cos(phi),
              v[0] * v[1] * (1 - math.cos(phi)) - v[2] * math.sin(phi),
              v[0] * v[2] * (1 - math.cos(phi)) + v[1] * math.sin(phi)],
             [v[1] * v[0] * (1 - math.cos(phi)) + v[2] * math.sin(phi),
              v[1] * v[1] * (1 - math.cos(phi)) + math.cos(phi),
              v[1] * v[2] * (1 - math.cos(phi)) - v[0] * math.sin(phi)],
             [v[2] * v[0] * (1 - math.cos(phi)) - v[1] * math.sin(phi),
              v[2] * v[1] * (1 - math.cos(phi)) + v[0] * math.sin(phi),
              v[2] * v[2] * (1 - math.cos(phi)) + math.cos(phi)]]
        R = np.around(R, decimals = 6)
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return v,phi,R
