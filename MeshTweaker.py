# -*- coding: utf-8 -*-

import math
from time import time, sleep
import re
import os
from collections import Counter
# upgrade numpy with: "pip install numpy --upgrade"
import numpy as np

# Constants used:
VECTOR_TOL = 0.001  # To remove alignment duplicates, the vector tolerance is 
# used to distinguish two vectors.
PLAFOND_ADV = 0.2   # Printing a plafond is known to be more effective than
# very step overhangs. This value sets the advantage in %.
FIRST_LAY_H = 0.25   # The initial layer of a print has an altitude > 0
# bottom layer and very bottom-near overhangs can be handled as similar.
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
    def __init__(self, content, extended_mode=False, verbose=True,
                 show_progress=False, favside=None):

        self.extended_mode = extended_mode
        self.show_progress = show_progress
        z_axis = -np.array([0, 0, 1], dtype=np.float64)
        orientations = [[list(z_axis), 0.0]]

        # Preprocess the input mesh format.
        t_start = time()
        progress = 0  # progress in percent of tweaking
        progress = self.print_progress(progress)

        mesh = self.preprocess(content)

        if favside:
            mesh = self.favour_side(mesh, favside)
        t_pre = time()
        progress = self.print_progress(progress)

        # Searching promising orientations:
        orientations += self.area_cumulation(mesh, 10)

        t_areacum = time()
        progress = self.print_progress(progress)
        if extended_mode:
            orientations += self.death_star(mesh, 8)
            orientations += self.add_supplements()
            orientations = self.remove_duplicates(orientations)

        if verbose:
            print("Examine {} orientations:".format(len(orientations)))
            print("  %-26s %-10s%-10s%-10s%-10s " % 
                  ("Alignment:", "Bottom:", "Overhang:", "Contour:", "Unpr.:"))

        t_ds = time()
        progress = self.print_progress(progress)

        # Calculate the unprintability for each orientation
        #results = np.array([None, None, None, None, np.inf])
        results = list()
        for side in orientations:
            #orientation = np.array([float("{:6f}".format(-i)) for i in side[0]])
            orientation = [float("{:6f}".format(-i)) for i in side[0]]

            mesh = self.project_verteces(mesh, orientation)
            bottom, overhang, contour = self.lithograph(mesh, orientation)
            unprintability = self.target_function(bottom, overhang, contour)
            # results = np.vstack((results, [orientation, bottom,
            #                                overhang, contour, unprintability]))
            results.append([orientation, bottom, overhang, contour, unprintability])
            if verbose:
                print("  %-26s %-10s%-10s%-10s%-10s "
                      % (str(np.around(orientation, decimals=4)),
                         round(bottom, 3), round(overhang, 3), round(contour, 3),
                         round(unprintability, 2)))

        t_lit = time()
        progress = self.print_progress(progress)

        # evaluate the best 5 alignments and calculate the rotation parameters
        results = np.array(results)
        best_5_results = results[results[:, 4].argsort()[:5]]
        best_5_results = list(best_5_results)

        for i, align in enumerate(best_5_results):
            best_5_results[i] = list(best_5_results[i])
            v, phi, matrix = self.euler(align)
            best_5_results[i].append([[v[0], v[1], v[2]], phi, matrix])

        if verbose:
            print("""Time-stats of algorithm:
  Preprocessing:    \t{pre:2f} s
  Area Cumulation:  \t{ac:2f} s
  Death Star:       \t{ds:2f} s
  Lithography Time:  \t{lt:2f} s
  Total Time:        \t{tot:2f} s""".format(pre=t_pre-t_start, ac=t_areacum-t_pre, ds=t_ds-t_areacum,
                                            lt=t_lit-t_ds, tot=t_lit-t_start))

        # The list best_5_results is of the form:
        # [[orientation0, bottom_area0, overhang_area0, contour_line_length, unprintability (gives the order),
        #       [euler_vector, euler_angle (in rad), rotation matrix]],
        #   orientation1, ..
        if len(best_5_results) > 0:
            self.euler_parameter = best_5_results[0][5][:2]
            self.matrix = best_5_results[0][5][2]
            self.alignment = best_5_results[0][0]
            self.bottom_area = best_5_results[0][1]
            self.overhang_area = best_5_results[0][2]
            self.contour = best_5_results[0][3]
            self.unprintability = best_5_results[0][4]
            self.best_5 = best_5_results

    def target_function(self, bottom, overhang, contour):
        """This function returns the Unprintability for a given set of bottom
        overhang area and bottom contour lenght, based on an ordinal scale.
        Args:
            bottom (float): bottom area size.
            overhang (float): overhanging area size.
            contour (float): length of the bottom's contour.
        Returns:
            a value for the unprintability. The smaller, the better."""
        unprintability = (overhang/ABSOLUTE_F
                          + (overhang + 1) / (1 + CONTOUR_F*contour + bottom) / RELATIVE_F)
        return round(unprintability, 6)

    def preprocess(self, content):
        """The Mesh format gets preprocessed for a better performance.
        Args:
            content (np.array): undefined representation of the mesh
        Returns:
            mesh (np.array): with format face_count x 6 x 3.
        """
        mesh = np.array(content, dtype=np.float64)
        
        # prefix area vector, if not already done (e.g. in STL format)
        if len(mesh[0]) == 3:
            row_number = int(len(content)/3)
            mesh = mesh.reshape(row_number, 3, 3)
            v0 = mesh[:, 0, :]
            v1 = mesh[:, 1, :]
            v2 = mesh[:, 2, :]
            normals = np.cross(np.subtract(v1, v0), np.subtract(v2, v0)) \
                .reshape(row_number, 1, 3)
            mesh = np.hstack((normals, mesh))
        
        face_count = len(mesh)
        
        # append columns with a_min, area_size
        addendum = np.zeros((face_count, 2, 3))
        addendum[:, 0, 0] = mesh[:, 1, 2]
        addendum[:, 0, 1] = mesh[:, 2, 2]
        addendum[:, 0, 2] = mesh[:, 3, 2]
        
        # calc area size
        addendum[:, 1, 0] = np.sqrt(np.sum(np.square(mesh[:, 0, :]), axis=-1)).reshape(face_count)
        addendum[:, 1, 1] = np.max(mesh[:, 1:4, 2], axis=1)
        addendum[:, 1, 2] = np.median(mesh[:, 1:4, 2], axis=1)
        mesh = np.hstack((mesh, addendum))
        
        # filter faces without area
        mesh = mesh[mesh[:, 5, 0] != 0]
        face_count = len(mesh)
        
        # normalise area vector and correct area size
        mesh[:, 0, :] = mesh[:, 0, :]/mesh[:, 5, 0].reshape(face_count, 1)
        mesh[:, 5, 0] = mesh[:, 5, 0]/2  # halfed because areas are triangle and no paralellograms
        
        # remove small facets (these are essential for countour calculation)
        if NEGL_FACE_SIZE > 0:  # TODO remove facets smaller than a relative proportion of the total size
            negl_size = [0.1*x if self.extended_mode else x for x in [NEGL_FACE_SIZE]][0]
            filtered_mesh = mesh[mesh[:, 5, 0] > negl_size]
            if len(filtered_mesh) > 100:
                mesh = filtered_mesh

        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return mesh

    def favour_side(self, mesh, favside):
        """This function weights the size of orientations closer than 45 deg
        to a favoured side higher.
        Args:
            mesh (np.array): with format face_count x 6 x 3.
            favside (string): the favoured side  "[[0,-1,2.5],3]"
        Returns:
            a weighted mesh or the original mesh in case of invalid input
        """
        if isinstance(favside, str):
            try:
                restring = r"(-?\d*\.{0,1}\d+)[, []]*(-?\d*\.{0,1}\d+)[, []]*(-?\d*\.{0,1}\d+)\D*(-?\d*\.{0,1}\d+)"
                x = float(re.search(restring, favside).group(1))
                y = float(re.search(restring, favside).group(2))
                z = float(re.search(restring, favside).group(3))
                f = float(re.search(restring, favside).group(4))
            except AttributeError:
                raise AttributeError("Could not parse input: favored side")
        else:
            raise AttributeError("Could not parse input: favored side")

        norm = np.sqrt(np.sum(np.array([x, y, z])**2))
        side = np.array([x, y, z])/norm

        print("You favour the side {} with a factor of {}".format(
            side, f))

        diff = np.subtract(mesh[:, 0, :], side)
        align = np.sum(diff*diff, axis=1) < 0.7654
        mesh_not_align = mesh[np.logical_not(align)]
        mesh_align = mesh[align]
        mesh_align[:, 5, 0] = f * mesh_align[:, 5, 0]  # weight aligning orientations

        mesh = np.concatenate((mesh_not_align, mesh_align), axis=0)
        return mesh

    def area_cumulation(self, mesh, best_n):
        """
        Gathering promising alignments by the accumulation of
        the magnitude of parallel area vectors.
        Args:
            mesh (np.array): with format face_count x 6 x 3.
            best_n (int): amount of orientations to return.
        Returns:
            list of the common orientation-tuples.
        """
        if not self.extended_mode:  # instead of 10
            best_n = 7

        alignments = mesh[:, 0, :]
        orient = Counter()
        for index in range(len(mesh)):       # Accumulate area-vectors
            orient[tuple(alignments[index])] += mesh[index, 5, 0]

        top_n = orient.most_common(best_n)
        top_n = [[list(el[0]), float("{:2f}".format(el[1]))] for el in top_n]
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return top_n

    def death_star(self, mesh, best_n):
        """
        Creating random faces by adding a random vertex to an existing edge.
        Common orientations of these faces are promising orientations for
        placement.
        Args:
            mesh (np.array): with format face_count x 6 x 3.
            best_n (int): amount of orientations to return.
        Returns:
            list of the common orientation-tuples.
        """

        # Small files need more calculations
        mesh_len = len(mesh)
        iterations = int(np.ceil(20000/(mesh_len + 100)))

        vertexes = mesh[:mesh_len, 1:4, :]
        orientations = list()
        for i in range(iterations):
            two_vertexes = vertexes[:, np.random.choice(3, 2, replace=False)]
            vertex_0 = two_vertexes[:, 0, :]
            vertex_1 = two_vertexes[:, 1, :]

            # Using a linear congruency generator instead to choice pseudo
            # random vertexes. Adding i to get more iterations.
            vertex_2 = vertexes[(np.arange(mesh_len) * 127 + 8191 + i) % mesh_len, i % 3, :]
            normals = np.cross(np.subtract(vertex_2, vertex_0),
                               np.subtract(vertex_1, vertex_0))

            # normalise area vector
            lengths = np.sqrt((normals*normals).sum(axis=1)).reshape(mesh_len, 1)
            # ignore ZeroDivisions
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_orientations = np.around(np.true_divide(normals, lengths),
                                                    decimals=6)

            # append hashable tuples to list
            orientations += [tuple(face) for face in normalized_orientations]
            sleep(0)  # Yield, so other threads get a bit of breathing space.

        # search the most common orientations
        orient = Counter(orientations)
        top_n = orient.most_common(best_n)
        top_n = list(filter(lambda x: x[1] > 2, top_n))

        top_n = [[list(v[0]), v[1]] for v in top_n]
        # also add anti-parallel orientations
        top_n += [[list((-v[0][0], -v[0][1], -v[0][2])), v[1]] for v in top_n]
        return top_n

    def add_supplements(self):
        """Supplement 18 additional vectors.
        Returns:
            Basic Orientation Field"""
        v = [[0, 0, -1], [0.70710678, 0, -0.70710678], [0, 0.70710678, -0.70710678],
             [-0.70710678, 0, -0.70710678], [0, -0.70710678, -0.70710678],
             [1, 0, 0], [0.70710678, 0.70710678, 0], [0, 1, 0], [-0.70710678, 0.70710678, 0],
             [-1, 0, 0], [-0.70710678, -0.70710678, 0], [0, -1, 0], [0.70710678, -0.70710678, 0],
             [0.70710678, 0, 0.70710678], [0, 0.70710678, 0.70710678],
             [-0.70710678, 0, 0.70710678], [0, -0.70710678, 0.70710678], [0, 0, 1]]
        v = [[list([float(j) for j in i]), 0] for i in v]
        return v

    def remove_duplicates(self, old_orients):
        """Removing duplicate and similar orientations.
        Args:
            old_orients (list): list of faces
        Returns:
            Unique orientations"""
        alpha = 5  # degrees
        tol_angle = np.sin(5 * np.pi / 180)
        orientations = list()
        for i in old_orients:
            duplicate = None
            for j in orientations:
                # redundant vectors have an angle smaller than
                # alpha = arcsin(atol). atol=0.087 -> alpha = 5 degrees
                if np.allclose(i[0], j[0], atol=tol_angle):
                    duplicate = True
                    break
            if duplicate is None:
                orientations.append(i)
        return orientations

    def project_verteces(self, mesh, orientation):
        """Supplement the mesh array with scalars (max and median)
        for each face projected onto the orientation vector.
        Args:
            mesh (np.array): with format face_count x 6 x 3.
            orientation (np.array): with format 3 x 3.
        Returns:
            adjusted mesh.
        """
        mesh[:, 4, 0] = np.inner(mesh[:, 1, :], orientation)
        mesh[:, 4, 1] = np.inner(mesh[:, 2, :], orientation)
        mesh[:, 4, 2] = np.inner(mesh[:, 3, :], orientation)

        mesh[:, 5, 1] = np.max(mesh[:, 4, :], axis=1)
        mesh[:, 5, 2] = np.median(mesh[:, 4, :], axis=1)
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return mesh

    def lithograph(self, mesh, orientation):
        """Calculating bottom and overhang area for a mesh regarding
        the vector n.
        Args:
            mesh (np.array): with format face_count x 6 x 3.
            orientation (np.array): with format 3 x 3.
        Returns:
            the total bottom size, overhang size and contour length of the mesh
        """
        ascent = np.cos(120*np.pi/180)
        anti_orient = -np.array(orientation)
        total_min = np.amin(mesh[:, 4, :])

        # filter bottom area        
        bottoms = mesh[mesh[:, 5, 1] < total_min + FIRST_LAY_H]
        if len(bottoms) > 0:
            bottom = np.sum(bottoms[:, 5, 0])
        else:
            bottom = 0

        # filter overhangs
        overhangs = mesh[np.inner(mesh[:, 0, :], orientation) < ascent]
        overhangs = overhangs[overhangs[:, 5, 1] > (total_min + FIRST_LAY_H)]

        if self.extended_mode:
            plafonds = overhangs[(overhangs[:, 0, :] == anti_orient).all(axis=1)]
            if len(plafonds) > 0:
                plafond = np.sum(plafonds[:, 5, 0])
            else:
                plafond = 0
        else:
            plafond = 0

        if len(overhangs) > 0:
            overhang = np.sum(overhangs[:, 5, 0] * 2 *
                              (np.amax((np.zeros(len(overhangs))+0.5,
                                        - np.inner(overhangs[:, 0, :], orientation)),
                                       axis=0) - 0.5)**2)
            overhang -= PLAFOND_ADV * plafond
        else:
            overhang = 0

        # filter the total length of the bottom area's contour
        if self.extended_mode:
            # contours = mesh[total_min+FIRST_LAY_H < mesh[:, 5, 1]]
            contours = mesh[mesh[:, 5, 2] < total_min+FIRST_LAY_H]
            
            if len(contours) > 0:
                conlen = np.arange(len(contours))
                sortsc0 = np.argsort(contours[:, 4, :], axis=1)[:, 0]
                sortsc1 = np.argsort(contours[:, 4, :], axis=1)[:, 1]

                con = np.array([np.subtract(
                    contours[conlen, 1+sortsc0, :],
                    contours[conlen, 1+sortsc1, :])])
                    
                contours = np.sum(np.power(con, 2), axis=-1)**0.5
                contour = np.sum(contours)     
            else:
                contour = 0
        else:  # consider the bottom area as square, bottom=a**2 ^ contour=4*a
            contour = 4*np.sqrt(bottom)
        
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return bottom, overhang, contour

    def print_progress(self, progress):
        progress += 18
        if self.show_progress:
            os.system('cls')
            print("Progress is: {} %".format(progress))
        return progress

    def euler(self, bestside):
        """Calculating euler rotation parameters and rotational matrix.
        Args:
            bestside (np.array): vector of the best orientation (3 x 3).
        Returns:
            rotation axis, rotation angle, rotational matrix.
        """
        if np.allclose(bestside[0], np.array([0, 0, -1]), atol=VECTOR_TOL):
            rotation_axis = [1, 0, 0]
            phi = np.pi
        elif np.allclose(bestside[0], np.array([0, 0, 1]), atol=VECTOR_TOL):
            rotation_axis = [1, 0, 0]
            phi = 0
        else:
            phi = float("{:2f}".format(np.pi - np.arccos(-bestside[0][2])))
            rotation_axis = [-bestside[0][1], bestside[0][0], 0]
            rotation_axis = [i / np.sum(np.power(rotation_axis, 2), axis=-1)**0.5 for i in rotation_axis]
            rotation_axis = np.array([float("{:2f}".format(i)) for i in rotation_axis])

        v = rotation_axis
        rotational_matrix = [[v[0] * v[0] * (1 - math.cos(phi)) + math.cos(phi),
                              v[0] * v[1] * (1 - math.cos(phi)) - v[2] * math.sin(phi),
                              v[0] * v[2] * (1 - math.cos(phi)) + v[1] * math.sin(phi)],
                             [v[1] * v[0] * (1 - math.cos(phi)) + v[2] * math.sin(phi),
                              v[1] * v[1] * (1 - math.cos(phi)) + math.cos(phi),
                              v[1] * v[2] * (1 - math.cos(phi)) - v[0] * math.sin(phi)],
                             [v[2] * v[0] * (1 - math.cos(phi)) - v[1] * math.sin(phi),
                              v[2] * v[1] * (1 - math.cos(phi)) + v[0] * math.sin(phi),
                              v[2] * v[2] * (1 - math.cos(phi)) + math.cos(phi)]]
        rotational_matrix = np.around(rotational_matrix, decimals=6)

        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return rotation_axis, phi, rotational_matrix
