# -*- coding: utf-8 -*-
import os
import re
import math
from time import time, sleep
from collections import Counter
# upgrade numpy with: "pip install numpy --upgrade"
import numpy as np


# These parameter were minimized by the evolutionary algorithm
# https://github.com/ChristophSchranz/Tweaker-3_optimize-using-ea, branch ea-optimize_20200414' on 100 objects
# with a fitness of 5.10246, and a miss-classification rate of 4.0
PARAMETER = {
    "TAR_A": 0.023251193283878126,
    "TAR_B": 0.17967732044591803,
    "RELATIVE_F": 11.250931864115714,
    "CONTOUR_F": 0.219523237806102,
    "BOTTOM_F": 1.3206227038470124,
    "TAR_C": -0.016564249433447253,
    "TAR_D": 1.0592490333488807,
    "TAR_E": 0.011503545133447014,
    "FIRST_LAY_H": 0.04754881938390257,
    "VECTOR_TOL": -0.0008385913582234466,
    "NEGL_FACE_SIZE": 0.4737309463791554,
    "ASCENT": -0.07809801382985776,
    "PLAFOND_ADV": 0.059937025927212395,
    "CONTOUR_AMOUNT": 0.018242751444131886,
    "OV_H": 2.574100894603089,
    "height_offset": 2.372824083342488,
    "height_log": 0.04137517666768212,
    "height_log_k": 1.9325457851679673
}
# https://github.com/ChristophSchranz/Tweaker-3_optimize-using-ea, branch ea-optimize_20200427_vol' on 100 objects
# with a fitness of 4.06166, and a miss-classification rate of 3.5
PARAMETER_VOL = {
    "TAR_A": 0.012826785357111374,
    "TAR_B": 0.1774847296275851,
    "RELATIVE_F": 6.610621027964314,
    "CONTOUR_F": 0.23228623269775997,
    "BOTTOM_F": 1.167152017941474,
    "TAR_C": 0.24308070476924726,
    "TAR_D": 0.6284515508160871,
    "TAR_E": 0.032157292647062234,
    "FIRST_LAY_H": 0.029227991916155015,
    "VECTOR_TOL": -0.0011163303070972383,
    "NEGL_FACE_SIZE": 0.4928696161029859,
    "ASCENT": -0.23897449119622627,
    "PLAFOND_ADV": 0.04079208948120519,
    "CONTOUR_AMOUNT": 0.0101472219892684,
    "OV_H": 1.0370178217794535,
    "height_offset": 2.7417608343142073,
    "height_log": 0.06442030687034085,
    "height_log_k": 0.3933594673063997
}


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

    def __init__(self, content, extended_mode=False, verbose=True, show_progress=False,
                 favside=None, min_volume=False, parameter=None,  progress_callback=None):
        # Load parameters
        if parameter is None:
            if min_volume:
                parameter = PARAMETER_VOL
            else:
                parameter = PARAMETER

        for k, v in parameter.items():
            setattr(self, k, v)

        if abs(self.OV_H - 2) < 0.1:  # set to nearby integers as they are faster
            self.OV_H = 2
        if abs(self.OV_H - 1) < 0.1:
            self.OV_H = 1

        self.progress_callback = progress_callback
        self.extended_mode = extended_mode
        self.show_progress = show_progress
        z_axis = -np.array([0, 0, 1], dtype=np.float64)
        orientations = [[z_axis, 0.0]]

        # Preprocess the input mesh format.
        t_start = time()
        self._progress = 0  # progress in percent of tweaking
        self.update_progress(self._progress + 18)
        # Load mesh from file into class variable
        self.mesh = self.preprocess(content)

        # if a favoured side is specified, load it to weight
        if favside:
            self.favour_side(favside)
        t_pre = time()
        self.update_progress(self._progress + 18)
        # Searching promising orientations:
        orientations += self.area_cumulation(10)

        t_areacum = time()
        self.update_progress(self._progress + 18)
        if extended_mode:
            orientations += self.death_star(12)
            orientations += self.add_supplements()
            orientations = self.remove_duplicates(orientations)

        if verbose:
            print("Examine {} orientations:".format(len(orientations)))
            print("  %-26s %-10s%-10s%-10s%-10s " %
                  ("Alignment:", "Bottom:", "Overhang:", "Contour:", "Unpr.:"))

        t_ds = time()
        self.update_progress(self._progress + 18)
        # Calculate the unprintability for each orientation found in the gathering algorithms
        results = list()
        for side in orientations:
            orientation = -1 * np.array(side[0], dtype=np.float64)

            self.project_vertices(orientation)
            bottom, overhang, contour = self.calc_overhang(orientation, min_volume=min_volume)
            unprintability = self.target_function(bottom, overhang, contour, min_volume=min_volume)
            results.append([orientation, bottom, overhang, contour, unprintability])
            if verbose:
                print("  %-26s %-10.2f%-10.2f%-10.2f%-10.4g "
                      % (str(np.around(orientation, decimals=4)),
                         bottom, overhang, contour, unprintability))
        t_lit = time()
        self.update_progress(self._progress + 18)

        # Remove the mesh structure as soon as it is not used anymore
        del self.mesh

        # evaluate the best alignments and calculate the rotation parameters
        results = np.array(results)
        best_results = list(results[results[:, 4].argsort()])  # [:5]]  # previously, the best 5 alignments were stored

        for i, align in enumerate(best_results):
            best_results[i] = list(best_results[i])
            v, phi, matrix = self.euler(align)
            best_results[i].append([[v[0], v[1], v[2]], phi, matrix])

        if verbose:
            print("""Time-stats of algorithm:
    Preprocessing:    \t{pre:2f} s
    Area Cumulation:  \t{ac:2f} s
    Death Star:       \t{ds:2f} s
    Lithography Time:  \t{lt:2f} s
    Total Time:        \t{tot:2f} s""".format(
                pre=t_pre - t_start, ac=t_areacum - t_pre, ds=t_ds - t_areacum,
                lt=t_lit - t_ds, tot=t_lit - t_start))

        # The list best_5_results is of the form:
        # [[orientation0, bottom_area0, overhang_area0, contour_line_length, unprintability (gives the order),
        #       [euler_vector, euler_angle (in rad), rotation matrix]],
        #   orientation1, ..
        if len(best_results) > 0:
            self.euler_parameter = best_results[0][5][:2]
            self.matrix = best_results[0][5][2]
            self.alignment = best_results[0][0]
            self.bottom_area = best_results[0][1]
            self.overhang_area = best_results[0][2]
            self.contour = best_results[0][3]
            self.unprintability = best_results[0][4]
            self.best_5 = best_results

        # Finish with a nice clean newline, as print_progress rewrites updates without advancing below.
        if show_progress:
            print("\n")

    def target_function(self, bottom, overhang, contour, min_volume):
        """This function returns the Unprintability for a given set of bottom
        overhang area and bottom contour length, based on an ordinal scale.
        Args:
            bottom (float): bottom area size.
            overhang (float): overhanging area size.
            contour (float): length of the bottom's contour.
            min_volume (bool): Minimise volume of support material or supported surface area
        Returns:
            a value for the unprintability. The smaller, the better."""
        if min_volume:  # minimize the volume of support material
            overhang /= 25  # a volume is of higher dimension, so the overhang have to be reduced
            return (self.TAR_A * (overhang + self.TAR_B) + self.RELATIVE_F * (overhang + self.TAR_C) /
                         (self.TAR_D + self.CONTOUR_F * contour + self.BOTTOM_F * bottom + self.TAR_E * overhang))
        else:
            return (self.TAR_A * (overhang + self.TAR_B) + self.RELATIVE_F *
                    (overhang + self.TAR_C) / (self.TAR_D + self.CONTOUR_F * contour + self.BOTTOM_F * bottom))

    def preprocess(self, content):
        """The Mesh format gets preprocessed for a better performance and stored into self.mesh
        Args:
            content (np.array): undefined representation of the mesh
        Returns:
            mesh (np.array): with format face_count x 6 x 3.
        """
        mesh = np.array(content, dtype=np.float64)

        # prefix area vector, if not already done (e.g. in STL format)
        if mesh.shape[1] == 3:
            row_number = int(len(content) / 3)
            mesh = mesh.reshape(row_number, 3, 3)
            v0 = mesh[:, 0, :]
            v1 = mesh[:, 1, :]
            v2 = mesh[:, 2, :]
            normals = np.cross(np.subtract(v1, v0), np.subtract(v2, v0)) \
                .reshape(row_number, 1, 3)
            mesh = np.hstack((normals, mesh))

        # saves the amount of facets
        face_count = mesh.shape[0]

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
        face_count = mesh.shape[0]

        # normalise area vector and correct area size
        mesh[:, 0, :] = mesh[:, 0, :] / mesh[:, 5, 0].reshape(face_count, 1)
        mesh[:, 5, 0] = mesh[:, 5, 0] / 2  # halve, because areas are triangles and not parallelograms

        # remove small facets (these are essential for contour calculation)
        if self.NEGL_FACE_SIZE > 0:
            negl_size = [0.1 * x if self.extended_mode else x for x in [self.NEGL_FACE_SIZE]][0]
            filtered_mesh = mesh[np.where(mesh[:, 5, 0] > negl_size)]
            if len(filtered_mesh) > 100:
                mesh = filtered_mesh

        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return mesh

    def favour_side(self, favside):
        """This function weights the size of orientations closer than 45 deg
        to a favoured side higher.
        Args:
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

        norm = np.sqrt(np.sum(np.array([x, y, z], dtype=np.float64) ** 2))
        side = np.array([x, y, z], dtype=np.float64) / norm

        print("You favour the side {} with a factor of {}".format(side, f))

        # Filter the aligning orientations
        diff = np.subtract(self.mesh[:, 0, :], side)
        align = np.sum(diff * diff, axis=1) < self.ANGLE_SCALE  # 0.7654, ANGLE_SCALE ist around 0.1
        mesh_not_align = self.mesh[np.logical_not(align)]
        mesh_align = self.mesh[align]
        mesh_align[:, 5, 0] = f * mesh_align[:, 5, 0]  # weight aligning orientations

        self.mesh = np.concatenate((mesh_not_align, mesh_align), axis=0)

    def area_cumulation(self, best_n):
        """
        Gathering promising alignments by the accumulation of
        the magnitude of parallel area vectors.
        Args:
            best_n (int): amount of orientations to return.
        Returns:
            list of the common orientation-tuples.
        """
        alignments = self.mesh[:, 0, :]
        orient = Counter()
        for index in range(len(self.mesh)):  # Accumulate area-vectors
            orient[tuple(alignments[index])] += self.mesh[index, 5, 0]

        top_n = orient.most_common(best_n)
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return top_n

    def death_star(self, best_n):
        """
        Creating random faces by adding a random vertex to an existing edge.
        Common orientations of these faces are promising orientations for
        placement.
        Args:
            best_n (int): amount of orientations to return.
        Returns:
            list of the common orientation-tuples.
        """

        # Small files need more calculations
        mesh_len = len(self.mesh)
        iterations = int(np.ceil(20000 / (mesh_len + 100)))

        vertexes = self.mesh[:mesh_len, 1:4, :]
        tot_normalized_orientations = np.zeros((iterations * mesh_len + 1, 3))
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
            lengths = np.sqrt((normals * normals).sum(axis=1)).reshape(mesh_len, 1)
            # ignore ZeroDivisions
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_orientations = np.around(np.true_divide(normals, lengths),
                                                    decimals=6)

            tot_normalized_orientations[mesh_len * i:mesh_len * (i + 1)] = normalized_orientations
            sleep(0)  # Yield, so other threads get a bit of breathing space.

        # search the most common orientations
        orientations = np.inner(np.array([1, 1e3, 1e6]), tot_normalized_orientations)
        orient = Counter(orientations)
        top_n = orient.most_common(best_n)
        top_n = list(filter(lambda x: x[1] > 2, top_n))

        candidate = list()
        for sum_side, count in top_n:
            face_unique, face_count = np.unique(tot_normalized_orientations[orientations == sum_side], axis=0,
                                                return_counts=True)
            candidate += [[list(face_unique[i]), count] for i, count in enumerate(face_count)]
        # Filter non-injective singles
        candidate = list(filter(lambda x: x[1] >= 2, candidate))
        # also add anti-parallel orientations
        candidate += [[list((-v[0][0], -v[0][1], -v[0][2])), v[1]] for v in candidate]
        return candidate

    @staticmethod
    def add_supplements():
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

    @staticmethod
    def remove_duplicates(old_orients):
        """Removing duplicate and similar orientations.
        Args:
            old_orients (list): list of faces
        Returns:
            Unique orientations"""
        alpha = 5  # in degrees
        tol_angle = np.sin(alpha * np.pi / 180)
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

    def project_vertices(self, orientation):
        """Supplement the mesh array with scalars (max and median)
        for each face projected onto the orientation vector.
        Args:
            orientation (np.array): with format 3 x 3.
        Returns:
            adjusted mesh.
        """
        self.mesh[:, 4, 0] = np.inner(self.mesh[:, 1, :], orientation)
        self.mesh[:, 4, 1] = np.inner(self.mesh[:, 2, :], orientation)
        self.mesh[:, 4, 2] = np.inner(self.mesh[:, 3, :], orientation)

        self.mesh[:, 5, 1] = np.max(self.mesh[:, 4, :], axis=1)
        self.mesh[:, 5, 2] = np.median(self.mesh[:, 4, :], axis=1)
        sleep(0)  # Yield, so other threads get a bit of breathing space.

    def calc_overhang(self, orientation, min_volume):
        """Calculating bottom and overhang area for a mesh regarding
        the vector n.
        Args:
            orientation (np.array): with format 3 x 3.
            min_volume (bool): minimize the support material volume or supported surfaces
        Returns:
            the total bottom size, overhang size and contour length of the mesh
        """
        total_min = np.amin(self.mesh[:, 4, :])

        # filter bottom area
        bottom = np.sum(self.mesh[np.where(self.mesh[:, 5, 1] < total_min + self.FIRST_LAY_H), 5, 0])
        # # equal than:
        # bottoms = mesh[np.where(mesh[:, 5, 1] < total_min + FIRST_LAY_H)]
        # if len(bottoms) > 0: bottom = np.sum(bottoms[:, 5, 0])
        # else: bottom = 0

        # filter overhangs
        overhangs = self.mesh[np.where(np.inner(self.mesh[:, 0, :], orientation) < self.ASCENT)]
        overhangs = overhangs[np.where(overhangs[:, 5, 1] > (total_min + self.FIRST_LAY_H))]

        if self.extended_mode:
            plafond = np.sum(overhangs[(overhangs[:, 0, :] == -orientation).all(axis=1), 5, 0])
        else:
            plafond = 0

        if len(overhangs) > 0:
            if min_volume:
                heights = np.inner(overhangs[:, 1:4, :].mean(axis=1), orientation) - total_min

                inner = np.inner(overhangs[:, 0, :], orientation) - self.ASCENT
                # overhang = np.sum(heights * overhangs[:, 5, 0] * np.abs(inner * (inner < 0)) ** 2)
                overhang = np.sum((self.height_offset + self.height_log * np.log(self.height_log_k * heights + 1)) *
                                  overhangs[:, 5, 0] * np.abs(inner * (inner < 0)) ** self.OV_H)
            else:
                # overhang = np.sum(overhangs[:, 5, 0] * 2 *
                #                   (np.amax((np.zeros(len(overhangs)) + 0.5,
                #                             - np.inner(overhangs[:, 0, :], orientation)),
                #                            axis=0) - 0.5) ** 2)
                # improved performance by finding maximum using the multiplication method, see:
                # https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
                inner = np.inner(overhangs[:, 0, :], orientation) - self.ASCENT
                overhang = 2 * np.sum(overhangs[:, 5, 0] * np.abs(inner * (inner < 0)) ** 2)
            overhang -= self.PLAFOND_ADV * plafond

        else:
            overhang = 0

        # filter the total length of the bottom area's contour
        if self.extended_mode:
            # contours = self.mesh[total_min+self.FIRST_LAY_H < self.mesh[:, 5, 1]]
            contours = self.mesh[np.where(self.mesh[:, 5, 2] < total_min + self.FIRST_LAY_H)]

            if len(contours) > 0:
                conlen = np.arange(len(contours))
                sortsc0 = np.argsort(contours[:, 4, :], axis=1)[:, 0]
                sortsc1 = np.argsort(contours[:, 4, :], axis=1)[:, 1]

                con = np.array([np.subtract(
                    contours[conlen, 1 + sortsc0, :],
                    contours[conlen, 1 + sortsc1, :])])

                contours = np.sum(np.power(con, 2), axis=-1) ** 0.5
                contour = np.sum(contours) + self.CONTOUR_AMOUNT * len(contours)
            else:
                contour = 0
        else:  # consider the bottom area as square, bottom=a**2 ^ contour=4*a
            contour = 4 * np.sqrt(bottom)

        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return bottom, overhang, contour

    def update_progress(self, new_progress):
        self._progress = new_progress
        if self.show_progress:
            os.system('cls')
            print("Progress is: {progress} ".format(progress=new_progress))
        if self.progress_callback:
            self.progress_callback(new_progress)

    def euler(self, bestside):
        """Calculating euler rotation parameters and rotational matrix.
        Args:
            bestside (np.array): vector of the best orientation (3 x 3).
        Returns:
            rotation axis, rotation angle, rotational matrix.
        """
        if np.allclose(bestside[0], np.array([0, 0, -1]), atol=abs(self.VECTOR_TOL)):
            rotation_axis = [1, 0, 0]
            phi = np.pi
        elif np.allclose(bestside[0], np.array([0, 0, 1]), atol=abs(self.VECTOR_TOL)):
            rotation_axis = [1, 0, 0]
            phi = 0
        else:
            phi = np.pi - np.arccos(-bestside[0][2])
            rotation_axis = [-bestside[0][1], bestside[0][0], 0]  # the z-axis is fixed to 0 for this rotation
            rotation_axis = [i / np.linalg.norm(rotation_axis) for i in rotation_axis]  # normalization

        v = rotation_axis
        rotational_matrix = np.array([[v[0] * v[0] * (1 - math.cos(phi)) + math.cos(phi),
                                       v[0] * v[1] * (1 - math.cos(phi)) - v[2] * math.sin(phi),
                                       v[0] * v[2] * (1 - math.cos(phi)) + v[1] * math.sin(phi)],
                                      [v[1] * v[0] * (1 - math.cos(phi)) + v[2] * math.sin(phi),
                                       v[1] * v[1] * (1 - math.cos(phi)) + math.cos(phi),
                                       v[1] * v[2] * (1 - math.cos(phi)) - v[0] * math.sin(phi)],
                                      [v[2] * v[0] * (1 - math.cos(phi)) - v[1] * math.sin(phi),
                                       v[2] * v[1] * (1 - math.cos(phi)) + v[0] * math.sin(phi),
                                       v[2] * v[2] * (1 - math.cos(phi)) + math.cos(phi)]], dtype=np.float64)
        # rotational_matrix = np.around(rotational_matrix, decimals=6)
        sleep(0)  # Yield, so other threads get a bit of breathing space.
        return rotation_axis, phi, rotational_matrix
