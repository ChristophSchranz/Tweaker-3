# Python 2.7 and 3.5
# Author: Christoph Schranz, Salzburg Research

import sys
import os
import struct
import time
import traceback
import ThreeMF
# upgrade numpy with: "pip install numpy --upgrade"
import numpy as np


class FileHandler:
    def __init__(self):
        pass

    def load_mesh(self, inputfile):
        """This module loads the content of a 3D file as mesh array."""

        filetype = os.path.splitext(inputfile)[1].lower()
        if filetype == ".stl":
            f = open(inputfile, "rb")
            if not f.readable():
                raise Exception("File is not readable.")
            try:
                if "solid" in str(f.read(5).lower()):
                    try:
                        f = open(inputfile, "r")
                        objs = self.load_ascii_stl(f)
                    except UnicodeDecodeError:
                        # There are cases of binary STL with prefix 'solid', reopen file
                        f = open(inputfile, "rb")
                        f.seek(5, os.SEEK_SET)
                        objs = self.load_binary_stl(f)
                else:
                    objs = self.load_binary_stl(f)
            except Exception as ex:
                # Get current system exception
                ex_type, ex_value, ex_traceback = sys.exc_info()
                # Extract unformatter stack traces as tuples
                trace_back = traceback.extract_tb(ex_traceback)
                # Format stacktrace
                stack_trace = list()
                for trace in trace_back:
                    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
                        trace[0], trace[1], trace[2], trace[3]))

                print("""Exception of type '{}' in reading the file:\n'{}'
Stack trace: '{}' 
The file may be corrupt, please check if the file can be opened with ofter software. 
If it is readable by other software, you can help to improve this software by 
opening a github issue and attaching the file.
Best,\nyour Auto-Rotate Developer\n""".format(ex_type.__name__, str(ex), stack_trace))
                raise Exception("File is not readable.")

        elif filetype == ".3mf":
            object = ThreeMF.Read3mf(inputfile)  # TODO not implemented
            # objs[0] = {"mesh": list(), "name": "binary file"}
            objs = {0: {"mesh": object[0]["mesh"], "name": "3mf file"}}
        elif filetype == ".obj":
            f = open(inputfile, "rb")
            objs = self.load_obj(f)

        else:
            raise Exception("File type is not supported.")

        return objs

    @staticmethod
    def load_obj(f):
        """Load the content of an OBJ file."""
        objects = dict()
        vertices = list()
        objects[0] = {"mesh": list(), "name": "obj file"}
        for line in f:
            if "v" in line:
                data = line.split()[1:]
                vertices.append([float(data[0]), float(data[1]), float(data[2])])
        f.seek(0, 0)
        for line in f:
            if "f" in line:
                data = line.split()[1:]
                objects[0]["mesh"].append(vertices[int(data[0]) - 1])
                objects[0]["mesh"].append(vertices[int(data[1]) - 1])
                objects[0]["mesh"].append(vertices[int(data[2]) - 1])

        return objects

    @staticmethod
    def load_ascii_stl(f):
        """Load the content of an ASCII STL file."""
        objects = dict()
        part = 0
        objects[part] = {"mesh": list()}
        for line in f:
            if "vertex" in line:
                data = line.split()[1:]
                objects[part]["mesh"].append([float(data[0]), float(data[1]), float(data[2])])
            if "endsolid" in line:
                objects[part]["name"] = line.split()[-1]
                part += 1
                objects[part] = {"mesh": list()}

        # Delete empty parts:
        objs = dict()
        for k, v in objects.items():
            if len(v["mesh"]) > 3:
                objs[k] = v
        return objs

    @staticmethod
    def load_binary_stl(f):
        """Load the content of a binary STL file."""
        # Skip the header
        f.read(80 - 5)
        face_count = struct.unpack('<I', f.read(4))[0]
        objects = dict()
        objects[0] = {"mesh": list(), "name": "binary file"}
        for idx in range(0, face_count):
            data = struct.unpack("<ffffffffffffH", f.read(50))
            objects[0]["mesh"].append([data[3], data[4], data[5]])
            objects[0]["mesh"].append([data[6], data[7], data[8]])
            objects[0]["mesh"].append([data[9], data[10], data[11]])
        return objects

    def write_mesh(self, objects, info, outputfile, output_type="binarystl"):
        # if output_type == "3mf":  # TODO not implemented yet
        #     # transformation = "{} {} {} {} {} {} {} {} {} 0 0 1".
        #     # format(x.matrix[0][0], x.matrix[0][1], x.matrix[0][2],
        #     # x.matrix[1][0], x.matrix[1][1], x.matrix[1][2], x.matrix[2][0], x.matrix[2][1], x.matrix[2][2])
        #     #     obj["transform"] = transformation
        #     #     FileHandler.rotate3MF(args.inputfile, args.outputfile, objs)
        #     raise TypeError('The 3mf output format is not implemented yet.')

        if output_type == "asciistl":
            # Create seperate files with rotated content. If an IDE supports multipart placement,
            # set outname = outputfile
            for part, content in objects.items():
                mesh = content["mesh"]
                filename = content["name"]

                mesh = self.rotate_ascii_stl(info[part]["matrix"], mesh, filename)
                if len(objects.keys()) == 1:
                    outname = outputfile
                else:
                    outname = ".".join(outputfile.split(".")[:-1]) + "_{}.stl".format(part)
                with open(outname, 'w') as outfile:
                    outfile.write(mesh)

        else:  # binary STL, binary stl can't support multiparts
            # Create seperate files with rotated content.
            header = "Tweaked on {}".format(time.strftime("%a %d %b %Y %H:%M:%S")
                                            ).encode().ljust(79, b" ") + b"\n"
            for part, content in objects.items():
                partlength = int(len(objects[part]["mesh"]) / 3)
                mesh = self.rotate_bin_stl(info[part]["matrix"], mesh)

                if len(objects.keys()) == 1:
                    outname = outputfile
                else:
                    outname = ".".join(outputfile.split(".")[:-1]) + "_{}.stl".format(part)
                length = struct.pack("<I", partlength)
                with open(outname, 'wb') as outfile:
                    outfile.write(bytearray(header + length + b"".join(mesh)))

    @staticmethod
    def rotate_3mf(*arg):
        ThreeMF.rotate3MF(*arg)

    def rotate_ascii_stl(self, rotation_matrix, content, filename):
        """Rotate the mesh array and save as ASCII STL."""
        mesh = np.array(content, dtype=np.float64)

        # prefix area vector, if not already done (e.g. in STL format)
        if len(mesh[0]) == 3:
            row_number = int(len(content) / 3)
            mesh = mesh.reshape(row_number, 3, 3)

        # upgrade numpy with: "pip install numpy --upgrade"
        mesh = np.matmul(mesh, rotation_matrix)
        
        v0 = mesh[:, 0, :]
        v1 = mesh[:, 1, :]
        v2 = mesh[:, 2, :]
        normals = np.cross(np.subtract(v1, v0), np.subtract(v2, v0)) \
            .reshape(int(len(mesh)), 1, 3)
        mesh = np.hstack((normals, mesh))

        tweaked = list("solid %s" % filename)
        tweaked += list(map(self.write_facett, list(mesh)))
        tweaked.append("\nendsolid %s\n" % filename)
        tweaked = "".join(tweaked)
        return tweaked

    @staticmethod
    def write_facett(facett):
        return """\nfacet normal %f %f %f
        outer loop
            vertex %f %f %f
            vertex %f %f %f
            vertex %f %f %f
        endloop
    endfacet""" % (facett[0, 0], facett[0, 1], facett[0, 2], facett[1, 0],
                   facett[1, 1], facett[1, 2], facett[2, 0], facett[2, 1],
                   facett[2, 2], facett[3, 0], facett[3, 1], facett[3, 2])

    def rotate_bin_stl(self, rotation_matrix, content):
        """Rotate the object and save as binary STL. This module is currently replaced
        by the ascii version. If you want to use binary STL, please do the
        following changes in Tweaker.py: Replace "rotatebinSTL" by "rotateSTL"
        and set in the write sequence the open outfile option from "w" to "wb".
        However, the ascii version is much faster in Python 3."""
        mesh = np.array(content, dtype=np.float64)

        # prefix area vector, if not already done (e.g. in STL format)
        if len(mesh[0]) == 3:
            row_number = int(len(content) / 3)
            mesh = mesh.reshape(row_number, 3, 3)

        # upgrade numpy with: "pip install numpy --upgrade"
        mesh = np.matmul(mesh, rotation_matrix)

        v0 = mesh[:, 0, :]
        v1 = mesh[:, 1, :]
        v2 = mesh[:, 2, :]
        normals = np.cross(np.subtract(v1, v0), np.subtract(v2, v0)
                           ).reshape(int(len(mesh)), 1, 3)
        mesh = np.hstack((normals, mesh))
        # header = "Tweaked on {}".format(time.strftime("%a %d %b %Y %H:%M:%S")
        #                                 ).encode().ljust(79, b" ") + b"\n"
        # header = struct.pack("<I", int(len(content) / 3))  # list("solid %s" % filename)

        mesh = list(map(self.write_bin_facett, mesh))

        # return header + b"".join(tweaked_array)
        # return b"".join(tweaked_array)
        return mesh

    @staticmethod
    def write_bin_facett(facett):
        tweaked = struct.pack("<fff", facett[0][0], facett[0][1], facett[0][2])
        tweaked += struct.pack("<fff", facett[1][0], facett[1][1], facett[1][2])
        tweaked += struct.pack("<fff", facett[2][0], facett[2][1], facett[2][2])
        tweaked += struct.pack("<fff", facett[3][0], facett[3][1], facett[3][2])
        tweaked += struct.pack("<H", 0)

        return tweaked
