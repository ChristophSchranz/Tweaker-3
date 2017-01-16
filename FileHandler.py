# Python 2.7 and 3.5
# Author: Christoph Schranz, Salzburg Research

import sys, os
import struct, time
import ThreeMF
# upgrade numpy with: "pip install numpy --upgrade"
import numpy as np

class FileHandler():
    def __init__(self):
        return None
        
    def loadMesh(self, inputfile):
        '''load meshs and object attributes from file'''
        ## loading mesh format
        
        filetype = os.path.splitext(inputfile)[1].lower()
        if filetype == ".stl":
            f=open(inputfile,"rb")
            if "solid" in str(f.read(5).lower()):
                f=open(inputfile,"r")
                objs = [{"Mesh": self.loadAsciiSTL(f)}]
                if len(objs[0]["Mesh"]) < 3:
                     f.seek(5, os.SEEK_SET)
                     objs = [{"Mesh": self.loadBinarySTL(f)}]
            else:
                objs = [{"Mesh": self.loadBinarySTL(f)}]
                
        elif filetype == ".3mf":
            
            objs = ThreeMF.Read3mf(inputfile)
        else:
            print("File type is not supported.")
            sys.exit()

        return objs


    def loadAsciiSTL(self, f):
        '''Reading mesh data from ascii STL'''
        mesh=list()
        for line in f:
            if "vertex" in line:
                data=line.split()[1:]
                mesh.append([float(data[0]), float(data[1]), float(data[2])])
        return mesh

    def loadBinarySTL(self, f):
        '''Reading mesh data from binary STL'''
        	#Skip the header
        f.read(80-5)
        faceCount = struct.unpack('<I', f.read(4))[0]
        mesh=list()
        for idx in range(0, faceCount):
            data = struct.unpack("<ffffffffffffH", f.read(50))
            mesh.append([data[3], data[4], data[5]])
            mesh.append([data[6], data[7], data[8]])
            mesh.append([data[9], data[10], data[11]])
        return mesh


    def rotate3MF(self, *arg):
        ThreeMF.rotate3MF(*arg)
        
                  
    def rotateSTL(self, R, content, filename):
        '''Rotate the object and save as ascii STL.'''
        mesh = np.array(content, dtype=np.float64)
        
        # prefix area vector, if not already done (e.g. in STL format)
        if len(mesh[0]) == 3:
            row_number = int(len(content)/3)
            mesh = mesh.reshape(row_number,3,3)
        
        # upgrade numpy with: "pip install numpy --upgrade"
        rotated_content = np.matmul(mesh, R)

        v0=rotated_content[:,0,:]
        v1=rotated_content[:,1,:]
        v2=rotated_content[:,2,:]
        normals = np.cross( np.subtract(v1,v0), np.subtract(v2,v0)
                                ).reshape(int(len(rotated_content)),1,3)
        rotated_content = np.hstack((normals,rotated_content))

        tweaked = list("solid %s" % filename)
        tweaked += list(map(self.write_facett, list(rotated_content)))
        tweaked.append("\nendsolid %s\n" % filename)
        tweaked = "".join(tweaked)
        
        return tweaked
    
    def write_facett(self, facett):
        return"""\nfacet normal %f %f %f
    outer loop
        vertex %f %f %f
        vertex %f %f %f
        vertex %f %f %f
    endloop
endfacet""" % (facett[0,0], facett[0,1], facett[0,2], facett[1,0], 
               facett[1,1], facett[1,2], facett[2,0], facett[2,1], 
                facett[2,2], facett[3,0], facett[3,1], facett[3,2])

    def rotatebinSTL(self, R, content, filename):
        '''Rotate the object and save as binary STL. This module is currently replaced
        by the ascii version. If you want to use binary STL, please do the
        following changes in Tweaker.py: Replace "rotatebinSTL" by "rotateSTL"
        and set in the write sequence the open outfile option from "w" to "wb".
        However, the ascii version is much faster in Python 3.'''
        
        mesh = np.array(content, dtype=np.float64)
        
        # prefix area vector, if not already done (e.g. in STL format)
        if len(mesh[0]) == 3:
            row_number = int(len(content)/3)
            mesh = mesh.reshape(row_number,3,3)
        
        # upgrade numpy with: "pip install numpy --upgrade"
        rotated_content = np.matmul(mesh, R)

        v0=rotated_content[:,0,:]
        v1=rotated_content[:,1,:]
        v2=rotated_content[:,2,:]
        normals = np.cross( np.subtract(v1,v0), np.subtract(v2,v0)
                                ).reshape(int(len(rotated_content)),1,3)
        rotated_content = np.hstack((normals,rotated_content))

        header = "Tweaked on {}".format(time.strftime("%a %d %b %Y %H:%M:%S")
                    ).encode().ljust(79, b" ") + b"\n"
        header += struct.pack("<I", int(len(content)/3)) #list("solid %s" % filename)
        
        tweaked_array = list(map(self.write_bin_facett, rotated_content))

        return header + b"".join(tweaked_array)
        
        
    def write_bin_facett(self, facett):
        tweaked = struct.pack("<fff", facett[0][0], facett[0][1], facett[0][2])
        tweaked += struct.pack("<fff", facett[1][0], facett[1][1], facett[1][2])
        tweaked += struct.pack("<fff", facett[2][0], facett[2][1], facett[2][2])
        tweaked += struct.pack("<fff", facett[3][0], facett[3][1], facett[3][2])
        tweaked += struct.pack("<H", 0)
            
        return tweaked
