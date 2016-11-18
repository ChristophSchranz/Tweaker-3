# Python 2.7 and 3.5
# Author: Christoph Schranz, Salzburg Research

import sys, os
import struct, time
import ThreeMF


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
        face=[]
        mesh=[]
        i=0

        rotated_content=list(map(self.rotate_vert, content, [R]*len(content)))
        
        for li in rotated_content:      
            face.append(li)
            i+=1
            if i%3==0:
                mesh.append([face[0],face[1],face[2]])
                face=[]

        mesh = map(self.calc_nomal, mesh)

        tweaked = list("solid %s" % filename)
        tweaked += list(map(self.write_facett, mesh))
        tweaked.append("\nendsolid %s\n" % filename)
        tweaked = "".join(tweaked)
        
        return tweaked

    def rotate_vert(self, a, R):
        return [a[0]*R[0][0]+a[1]*R[1][0]+a[2]*R[2][0],
                              a[0]*R[0][1]+a[1]*R[1][1]+a[2]*R[2][1],
                              a[0]*R[0][2]+a[1]*R[1][2]+a[2]*R[2][2]]
    def calc_nomal(self, face):
        v=[face[1][0]-face[0][0],face[1][1]-face[0][1],face[1][2]-face[0][2]]
        w=[face[2][0]-face[0][0],face[2][1]-face[0][1],face[2][2]-face[0][2]]
        a=[v[1]*w[2]-v[2]*w[1],v[2]*w[0]-v[0]*w[2],v[0]*w[1]-v[1]*w[0]]        
        return [[a[0],a[1],a[2]],face[0],face[1],face[2]]
    
    def write_facett(self, facett):
        return"""\nfacet normal %f %f %f
    outer loop
        vertex %f %f %f
        vertex %f %f %f
        vertex %f %f %f
    endloop
endfacet""" % (facett[0][0], facett[0][1], facett[0][2], facett[1][0], 
               facett[1][1], facett[1][2], facett[2][0], facett[2][1], 
                facett[2][2], facett[3][0], facett[3][1], facett[3][2])

    def rotatebinSTL(self, R, content, filename):
        '''Rotate the object and save as binary STL. This module is currently replaced
        by the ascii version. If you want to use binary STL, please do the
        following changes in Tweaker.py: Replace "rotatebinSTL" by "rotateSTL"
        and set in the write sequence the open outfile option from "w" to "wb".
        However, the ascii version is much faster in Python 3.'''
        face=[]
        mesh=[]
        i=0

        rotated_content=list(map(self.rotate_vert, content, [R]*len(content)))
        
        for li in rotated_content:      
            face.append(li)
            i+=1
            if i%3==0:
                mesh.append([face[0],face[1],face[2]])
                face=[]

        mesh = map(self.calc_nomal, mesh)

        tweaked = "Tweaked on {}".format(time.strftime("%a %d %b %Y %H:%M:%S")
                                ).encode().ljust(79, b" ") + b"\n"
        tweaked += struct.pack("<I", int(len(content)/3)) #list("solid %s" % filename)
        #tweaked += list(map(self.write_bin_facett, mesh))
        for facett in mesh:
            tweaked += struct.pack("<fff", facett[0][0], facett[0][1], facett[0][2])
            tweaked += struct.pack("<fff", facett[1][0], facett[1][1], facett[1][2])
            tweaked += struct.pack("<fff", facett[2][0], facett[2][1], facett[2][2])
            tweaked += struct.pack("<fff", facett[3][0], facett[3][1], facett[3][2])
            tweaked += struct.pack("<H", 0)
            
        return tweaked
