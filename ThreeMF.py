# Python 2.7 and 3.5
# Author: Christoph Schranz, Salzburg Research

import sys, os
import struct
import time
import zipfile
import xml.etree.ElementTree as ET


namespace = {
    "3mf": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
    "m" : "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
}

def Read3mf(f):
    '''load parts of the 3mf with their properties'''
    # The base object of 3mf is a zipped archive.
    archive = zipfile.ZipFile(f, "r")
    try:
        root = ET.parse(archive.open("3D/3dmodel.model"))

        # There can be multiple objects, try to load all of them.
        objects = root.findall("./3mf:resources/3mf:object", namespace)
        if len(objects) == 0:
            print("No objects found in 3MF file %s, either the file is damaged or you are using an outdated format", f)
            return None
        
        obj_meshs = list()
        c=0
        for obj in objects:
            if obj.findall(".//3mf:mesh", namespace) == []:
                continue
            obj_meshs.append(dict())
            
            objectid = obj.get("id")
            obj_meshs[c]["objectid"] = objectid
            
            vertex_list = []
            obj_meshs[c]["Mesh"] = list()
            #for vertex in object.mesh.vertices.vertex:
            for vertex in obj.findall(".//3mf:vertex", namespace):
                vertex_list.append([vertex.get("x"), vertex.get("y"), vertex.get("z")])
                
            triangles = obj.findall(".//3mf:triangle", namespace)
            #for triangle in object.mesh.triangles.triangle:
            for triangle in triangles:
                v1 = int(triangle.get("v1"))
                v2 = int(triangle.get("v2"))
                v3 = int(triangle.get("v3"))
                obj_meshs[c]["Mesh"].append([float(vertex_list[v1][0]),float(vertex_list[v1][1]),float(vertex_list[v1][2])])
                obj_meshs[c]["Mesh"].append([float(vertex_list[v2][0]),float(vertex_list[v2][1]),float(vertex_list[v2][2])])
                obj_meshs[c]["Mesh"].append([float(vertex_list[v3][0]),float(vertex_list[v3][1]),float(vertex_list[v3][2])])


            try:
                obj_meshs[c]["Transform"] = getTransformation(root, objectid)
            except:
                pass

##            try:
##                color_list = list()
##                colors = root.findall('.//m:color', namespace)
##                if colors:
##                    for color in colors:
##                        color_list.append(color.get("color",0))
##                    obj_meshs[c]["color"] = color_list
##            except AttributeError:
##                pass # Empty list was found. Getting transformation is not possible

            c=c+1

            
    except Exception as e:
        print("exception occured in 3mf reader: %s" % e)
        return None
    return obj_meshs



def getTransformation(root, objectid):
    builds = root.findall(".//3mf:item", namespace)
    transforms = list()
    for item in builds:
        if item.get("transform"):
            transforms.append((item.get("objectid"), item.get("transform")))
    components = root.findall(".//3mf:components", namespace)
    objects = root.findall("./3mf:resources/3mf:object", namespace)
    for (transid, transform) in transforms:
        for obj in objects:
            if transid == obj.get("id"):
                obj_ids = obj.findall(".//3mf:component", namespace)
                for obj_id in obj_ids:
                    if obj_id.get("objectid") == objectid:
                              #print(transform)
                              break
    return transform
                              
def rotate3MF(f, outfile, objs):
    #TODO doesn't work at the moment
    archive = zipfile.ZipFile(f, "r")
    root = ET.parse(archive.open("3D/3dmodel.model"))
    
    for obj in objs:
        itemid = None
        # get build id for transform value
        objects3MF = root.findall("./3mf:resources/3mf:object", namespace)
        for elem in objects3MF:
            for component in elem.findall(".//3mf:component", namespace):
                    if component.get("objectid") == obj["objectid"]:
                            #print("objid", elem.get("id"))
                            itemid = elem.get("id")

        if itemid:
            for item in root.findall(".//3mf:build/3mf:item", namespace):
                if item.get("objectid") == itemid:
                    item.set("transform", obj["transform"])
        else:
            pass

    #Writing the changed model in the output file
    indir = os.path.splitext(f)[0]
    zipf = zipfile.ZipFile(outfile, 'w', zipfile.ZIP_DEFLATED)
    zipdir(indir, zipf)
    zipf.close()


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))
