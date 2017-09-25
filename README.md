# Tweaker-3
## The Tweaker is an auto-rotate module which finds the STL or 3MF object's optimal orientation on the printing platform to improve the efficiency of 3D printing.

![Auto-rotation of a model](https://github.com/ChristophSchranz/Tweaker-3/blob/master/auto-rotation.png)
Author: Christoph Schranz, 12.01.2016 

[Salzburg Resarch](http://www.salzburgresearch.at/blog/3d-print-positioning/)

## Quickstart:  

`python Tweaker.py -i demo_object.stl -vb`


### Extended mode:

This mode yields the most reliable results, but needs more computation time.

`python Tweaker.py -i death_star.stl -vb -x`


### Convert a 3mf object to stl without tweaking:

`python Tweaker.py -i pyramid.3mf -c`


### Choose the output type of the STL format:

`python Tweaker.py -i pyramid.3mf -t asciistl`

You can choose the output types "asciistl" and 
"binarystl" (default). "3mf" is not supported yet.


### Just see the results:

`python Tweaker.py -i demo_object.stl -r`


### Show the progress of tweaking:

`python Tweaker.py -i demo_object.stl -x -p`


### Designer Mode:

In some cases the smoothness of one side's surface 
may be more important. Therefore, orientations closer than 
45 degrees to a vector can be weighted. The use of the 
extended mode -x is also recommeded. Here is an example 
of how to favour the side x,y,z=0,-1,2.5 with a factor 
of 3:

`python Tweaker.py -i demo_object.stl -vb -x -fs "[[0,-1,2.5],3]"`


### Find Help:

`python Tweaker.py -h`

### Version:

`python Tweaker.py -v`

## Cura Plugin:

Cura 15 and Cura 2.3 are supported. Installation infos 
are in the PlugIn folder, or you can also download the 
PlugIn from Cura 2.7 PlugIn Browser.


## Interested in how the algorithm works?

This [Whitepaper](https://www.researchgate.net/publication/311765131_Tweaker_-_Auto_Rotation_Module_for_FDM_3D_Printing) 
declares this program. Additionally, background 
infos and benchmarks are provided.
