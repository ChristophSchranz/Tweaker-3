# Tweaker-3
##The Tweaker is an auto-rotate module which finds the STL or 3MF object's optimal orientation on the printing platform to improve the efficiency of 3D printing.

Author: Christoph Schranz, 12.01.2016 

[STL-tweaker](http://www.salzburgresearch.at/blog/3d-print-positioning/)

## Quickstart:  

`python Tweaker.py -i demo_object.stl -vb`

Make sure you've installed the latest version of numpy:

`pip install numpy --upgrade`


## Extended mode:

This mode yields more reliable results, but needs more time.

`python Tweaker.py -i death_star.stl -vb -x`

## Designer Mode:

For many Designs, the smoothness of one side's surface is more important.
Therefore, orientations closer than 45 deg to a vector can be weighted.
The use of the extended mode -x is also recommeded. Here is an example on
how to favour the side x,y,z=0,-1,2.5 with a factor of 3:

`python Tweaker.py -i demo_object.stl -vb -x -fs "[[0,-1,2.5],3]"`

## Converting a 3mf object to stl without tweaking:

`python Tweaker.py -i pyramid.3mf -c`

If you want to change the default output representation to ASCII, uncomment/comment
the block in Tweaker.py as described there. (Search for "ASCII" or "binary")

## Just watch the results:

`python Tweaker.py -i demo_object.stl -r`

## Find more options:

`python Tweaker.py -h`

## Not installed numpy yet?

No Problem, the previous version 2 is completely numpy-less (but slower):

[STL-tweaker-V2](https://github.com/iot-salzburg/STL-tweaker/)

## Cura Plugin:

PlugIn for both Cura 15 and Cura 2.3 are supported. Infos are in the descriptions.

## Want to build your own application?

This [Whitepaper](https://www.researchgate.net/publication/311765131_Tweaker_-_Auto_Rotation_Module_for_FDM_3D_Printing) declares how this function works. Additionally, background infos and benchmarks are provided.
