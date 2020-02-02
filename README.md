# Tweaker-3
## The Tweaker is an auto-rotate module which finds the STL or 3MF object's optimal orientation on the printing platform to improve the efficiency of 3D printing.

![Auto-rotation of a model](https://github.com/ChristophSchranz/Tweaker-3/blob/master/auto-rotation.png)
Author: Christoph Schranz, 12.01.2016 

[Salzburg Research](http://www.salzburgresearch.at/blog/3d-print-positioning/)

## Quickstart:  

`python Tweaker.py -i demo_object.stl -vb`


### Extended mode:

This mode yields the most reliable results, but needs more computation time.

`python Tweaker.py -i death_star.stl -vb -x`


### Minimise the necessary support material:

If you want to optimise the print in terms of minimal support material volume, add the flag -vol.
The extended mode is suggested for this operation.

`python Tweaker.py -i demo_object.stl -vol -x`



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
extended mode -x is also recommended. Here is an example 
of how to favour the side x,y,z=0,-1,2.5 with a weighting factor 
of 3:

`python Tweaker.py -i demo_object.stl -vb -x -fs "[[0,-1,2.5],3]"`


### Find Help:

`python Tweaker.py -h`

### Version:

`python Tweaker.py -v`


## Plugins:

### Cura 15

There is a plugin for Cura 15.04 in the directory `AutoRotate Plugin/Cura15_04`. This version of Cura, however, is outdated and later versions are recommended.

### Cura 2, 3 and 4

In recent versions of Cura, the **Tweaker-3** is available as **Auto-Orientation** in the *Marketplace* respectively the *plugin manager*.

### OctoPrint

The **Tweaker-3** is available as **PrePrint-Service** in OctoPrint and can be installed as described in this 
[instruction](https://plugins.octoprint.org/plugins/preprintservice/). 
Note that this module combines the auto-rotation functionality with slicing in an external Docker service.

### Webservice

Another way to test the functionality of the **Tweaker-3** is the usage of a Docker-based web service
as described in this [PrePrintService-repository](https://github.com/ChristophSchranz/Pre-Print-Service).


## Interested in how the algorithm works?

This [Whitepaper](https://www.researchgate.net/publication/311765131_Tweaker_-_Auto_Rotation_Module_for_FDM_3D_Printing) 
declares this program. Additionally, background 
infos and benchmarks are provided.


## Donation

Most of this code was developed in my spare time to provide a performant auto-rotation module to the open-source 3D printing community.
If you like this project or it helps you to reduce time to develop, I would be very thankful about a cup of coffee :) 

[![More coffee, more code](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=RG7UBJMUNLMHN&source=url)
