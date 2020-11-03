#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import os
from time import time

if __name__ == '__main__':
    from MeshTweaker import Tweak
    import FileHandler
else:
    from .MeshTweaker import Tweak
    from . import FileHandler

# You can preset the default model in line 42

__author__ = "Christoph Schranz, Salzburg Research"
__version__ = "3.9, October 2020"


def getargs():
    parser = argparse.ArgumentParser(description="Orientation tool for better 3D prints")
    parser.add_argument('-i ', action="store",
                        dest="inputfile", help="select input file")
    parser.add_argument('-o ', action="store", dest="outputfile", type=str,
                        help="select output file. '_tweaked' is postfix by default")
    parser.add_argument('-vb ', '--verbose', action="store_true", dest="verbose",
                        help="increase output verbosity", default=False)
    parser.add_argument('-p ', '--progress', action="store_true", dest="show_progress",
                        help="show the progress of Tweaking", default=False)
    parser.add_argument('-c ', '--convert', action="store_true", dest="convert",
                        help="convert 3mf to stl without tweaking", default=False)
    parser.add_argument('-t ', '--outputtype', action="store", dest="output_type", default=False,
                        help='set output representation [default="binarystl", "asciistl", "3mf"]')
    parser.add_argument('-x ', '--extended', action="store_true", dest="extended_mode", default=False,
                        help="using more algorithms and examine more alignments")
    parser.add_argument('-v ', '--version', action="store_true", dest="version",
                        help="print version number and exit", default=False)
    parser.add_argument('-r ', '--result', action="store_true", dest="result",
                        help="show result of calculation and exit without creating output file",
                        default=False)
    parser.add_argument('-fs', '--favside', type=str, dest="favside",
                        help="favour one orientation with a vector and weighting, e.g.  '[[0,-1,2],3]'",
                        default=None)
    parser.add_argument('-min', '--minimize', action="store", dest="minimize", default="vol",
                        help="choose to minimise overhanging surface [sur] or volume default=[vol] of support material")
    arguments = parser.parse_args()

    if arguments.version:
        print("Tweaker 3.9, (November 2020, parameter are optimized by an evolutionary algorithm)")
        return None

    if not arguments.inputfile:
        try:
            curpath = os.path.dirname(os.path.realpath(__file__))
            arguments.inputfile = curpath + os.sep + "demo_object.stl"
            # arguments.inputfile = curpath + os.sep + "death_star.stl"
            # arguments.inputfile = curpath + os.sep + "pyramid.3mf"
            # arguments.inputfile = curpath + os.sep + "3DBenchy2.stl"
            arguments.inputfile = curpath + os.sep + "all.stl"
        except FileNotFoundError:
            return None
    if arguments.minimize:
        if "sur" in arguments.minimize.lower():
            arguments.volume = False
        elif "vol" in arguments.minimize.lower():
            arguments.volume = True
        else:
            print("Can't understand input '-min {}', using 'vol'.".format(arguments.minimize))
            arguments.volume = True
    if arguments.output_type:
        # print(arguments.output_type)
        if "3mf" in arguments.output_type.lower():
            filetype = "3mf"
        elif "asci" in arguments.output_type.lower():
            filetype = "asciistl"
        else:
            filetype = "binarystl"
    else:
        if "3mf" in os.path.splitext(arguments.inputfile)[1]:
            filetype = "3mf"
        else:
            filetype = "binarystl"
    arguments.output_type = filetype
    # print("Tweaker, arguments.output_type", arguments.output_type)
    if arguments.outputfile:
        filetype = arguments.outputfile.split(".")[-1].lower()
        if filetype not in ["stl", "3mf", "obj"]:
            raise TypeError("Filetype not supported")
        arguments.outputfile = ".".join(arguments.outputfile.split(".")[:-1]) + "." + filetype
        if not arguments.output_type:
            arguments.output_type = filetype
    else:
        if arguments.convert:
            arguments.outputfile = os.path.splitext(arguments.inputfile)[0] + "_converted"
        else:
            arguments.outputfile = os.path.splitext(arguments.inputfile)[0] + "_tweaked"

        if arguments.output_type == "3mf":
            arguments.outputfile += ".3mf"  # TODO not supported yet
        else:
            arguments.outputfile += ".stl"

    argv = sys.argv[1:]
    if len(argv) == 0:
        print("""No additional arguments. Testing calculation with 
demo object in verbose mode. Use argument -h for help.
""")
        arguments.convert = False
        arguments.verbose = False  # True
        # arguments.show_progress = True
        arguments.extended_mode = True
        arguments.favside = None  # "[[0,-0.5,1],2.5]"
        # arguments.output_type = "asciistl"
        # arguments.volume = True

    return arguments


if __name__ == "__main__":
    # Get the command line arguments. Run in IDE for demo tweaking.
    stime = time()
    try:
        args = getargs()
        if args is None:
            sys.exit()
    except:
        raise

    try:
        FileHandler = FileHandler.FileHandler()
        objs = FileHandler.load_mesh(args.inputfile)
        if objs is None:
            sys.exit()
    except(KeyboardInterrupt, SystemExit):
        raise SystemExit("Error, loading mesh from file failed!")

    # Start of tweaking.
    if args.verbose:
        print("Calculating the optimal orientation:\n  {}"
              .format(args.inputfile.split(os.sep)[-1]))

    c = 0
    info = dict()
    for part, content in objs.items():
        mesh = content["mesh"]
        info[part] = dict()
        if args.convert:
            info[part]["matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            try:
                cstime = time()
                x = Tweak(mesh, args.extended_mode, args.verbose, args.show_progress, args.favside, args.volume)
                info[part]["matrix"] = x.matrix
                info[part]["tweaker_stats"] = x
            except (KeyboardInterrupt, SystemExit):
                raise SystemExit("\nError, tweaking process failed!")

            # List tweaking results
            if args.result or args.verbose:
                print("Result-stats:")
                print(" Tweaked Z-axis: \t{}".format(x.alignment))
                print(" Axis, angle:   \t{}".format(x.euler_parameter))
                print(""" Rotation matrix: 
            {:2f}\t{:2f}\t{:2f}
            {:2f}\t{:2f}\t{:2f}
            {:2f}\t{:2f}\t{:2f}""".format(x.matrix[0][0], x.matrix[0][1], x.matrix[0][2],
                                          x.matrix[1][0], x.matrix[1][1], x.matrix[1][2],
                                          x.matrix[2][0], x.matrix[2][1], x.matrix[2][2]))
                print(" Unprintability: \t{}".format(x.unprintability))

                print("Found result:    \t{:2f} s\n".format(time() - cstime))

    if not args.result:
        try:
            FileHandler.write_mesh(objs, info, args.outputfile, args.output_type)
        except FileNotFoundError:
            raise FileNotFoundError("Output File '{}' not found.".format(args.outputfile))

    # Success message
    if args.verbose:
        print("Tweaking took:  \t{:2f} s".format(time() - stime))
        print("Successfully Rotated!")
