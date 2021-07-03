import pytest
import numpy as np

# import Tweaker modules
import FileHandler
from MeshTweaker import Tweak

file_handler = FileHandler.FileHandler()


@pytest.mark.skip("Test routine")
def test_routine(file_name, target_alignment, kwargs):
    objs = file_handler.load_mesh(file_name)

    info = dict()
    for part, content in objs.items():
        mesh = content["mesh"]
        info[part] = dict()
        x = Tweak(mesh, **kwargs)

        info[part]["alignment"] = x.alignment
        info[part]["rotation_axis"] = x.rotation_axis
        info[part]["rotation_angle"] = x.rotation_angle
        info[part]["euler_parameter"] = x.euler_parameter
        info[part]["bottom_area"] = x.bottom_area
        info[part]["overhang_area"] = x.overhang_area
        info[part]["contour"] = x.contour
        info[part]["unprintability"] = x.unprintability
        info[part]["best_5"] = x.best_5
        info[part]["time"] = x.time
        info[part]["tweaker_stats"] = x
        print("Object: {}, result: new alignment: \t{}".format(file_name, x.alignment))

        assert np.allclose(info[part]["alignment"], target_alignment, atol=1e-4)


# Testing four main configs on demo_object
def test_11():
    file_name = "demo_object.stl"
    kwargs = dict({"extended_mode": True, "min_volume": True, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0., -0.64278733, -0.76604468], kwargs=kwargs)


def test_12():
    file_name = "demo_object.stl"
    kwargs = dict({"extended_mode": True, "min_volume": False, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0., -0.64278733, -0.76604468], kwargs=kwargs)


def test_13():
    file_name = "demo_object.stl"
    kwargs = dict({"extended_mode": False, "min_volume": True, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0., -0.64278733, -0.76604468], kwargs=kwargs)


def test_14():
    file_name = "demo_object.stl"
    kwargs = dict({"extended_mode": False, "min_volume": False, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0., -0.64278733, -0.76604468], kwargs=kwargs)


# Testing with extended mode only on death_star
def test_21():
    file_name = "death_star.stl"
    kwargs = dict({"extended_mode": True, "min_volume": True, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0.422389,  0.069502, -0.903746], kwargs=kwargs)


def test_22():
    file_name = "death_star.stl"
    kwargs = dict({"extended_mode": True, "min_volume": False, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0.422389,  0.069502, -0.903746], kwargs=kwargs)


# Testing fav-side option for each configuration
def test_31():
    file_name = "demo_object.stl"
    kwargs = dict({"extended_mode": True, "min_volume": True, "favside": "[[1,1,2.3],2.5]", "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0., -0.64278733, -0.76604468], kwargs=kwargs)


def test_32():
    file_name = "demo_object.stl"
    kwargs = dict({"extended_mode": True, "min_volume": False, "favside": "[[1,1,2.3],2.5]", "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0., -0.64278733, -0.76604468], kwargs=kwargs)


def test_33():
    file_name = "demo_object.stl"
    kwargs = dict({"extended_mode": False, "min_volume": True, "favside": "[[1,1,2.3],2.5]", "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0., -0.64278733, -0.76604468], kwargs=kwargs)


def test_34():
    file_name = "demo_object.stl"
    kwargs = dict({"extended_mode": False, "min_volume": False, "favside": "[[1,1,2.3],2.5]", "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0., -0.64278733, -0.76604468], kwargs=kwargs)


# Testing four main configs on 3mf object
def test_41():
    file_name = "pyramid.3mf"
    kwargs = dict({"extended_mode": True, "min_volume": True, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0.57735027,  0.57735027,  0.57735027], kwargs=kwargs)


def test_42():
    file_name = "pyramid.3mf"
    kwargs = dict({"extended_mode": True, "min_volume": False, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0.57735027,  0.57735027,  0.57735027], kwargs=kwargs)


def test_43():
    file_name = "pyramid.3mf"
    kwargs = dict({"extended_mode": False, "min_volume": True, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0.57735027,  0.57735027,  0.57735027], kwargs=kwargs)


def test_44():
    file_name = "pyramid.3mf"
    kwargs = dict({"extended_mode": False, "min_volume": False, "favside": None, "verbose": False})
    test_routine(file_name=file_name, target_alignment=[-0.57735027,  0.57735027,  0.57735027], kwargs=kwargs)
