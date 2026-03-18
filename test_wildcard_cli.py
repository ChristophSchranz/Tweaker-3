import os
import runpy
import sys
import types
from pathlib import Path

import pytest


TWEAKER_PATH = Path(__file__).resolve().parent / "Tweaker.py"


def _install_stubs(monkeypatch, record):
    # Stub MeshTweaker so importing Tweaker.py never pulls in the real numpy-heavy stack.
    mesh_mod = types.ModuleType("MeshTweaker")

    class Tweak:
        def __init__(self, *args, **kwargs):
            # Not used in convert mode (-c), but keep attributes in case tests change.
            self.matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.alignment = [0, 0, 1]
            self.rotation_axis = [1, 0, 0]
            self.rotation_angle = 0.0
            self.unprintability = 0.0
            self.euler_parameter = [self.rotation_axis, self.rotation_angle]
            self.bottom_area = 0.0
            self.overhang_area = 0.0
            self.contour = 0.0
            self.best_5 = []
            self.time = 0.0

    mesh_mod.Tweak = Tweak

    # Stub FileHandler so no files are actually parsed/written.
    fh_mod = types.ModuleType("FileHandler")

    class FileHandler:
        def __init__(self):
            pass

        def load_mesh(self, inputfile):
            record["loaded"].append(inputfile)
            # One "part" is enough for Tweaker.py's loops.
            return {0: {"mesh": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "name": "part0"}}

        def write_mesh(self, objects, info, outputfile, output_type="binarystl"):
            record["written"].append(
                {"outputfile": outputfile, "output_type": output_type}
            )

    fh_mod.FileHandler = FileHandler

    monkeypatch.setitem(sys.modules, "MeshTweaker", mesh_mod)
    monkeypatch.setitem(sys.modules, "FileHandler", fh_mod)


def _run_cli(tmp_path, argv, monkeypatch, record):
    _install_stubs(monkeypatch, record)

    old_cwd = os.getcwd()
    old_argv = sys.argv[:]
    os.chdir(tmp_path)
    sys.argv = ["Tweaker.py", *argv]

    try:
        runpy.run_path(str(TWEAKER_PATH), run_name="__main__")
        return None
    except SystemExit as e:
        return e
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def test_wildcard_expands_and_generates_per_file_outputs(tmp_path, monkeypatch):
    (tmp_path / "input_a.stl").write_text("dummy")
    (tmp_path / "input_b.stl").write_text("dummy")

    record = {"loaded": [], "written": []}
    exc = _run_cli(
        tmp_path,
        ["-i", "input_*.stl", "-c"],
        monkeypatch,
        record,
    )
    assert exc is None

    assert record["loaded"] == ["input_a.stl", "input_b.stl"]
    assert [c["outputfile"] for c in record["written"]] == [
        "input_a_converted.stl",
        "input_b_converted.stl",
    ]


def test_wildcard_no_matches_exits_with_message(tmp_path, monkeypatch, capsys):
    record = {"loaded": [], "written": []}
    exc = _run_cli(tmp_path, ["-i", "missing_*.stl", "-c"], monkeypatch, record)
    assert exc is not None
    assert exc.code == 1

    out = capsys.readouterr().out
    assert "No input files match pattern: missing_*.stl" in out


def test_wildcard_with_explicit_o_multiple_inputs_errors(tmp_path, monkeypatch):
    (tmp_path / "input_a.stl").write_text("dummy")
    (tmp_path / "input_b.stl").write_text("dummy")

    record = {"loaded": [], "written": []}
    exc = _run_cli(
        tmp_path,
        ["-i", "input_*.stl", "-o", "out.stl", "-c"],
        monkeypatch,
        record,
    )
    assert exc is not None
    assert str(exc) == "Option '-o' cannot be used with a wildcard that matches multiple input files."
    assert record["loaded"] == []
    assert record["written"] == []


def test_single_file_no_wildcard_uses_single_outputfile(tmp_path, monkeypatch):
    (tmp_path / "input_a.stl").write_text("dummy")

    record = {"loaded": [], "written": []}
    exc = _run_cli(tmp_path, ["-i", "input_a.stl", "-c"], monkeypatch, record)
    assert exc is None

    assert record["loaded"] == ["input_a.stl"]
    assert [c["outputfile"] for c in record["written"]] == ["input_a_converted.stl"]

