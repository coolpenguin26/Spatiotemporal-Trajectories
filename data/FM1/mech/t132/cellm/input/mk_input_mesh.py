import flower.mk_cell_mesh as m
import pyvista as pv
import pandas as pd


m.mk_cell_mesh_atlas(132)

mesh = pv.read("t132_cellm.ply")
