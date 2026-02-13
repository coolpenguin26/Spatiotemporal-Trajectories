import flower.mk_cell_mesh as m
import pyvista as pv
import pandas as pd


m.mk_cell_mesh_atlas(96)

mesh = pv.read("t96_cellm.ply")
cids = set(pd.read_csv("t96_cellm_removed_cids.csv")["Cell ID"])
cids_in = [i for i in range(mesh.n_cells) if i not in cids]


mesh.extract_cells(cids_in).extract_surface().save("t96_cellm_clean.ply")
