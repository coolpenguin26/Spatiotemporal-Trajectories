import flower.mk_cell_mesh as m
import pyvista as pv
import pandas as pd

m.mk_cell_mesh_atlas(120)

mesh = pv.read("t120_cellm.ply")

cids = set(pd.read_csv("t120_cellm_rm_cids.csv")["Cell ID"])
cids_in = [i for i in range(mesh.n_cells) if i not in cids]

mesh.extract_cells(cids_in).extract_surface().save("t120_cellm_clean.ply")
