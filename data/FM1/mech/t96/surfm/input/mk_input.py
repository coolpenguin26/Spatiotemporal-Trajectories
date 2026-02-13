import flower.init as init
import pyvista as pv

mesh_ = pv.read("t96_smooth_ideal_05ts.ply")

with open("t96.init", "w") as fout:
    fout.write(init.to_init(mesh_).to_init())
