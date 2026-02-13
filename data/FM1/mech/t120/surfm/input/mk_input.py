import flower.init as init
import pyvista as pv

mesh_ = pv.read("t120_smooth_p2k.ply")

with open("t120.init", "w") as fout:
    fout.write(init.to_init(mesh_).to_init())
