import plegma.mesh as mesh
import pyvista as pv
import flower.init as init


n = 3
m = pv.read("t132_seg_smooth.ply")

for i in range(n):
    mesh_ = mesh.remesh(m, n=2000)

mesh_.save("t132_seg_smooth_2k.ply", binary=False)

with open("t132.init", "w") as fout:
    fout.write(init.to_init(mesh_).to_init())
