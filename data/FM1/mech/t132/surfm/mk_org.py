import pyvista as pv
import common.seg as seg
import flower.output as output
import numpy as np


def get_ts(m, n=1):
    ccs = m.cell_centers().points[::n]
    
    cells = []

    gexprs = {}
    
    for i in range(ccs.shape[0]):
        x, y, z = ccs[i, :]
        print(m.cell_data["cell variable 7"][::n][i])
        cells.append(seg.Cell(cid=i, pos=seg.Vec3(x, y, z), vol=0,
                              exprs={}))

        gexprs[i] = {"stress": m.cell_data["cell variable 7"][::n][i],
                     "CUC1_2_3": 0.0}

    for c in cells:
        c.geneNms = ["stress", "CUC1_2_3"]

            
    cells = {c.cid: c for c in cells}

    ts = seg.STissue(cells=cells, neighs={}, gexprs=gexprs, geneNms=["stress", "CUC1_2_3"])

    return ts


def get_last_while(ms, strain_thr=0.02):
    ms_ = [m for m in ms if np.max(m.cell_data["cell variable 11"]) < strain_thr]
 
    return ms_[-1]


def go(d, n=1):
    out = output.TissueOutput.from_dir(d, everyN=10)

    m = get_last_while(out.tcells)

    return get_ts(m, n=n)
    










