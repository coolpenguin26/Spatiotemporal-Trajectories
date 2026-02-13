import igl
import plegma.mesh as mp
import pyvista as pv
import common.lin as lin
import numpy as np
import common.seg as seg
import flower.mk_org as org
import flower.output as output
import pandas as pd


def mk_meshes(ts=[96, 104, 112, 120, 128, 132], npoints=5000):
    ress = seg.readRess("./data/FM1/resolutions.txt")

    for t in ts:
        print(t)
        res = np.array(ress[t])
        m  = g.get_mesh_from_segfile(f"./data/FM1/segmentation_tiffs/{t}h_segmented.tif",
                                     npoints=npoints)
        m.points = m.points * res[[2, 1, 0]]
        m.save(f"./data/FM1/meshes/{t}h.ply")

    return


def remesh(ts=[96, 104, 112, 120, 128, 132], npoints=5000, iters=3):
    for t in ts:
        print(t)
        m = pv.read(f"./data/FM1/meshes/{t}h.ply")
        for i in range(iters):
            print(i)
            m = mp.remesh(m, npoints)
            
        m.save(f"./data/FM1/meshes/{t}h.ply")


def add_curvs(ts=[96, 104, 112, 120, 128, 132]):
    for t in ts:
        print(t)
        v, f = igl.read_triangle_mesh(f"./data/FM1/meshes/{t}h.ply")
        pd1, pd2, pv1, pv2 = igl.principal_curvature(v, f)
        m = pv.read(f"./data/FM1/meshes/{t}h.ply")
        m.point_data["curv1"] = pv1
        m.point_data["curv2"] = pv2
        m.point_data["gauss_curv"] = pv1 * pv2
        m.point_data["mean_curv"] = (pv1 + pv2) / 2
        m.point_data["dev_curv"] = (pv1 - pv2) / 2

        m = m.point_data_to_cell_data()

        m.save(f"./data/FM1/meshes/{t}h.vtk")

    return

def get_meshes(ts=[96, 104, 112, 120, 128, 132]):
    ms = {}
    for t in ts:
        ms[t] = pv.read(f"./data/FM1/meshes/{t}h.vtk")

    return ms


def get_segs(ts=[96, 104, 112, 120, 128, 132]):
    tss = {}

    for t in ts:
        tss[t] = seg.STissue.fromCSV(f"./data/FM1/csv/t{t}.csv")


    return tss


def get_atlas():
    tss, _ = lin.mkSeries1(d="./data/FM1/tv/",
                           dExprs="./data/FM1/geneExpression/",
                           linDataLoc="./data/FM1/tracking_data/",
                           ft = lambda t: t in {10, 40, 96, 120, 132})

    return tss


def get_full_atlas():
    tss, linss = lin.mkSeriesGeomIm(dataDir="./data/")


    return tss, linss


def invertMap(ln):
    iLin = [list(zip(ds, repeat(m, len(ds)))) for m, ds in ln.items()]

    return dict(sum(iLin, []))


def find_nearest_p(t_, tpoints):
    current = -1
    for t in tpoints:
        if t <= t_:
            current = t
  
    return current


def transfer(ts, ts_, iLin):
    gexprs = {}
    for c in ts:
        try:
            cm = iLin[c.cid]
            gexprs[c.cid] = ts_.cells[cm].exprs
        except KeyError:
            gexprs[c.cid] = {g: False for g in seg.geneNms}
            
    sts = seg.STissue(ts.cells, ts.neighs, gexprs, seg.geneNms)

    return sts


def transfer_(tss, tss_, linss):
    tpoints = np.array(sorted(list(tss_.keys())))
    tpoints_full = np.array(sorted(list(tss.keys())))
    
    tss_full = dict()
    for t in tpoints_full[1:]:
        t_ = find_nearest_p(t, tpoints)
        print(t_, t)
        if t == t_:
            tss_full[t] = tss_[t_]
        else:
            iLin = lin.invertMap(lin.mergeLinss(lin.filterLinssBetween(linss, (t_, t))))
            tss_full[t] = transfer(tss[t], tss_[t_], iLin)

    empty_gexprs = {g: False for g in seg.geneNms}
    ts0_gexprs = {c.cid: empty_gexprs for c in tss[0]}
    tss_full[0] = seg.STissue(tss[0].cells, {}, ts0_gexprs, seg.geneNms)

    return tss_full


def add_curvs_segs(ms, tss):
    curv_feats = ["curv1", "curv2", "gauss_curv", "mean_curv", "dev_curv"]
    
    for t in ms:
        m = ms[t]
        m.points[:, [0, 1, 2]] = m.points[:, [2, 1, 0]]
        org.transfer_feat_graph_mesh(tss[t], m, feats=curv_feats)
        tss[t].geneNms = tss[t].geneNms + curv_feats
        for c in tss[t]:
            c.geneNms = c.geneNms + curv_feats

    return


def write_diff_inits():
    tss = get_segs()

    for t in tss:
        print(t)
        tss[t].geneNms = seg.geneNms
        for c in tss[t]:
            c.geneNms = seg.geneNms
        
        with open(f"./diff_model/t{t}/t{t}.init", "w") as fout:
            fout.write(tss[t].toOrganism())

    return


def add_diff(tss, ts=[96, 104, 112, 120, 128, 132]):
    for t in ts:
        print(t)
        d = pd.read_csv(f"diff_model/t{t}/cells9.csv")
    
        for i, c in enumerate(tss[t]):
            crow = d.iloc[i, :]
            
            for gnm in seg.geneNms:
                c.exprs[gnm] = crow[gnm]

    return


def add_mechs():
    tss = get_segs()

    for t, ts in tss.items():
        print(t)
        out = output.TissueOutput.from_dir(f"./data/FM1/mech/t{t}/surfm/vtk/",
                                           everyN=10)
        m = out.tcells[3]
        m.points[:, [0, 1, 2]] = m.points[:, [2, 1, 0]]
        m["stress1"] = m["cell variable 7"]
        m["stress_anisotropy"] = m["cell variable 18"]
        org.transfer_feat_graph_mesh(ts, m=m, feats=["stress1", "stress_anisotropy"])
        ts.geneNms = ts.geneNms + ["stress1", "stress_anisotropy"]
        for c in ts:
            c.geneNms = c.geneNms + ["stress1", "stress_anisotropy"]
    
        with open(f"./data/FM1/csv/t{t}_mech.csv", "w") as fout:
            fout.write(ts.toCSV())


def mk_cell_shape(f):
    from skimage.measure import regionprops
    import plegma.generate as g
    from collections import defaultdict
    
    img = g.read_image(f)
    props = regionprops(img)

    table = defaultdict(list)

    feats = {'label',
             'axis_major_length',
             'axis_minor_length',
             'inertia_tensor_eigvals',
             'area'}
    
    for i, prop in enumerate(props):
        print(i)
        for f in feats:
            if f == "inertia_tensor_eigvals":
                for i, val in enumerate(getattr(prop, f)):
                    table[f"{f}_{i}"].append(float(val))

            else:
                table[f].append(getattr(prop, f))

    shape_d = pd.DataFrame(table)

    shape_d = shape_d.rename({'label': 'id'}, axis=1)

    return shape_d
            
