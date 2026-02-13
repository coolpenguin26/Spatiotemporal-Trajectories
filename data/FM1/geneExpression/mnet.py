from collections import defaultdict
import os
import common.edict as edict
from os import listdir
from os.path import isfile, join
from typing import Dict, List
from common.seg import geneNms


class MorphoNetInfo():
    name: str
    info: Dict[int, Dict[int, str]]

    def __init__(self, fpath):
        _, fname = os.path.split(fpath)
        info_name, _ = os.path.splitext(fname)

        self.name = info_name
        self.info = self._parse_info(fpath)

    def _parse_info(self, fpath):
        info = defaultdict(list)
        with open(fpath, "r") as fin:
            fin.readline()  # ignore header

            for ln in fin:
                tInd, rest = ln.strip().split(",")
                cid, val = rest.strip().split(":")

                info[int(tInd)].append((int(cid), val))

        info_d = dict()
        for tid, tinfo in info.items():
            info_d[tid] = dict(tinfo)

        return info_d

    def to_named_dict(self, t: int) -> Dict[int, Dict[str, bool]]:
        try:
            infoT = self.info[t]
        except KeyError:
            return {}

        info_namedD = dict()

        for cid, val in infoT.items():
            info_namedD[cid] = {self.name: True}

        return info_namedD


def getMNetInfo(fpath: str) -> List[MorphoNetInfo]:
    infos = list()
    mnet_files = [join(fpath, f) for f in listdir(fpath)
                  if isfile(join(fpath, f))]

    for f in mnet_files:
        infos.append(MorphoNetInfo(f))

    return infos


def nonEmpty(xs):
    for x in xs:
        if x:
            return x


def combineMNetInfo(infos: List[MorphoNetInfo],
                    tpoint: int) -> Dict[int, Dict[str, bool]]:
    info_ds = [info.to_named_dict(tpoint) for info in infos]

    return edict.unionsWith(info_ds,
                            lambda ms:
                            edict.unionsWith(ms, nonEmpty, {}), {})


def writeCell(cid: int, info_c: Dict[str, bool], gNms: List[str]):

    def boolToInt(b):
        if b:
            return 1
        else:
            return 0

    return " ".join([str(cid)] +
                    [str(boolToInt(info_c.get(g, False)))
                        for g in gNms])


def writeInfo(infos_ds: Dict[int, Dict[str, bool]], gNms: List[str]) -> str:
    header = " ".join(["CID"] + [g for g in geneNms])

    return "\n".join([header] + [writeCell(cid, gexprs, gNms)
                                 for cid, gexprs in infos_ds.items()])


def writeQInfo(info_ds, lb="values"):
    from functools import reduce
    from operator import add

    header = ["type:float"]

    ts = sorted(info_ds.keys())

    lns = list()
    for i, t in enumerate(ts):
        lns.append(["{tInd}, {cid}: {val}".format(tInd=i+1, cid=cid, val=v)
                    for cid, v in info_ds[t].items()])

    with open("{lb}.txt".format(lb=lb), "w") as fout:
        content = "\n".join(header + reduce(add, lns))
        fout.write(content)


def go_dir(d: str):
    stage_to_tpoint = {1: 10,
                       2: 40,
                       3: 96,
                       4: 120,
                       5: 132}
    infos = getMNetInfo(d)

    for t in [1, 2, 3, 4, 5]:
        infos_t = combineMNetInfo(infos, t)
        info_repr = writeInfo(infos_t, geneNms)
        fname = "t_{t}h.txt".format(t=str(stage_to_tpoint[t]))

        with open(join(d, fname), "w") as fout:
            fout.write(info_repr)
