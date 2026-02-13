# trajectories

```

### Data loading
#for one timepoint t=132
import pandas as pd

d = pd.read_csv("./data/FM1/csv/t132.csv")
shape_feats = ['curv1', 'curv2', 'dev_curv', 'gauss_curv', 'mean_curv', 'vol']
gene_feats = ['AG',
	      'AHP6',
	      'ANT',
	      'AP1',
	      'AP2',
	      'AP3',
	      'AS1',
	      'ATML1',
	      'CLV3',
	      'CUC1_2_3',
	      'ETTIN',
	      'FIL',
	      'LFY',
	      'MP',
	      'PHB_PHV',
	      'PI',
	      'PUCHI',
	      'REV',
	      'SEP1',
	      'SEP2',
	      'SEP3',
	      'STM',
	      'SUP',
	      'SVP',
	      'WUS']


d_shape = d[shape_feats]
d_gene = d[gene_feats]
```

If you look at this code: [genes-shape repository](https://gitlab.developers.cam.ac.uk/slcu/teamhj/students/sashab/genes-shape/-/tree/main/sasha's_code_new)
there are functions for dimensionality reduction and clustering.


### Lineage processing
I added some code in `mk_lins.py` (`get_inverted_linss` function) to load and invert the lineages. You'll need the library `common` from here: [common](https://gitlab.developers.cam.ac.uk/slcu/teamhj/argyris/common).