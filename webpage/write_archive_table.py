import os
import numpy as np
import pandas as pd


tag = 'cosmos_bluecore'
imdir = f'../thumbnails/cosmos_targets/cosmos_1sig_interesting/{tag}'
table_fn = f'tables/{tag}.csv'
cat_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_cosmos.csv'
cat = pd.read_csv(cat_fn)
cat = cat.set_index('Unnamed: 0')

# get objects in folder, write readable table
if os.path.isfile(table_fn):
	print(f"Reading table {table_fn}...")
	table = np.loadtxt(table_fn, delimiter=',')
	idxs = np.array(table[:,-1], dtype=int)
else:
	print(f"Getting files from image dir {imdir}")
	idxs = []
	for fn in os.listdir(imdir):
	    prestr = 'idx'
	    idxstr = fn.split('_')[1][len(prestr):]
	    idxs.append(int(idxstr))
	np.set_printoptions(precision=15)
	
	ras, decs, objs = [], [], []
	header = 'object_id, ra, dec, hsc_id'
	ids_cat = []
	for i in range(len(idxs)):
	    idx = idxs[i]
	    row = cat.loc[idx]
	    print(str(row['object_id']), idx, row['ra_x'], row['dec_x'])
	    objs.append(str(row['object_id']))
	    ras.append(row['ra_x'])
	    decs.append(row['dec_x'])
	    
	data = np.array([objs, ras, decs, idxs])
	np.savetxt(table_fn, data.T, delimiter=',', header=header, fmt='%s')
	
# write table properly formatted for archive
header_table =("| name   | ra           | dec          |\n"
               "| char   | double       | double       |")
ras = []
decs = []
for i in range(len(idxs)):
    idx = idxs[i]

    row = cat.loc[idx]
    #objid = row['object_id']
    ras.append(row['ra_x'])
    decs.append(row['dec_x'])
    
data = np.array([idxs, ras, decs])
np.savetxt(f'tables/{tag}_archive.dat', data.T, delimiter=' ', header=header_table, comments='', fmt=['%d', '%s','%s']) #fmt=['%17s', '%.14f', '%.16f', '%d'])
