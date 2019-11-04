#!/usr/bin/env python
"""
makes a set of web images for cosmos HSC data
"""

import sys,os,re
import glob
import collections

import pandas as pd


def writeInfo(galid, **kwargs):
	"""
	write extra information about the galaxy, given by keyword arguments
	"""
	ext_dict = {0.0: 'compact', 1.0:'extended'}
	info='<td>\n<table>\n'
	#info='<td>'
	info += "<tr><td>info_id:</td><td>%s</td></tr>\n" % repr(int(galid))
	skeys = kwargs.keys() #sorted(kwargs.keys(), key=str.lower)
	for key in skeys:
		val = str(kwargs[key])
		if 'extendedness' in key:
			val += ' ('+ext_dict[float(val)]+')' 
		info += "<tr><td>%s:</td><td>%s</td></tr>\n" % (key, val)
	info += "</table>\n</td>\n"
	#info+='</td>'
	return info
	
def readInfo(objidx, info_df):
	"""
	reads a text file with parameters
	"""
	params=['object_id', 'ra_x', 'dec_x', 'i_extendedness_value']
	info_dict = {}
	for p in params:
		info_dict[p] = info_df[p].loc[int(objidx)]
	return info_dict

def main(folder, info_fn, thumbnail_folder, outputname):

	info_df = pd.read_csv(info_fn)
	info_df = info_df.set_index('Unnamed: 0')
	print("Read in info file {} with {} rows".format(info_fn, info_df.size))
	filepatterns = [r'.*idx([0-9]*).*.png']    
	datasets = ['COSMOS', 'acs']
	#get the images and sort them into a dictionary of galaxies/objects
	imgfiles = glob.glob(folder+'/*')
	#thumbfiles = glob.glob(thumbnail_folder+'/*')
	objcs = collections.defaultdict(list)
	for pattern in filepatterns:
		for f in imgfiles:
			print(f)
			match = re.search(pattern, f)
			if match:
				print(match.group(1))
				objcs[match.group(1)].append(f)

	with open(outputname+'.html', 'w') as out:
		out.write("""<html>
		<body>
		<table border="1">
		""")
		
		headers = ['HSC', 'info', 'Subaru', 'HST']
		out.write("<tr>\n")
		for head in headers:
			out.write("<th>%s</th>\n"%head)
		out.write("<tr>\n")
			

		
		#sort the object keys by name (which is halo mass)
		keys = sorted(objcs.keys(), key=float, reverse=True)

		for iobj, obj in enumerate(keys):
			out.write("<tr>\n")
			#out.write('<td>\n<table>\n')
			#out.write(writeInfo(iobj, **readInfo(objcs[obj][0])))
			#out.write('<tr><td><img src="%s"></td></tr>\n'%objcs[obj][1])
			#out.write(writeInfo(iobj, **readInfo(obj, info_df)))
			out.write('\t<td><img src="%s" width="100"></td>\n'%objcs[obj][0])
			out.write(writeInfo(obj, **readInfo(obj, info_df)))
			#out.write('</table>\n')
			print(iobj, obj, objcs[obj])
			for dataset in datasets:
				tfn = f'{thumbnail_folder}/cosmos_{dataset}_idx{obj}.png'
				if os.path.isfile(tfn):
					out.write('<td><img src="%s" width="100"></td>\n'%tfn)
				else:
					out.write('<td></td>')
			#for tf in thumbfiles:
			#	if obj in tf:
			#		out.write('<td><img src="%s" width="100"></td>\n'%tf)
			#for img in objcs[obj][2:]:
			#	out.write('<td><img src="%s" width="1200"></td>\n'%img)
			out.write('</tr>\n')
			
		out.write("""</table>
		</body>
		</html>
		""")
	  


if __name__=='__main__':
	tag = 'cosmos_redextended'
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-f' '--folder', dest='folder', help='folder with images', 
		default=f'../thumbnails/cosmos_targets/cosmos_1sig_interesting/{tag}')
	parser.add_argument('-i' '--info_fn', dest='info_fn', help='name of info csv file', 
		default='../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_cosmos.csv')
	parser.add_argument('-t' '--thumbnails', dest='thumbnails', help='name of thumbnail folder',
        default=f'../thumbnails/cosmos_targets/cosmos_1sig_interesting/{tag}_archive')
	parser.add_argument('-o' '--outputname', dest='outputname', help='name of output html',
		default=f'{tag}_web')
	args = parser.parse_args()
	main(args.folder, args.info_fn, args.thumbnails, args.outputname)
