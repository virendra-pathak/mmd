import numpy as np
import scipy.sparse as sp
import os,json,glob
#import pandas as pd

t=0 # make it as a user defined variable
adj_matrix = sp.lil_matrix((764719,747806))
#filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_subset/A/A/A/**/*.json"
#filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_subset/A/A/A/TRAAAAW128F429D538.json"
filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_test/**/*.json"
dict_trackid_rowno = {}
row_num = 0

#write all below code in a function to make it clean
for filename in glob.iglob(filepath, recursive=True):
	#print(filename)
	#assumption 1 jason file contain only one line
	with open(filename) as data_file:
		data = json.load(data_file)

		if data['track_id'] not in dict_trackid_rowno:
			dict_trackid_rowno[data['track_id']] =  [row_num] 
			row_num += 1
		dict_trackid_rowno[data['track_id']] += [data['tags']]
		row = dict_trackid_rowno[data['track_id']][0]
		for similar in data['similars']:
			if similar[0] not in dict_trackid_rowno:
				dict_trackid_rowno[similar[0]] = [row_num]
				row_num += 1
			col = dict_trackid_rowno[similar[0]][0]
			# what we have to store is adj_matirx? 1 or similar value?
			# if its only 1/0 => just use a additional if (similar[1] > t)
			adj_matrix[row,col] = similar[1] 
			#print(row, col, similar[1])

print("max row", row, "max col", col)
#print("trackid --> rowno")
#print(dict_trackid_rowno.items())
print("adj_matrix shape", adj_matrix.shape)
#print((adj_matrix.todense())[8296,8593])
