import numpy as np
import scipy.sparse as sp
import os,json,glob
#import pandas as pd

adj_matrix = sp.lil_matrix((20000, 20000))
filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_test/A/A/**/*.json"
dict_trackid_rowno = {}
row_num = 0


for filename in glob.iglob(filepath, recursive=True):
	#print(filename)
	#assumption 1 jason file contain only one line
	with open(filename) as data_file:
		data = json.load(data_file)

		if data['track_id'] not in dict_trackid_rowno:
			dict_trackid_rowno[data['track_id']] = row_num
			row_num += 1
		row = dict_trackid_rowno[data['track_id']]
		#print("row", row)
		for similar in data['similars']:
			if similar[0] not in dict_trackid_rowno:
				dict_trackid_rowno[similar[0]] = row_num
				row_num += 1
			col = dict_trackid_rowno[similar[0]]
			adj_matrix[row,col] = similar[1]
			#print(row, col, similar[1])

#print("max row", row)
#print("trackid --> rowno")
#print(dict_trackid_rowno.items())
#print("adj_matrix")
#print((adj_matrix.todense())[8296,8593])
