import numpy as np
import scipy.sparse as sp
import os,json,glob
#import pandas as pd

t=0 # make it as a user defined variable
adj_matrix = sp.lil_matrix((764719,747806))
M = sp.lil_matrix((764719,747806))
song_tag_matrix = sp.lil_matrix((764719,747806))
#filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_subset/A/A/A/**/*.json"
#filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_subset/A/A/A/TRAAAAW128F429D538.json"
filepath="C:\\Users\\Sarthak\\MMD\\group_17\\song-rank-mmd4\\lastfm_test\\A\\A\\*.json"
dict_trackid_rowno = {}
row_num = 0

def create_M(adj_matrix):
    print("Creating M Matrix...")

    outgoing_edges = {}
    for song in range(adj_matrix.shape[0]):
                
       song_row = adj_matrix.getrow(song)
       outgoing_edges[song] = song_row.count_nonzero()


    non_zero_row_index = M.nonzero()[0]
    non_zero_col_index = M.nonzero()[1]
    for index in range(non_zero_row_index):
        if(adj_matrix[non_zero_row_index[index],non_zero_col_index[index]]):
            M[non_zero_col_index[index],non_zero_row_index[index]] = 1/outgoing_edges[non_zero_row_index[index]]
    
    print("M shape : ", M.shape)
#write all below code in a function to make it clean


def create_R(M, song_to_tag_map):
    beta = 0.2
    # TODO create R vector with initial values as 1/number of songs
    R = np.array(adj_matrix.shape[0])
    # TODO create the vector "song_in_tag_specified" if a song i belongs to given tag, then value is 1/mod(s)
    summed_row = song_tag_matrix.sum(axis = 1)
    num_songs_in_tags = summed_row.count_nonzero()
    value = (1-beta)/num_songs_in_tags
    for song in range(M.shape[0]):
        if(song_tag_matrix[song].sum > 0):
            song_in_tag_specified[song] = value

    for iterator in range(20):
        
        # TODO import math class, declare a threshold, declare R_old
        R = ((beta * M) * R) + song_in_tag_specified
        if(math.abs(R_old - R) < threshold):
            # stop the iteration
            break
        R_old = R

def song_to_tag_map(dict_tag):
    
    for key, value in  dict_tag.items():
        
            for genres in value[1]:
                
                # TODO create user_specified_genre values
                if(genres[0] in user_specified_genre):
                    
                    song_tag_matrix[value[0], user_specified_genre.index(genres[0])] = 1
    
print("Creating Similarity Matrix...")
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
                 
			if(similar[1] >= t):
                     
			 adj_matrix[row,col] = 1 
			#print(row, col, similar[1])

create_M(adj_matrix)

song_to_tag_map(dict_trackid_rowno)

create_R(M, song_to_tag_map)

print("max row", row, "max col", col)
#print("trackid --> rowno")
#print(dict_trackid_rowno.items())
print("adj_matrix shape", adj_matrix.shape)
#print((adj_matrix.todense())[8296,8593])
