import numpy as np
import scipy.sparse as sp
import os,json,glob
import time
#import pandas as pd

#filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_subset/A/A/A/**/*.json"
#filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_subset/A/A/A/TRAAAAW128F429D538.json"
#filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_subset/A/A/A/**/*.json"
#filepath="/media/virendra/data/study/1sem/mmd/rank/lastfm_subset/**/*.json"
#filepath="C:\\Users\\Sarthak\\MMD\\group_17\\song-rank-mmd4\\lastfm_test\\**\\*.json"
#filepath="C:\\Users\\Sarthak\\MMD\\group_17\\song-rank-mmd4\\lastfm_test\\A\\A\\*.json"
filepath="C:\\Users\\Sarthak\\MMD\\group_17\\song-rank-mmd4\\lastfm_test\\**\\*.json"

t=0 # make it as a user defined variable

#list of tags
user_specified_genre = ["Hip-Hop", "classic"]

max_row = 764719
max_col = 764719
#max_col = 747806
adj_matrix = sp.lil_matrix((max_row, max_col))
M = sp.lil_matrix((max_row, max_col))
song_tag_matrix = sp.lil_matrix((max_row, max_col))
dict_trackid_rowno = {}
trackID_title_dict = {}
track_ID = []


def create_M(adj_matrix):
    start_time = time.time()
    print("Creating M Matrix...")

    outgoing_edges = {}
    for song in range(adj_matrix.shape[0]):
                
       song_row = adj_matrix.getrow(song)
       outgoing_edges[song] = song_row.count_nonzero()

    # and why we are using M.nonzero?
    #non_zero_row_index = M.nonzero()[0]
    #non_zero_col_index = M.nonzero()[1]
    non_zero_row_index = adj_matrix.nonzero()[0]
    non_zero_col_index = adj_matrix.nonzero()[1]
    #print(non_zero_row_index.shape)
    for index in range(len(non_zero_row_index)):
        if(adj_matrix[non_zero_row_index[index],non_zero_col_index[index]]):
            M[non_zero_col_index[index],non_zero_row_index[index]] = 1/outgoing_edges[non_zero_row_index[index]]
    
    print("M shape : ", M.shape)
    print("Time taken for creating M matrix: ", time.time() - start_time)

#write all below code in a function to make it clean
def create_R(M, song_to_tag_map):
    print("Creating R matrices...")
    beta = 0.2
    # created R as all value 1/num_songs at all its indexes
    R = np.full(max_row, 1/max_row)

    # created song_in_tag_specified as all 0 
    song_in_tag_specified = np.zeros(max_row)
    summed_row = song_tag_matrix.sum(axis = 1)
    print("summed row : ", summed_row)
    #num_songs_in_tags = summed_row.count_nonzero()
    print("shape of summed_row", summed_row.shape)
    num_songs_in_tags = np.count_nonzero(summed_row)
    print("num_songs in tags : ", num_songs_in_tags)
    value = (1-beta)/num_songs_in_tags
    print("value: ", value)
    is_song_in_tag = song_tag_matrix.sum(axis=1)
    print("is song in tag : ", is_song_in_tag)
    for song in range(M.shape[0]):
        if is_song_in_tag[song] > 0:
            song_in_tag_specified[song] = value

    R_old = np.zeros(max_row)
    for iterator in range(20):
        #print("R.shape", R.shape, "M.shape", M.shape, "song_in_tag_specified.shape", song_in_tag_specified.shape) 
        R = ((beta * M).dot(R)) + song_in_tag_specified
        print("R : ", R, " iterator : ", iterator)
        # using numpy allclose for convergence
        #if(math.abs(R_old - R) < threshold):
        if(np.allclose(R,R_old)):
            break
        R_old = R
    R_index = np.argsort(R)
    print("Sorted indexes : ", R_index)
    #print("final R", R[R_index[0]],R[R_index[1]],R[R_index[2]],R[R_index[3]],R[R_index[4]])
    print("Sorted R : ", R[R_index])
    R = R[R_index]
    print("top 5 elements..")
    for index in range(5):
        print("Sorted index: ", R_index[R_index.shape[0] - (index+1)])
        print(R[R.shape[0] - (index+1)], " TRACK ID : ",  track_ID[R_index[R_index.shape[0] - (index+1)]], " Track name : ", trackID_title_dict[track_ID[R_index[R_index.shape[0] - (index+1)]]])
       
def song_to_tag_map(dict_tag):
    start_time = time.time()
    print("Creating Song to Tag map...")
    for key, value in  dict_tag.items():
        if len(value) >= 2:    
           for genres in value[1]:
              # created at the beginning of file as list
              if(genres[0] in user_specified_genre):
                 song_tag_matrix[value[0], user_specified_genre.index(genres[0])] = 1
    
    print("Time taken for creating song to tag map: ", time.time() - start_time)
def create_similarty_matrix():
    start_time = time.time()
    print("Creating Similarity Matrix...")
    row_num = 0
    for filename in glob.iglob(filepath, recursive=True):
        #print(filename)
        #assumption 1 jason file contain only one line
	
        with open(filename) as data_file:
            data = json.load(data_file)
            trackID_title_dict[data['track_id']] = data['title']
            if data['track_id'] not in dict_trackid_rowno:
        	 
                dict_trackid_rowno[data['track_id']] =  [row_num] 
                track_ID.insert(row_num, data['track_id'])  
                row_num += 1
                dict_trackid_rowno[data['track_id']] += [data['tags']]
                row = dict_trackid_rowno[data['track_id']][0]
                for similar in data['similars']:
                    if similar[0] not in dict_trackid_rowno:
                        dict_trackid_rowno[similar[0]] = [row_num]
                        track_ID.insert(row_num, similar[0])
                        row_num += 1
                        col = dict_trackid_rowno[similar[0]][0]

    			# what we have to store is adj_matirx? 1 or similar value?
			# if its only 1/0 => just use a additional if (similar[1] > t)
                    if(similar[1] >= t):
                     
                        adj_matrix[row,col] = 1 
    print("Time taken for creating similarity matrix: ", time.time() - start_time)
			#print(row, col, similar[1])


#print("track ID array: ", track_ID)

start_time = time.time()
create_similarty_matrix()

create_M(adj_matrix)

song_to_tag_map(dict_trackid_rowno)

create_R(M, song_to_tag_map)
end_time = time.time()

print("time taken : ", end_time - start_time)
#print("max row", row, "max col", col)
#print("trackid --> rowno")
#print(dict_trackid_rowno.items())
print("adj_matrix shape", adj_matrix.shape)
#print((adj_matrix.todense())[8296,8593])
