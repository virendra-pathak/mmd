import re
import numpy as np
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt


# The sparse matrix function
def create_sparse_matrix(users, songs, play_count, row=True):
    if row:
        return sp.csr_matrix((play_count, (users, songs)), shape=(len(users), len(songs)))
    else:
        return sp.csc_matrix((play_count, (users, songs)), shape=(len(users), len(songs)))

# The parse triplets function
def parse_triplets(file_path, max_rows, whole_dataset, b, use_vectors, use_dikt):
    # Vectors

    users = []
    songs = []
    play_count = []

    mapping_users = {}
    mapping_songs = {}
    
    # Code to extract users, songs and playcount from the train_triplets.txt 

    triplets = open(file_path, 'r')
    
    line_regex = re.compile("^([\w\d]+)\t([\w\d]+)\t([\w]+)$")

    current_row = 0

    highest_playcount = 0

    for triplet in triplets:
        triplet_match = line_regex.match(triplet)

        if (use_vectors):
            if triplet_match.group(1) not in mapping_users:
                mapping_users[triplet_match.group(1)] = len(mapping_users)
    
            if triplet_match.group(2) not in mapping_songs:
                mapping_songs[triplet_match.group(2)] = len(mapping_songs)

            users.append(int(mapping_users[triplet_match.group(1)]))
            songs.append(int(mapping_songs[triplet_match.group(2)]))
            play_count.append(int(triplet_match.group(3)))

        if int(triplet_match.group(3)) > highest_playcount:
            highest_playcount = int(triplet_match.group(3))

        current_row += 1
        if current_row >= max_rows and not whole_dataset:
            break

    triplets.close()

    print("Highest playcount: ", highest_playcount)

    return (users, songs, play_count)


def cold_start(user_song_matrix, min_threshold):
    #We need to remove songs & users with less or equal to min_threshold 
    songs_per_user = np.ravel(np.sum(user_song_matrix, axis=1)) #sum of row
    users_to_delete = np.where(songs_per_user < min_threshold)[0]
    
    ##set the users i.e rows to 0
    for i in users_to_delete:
            user_song_matrix.data[user_song_matrix.indptr[i]:user_song_matrix.indptr[i+1]]=0

    user_song_matrix.eliminate_zeros()
    mask = np.concatenate(([True], user_song_matrix.indptr[1:] != user_song_matrix.indptr[:-1]))
    user_song_matrix = sp.csr_matrix((user_song_matrix.data, user_song_matrix.indices, user_song_matrix.indptr[mask]))

    ##set the songs i.e columns to 0
    user_song_matrix = sp.csc_matrix(user_song_matrix)
    
    users_per_song = np.ravel(np.sum(user_song_matrix, axis=0)) #sum of column
    songs_to_delete = np.where(users_per_song < min_threshold)[0]

    for i in songs_to_delete:
            user_song_matrix.data[user_song_matrix.indptr[i]:user_song_matrix.indptr[i+1]]=0
    
    user_song_matrix.eliminate_zeros()
    mask = np.concatenate(([True], user_song_matrix.indptr[1:] != user_song_matrix.indptr[:-1]))
    user_song_matrix = sp.csc_matrix((user_song_matrix.data, user_song_matrix.indices, user_song_matrix.indptr[mask]))	
    
    return user_song_matrix.tocsr()

if len(sys.argv) < 2:
    print("To few arguments...")
    sys.exit(1)

min_threshold = 5

print("Parsing the triplets...")
users, songs, play_count = parse_triplets(sys.argv[1], 300000, False, 10, True, False)

# Here the logic for the binig should be placed

# Easiest way to create a sparse matrix is done by using the vectors
resulting_sparse_matrix = create_sparse_matrix(users, songs, play_count, row=True)
print("Done parsing the triplets -> Sparse matrix")

shape_orig = resulting_sparse_matrix.shape
while True:
    shape_before = resulting_sparse_matrix.shape
    resulting_sparse_matrix = cold_start(resulting_sparse_matrix, min_threshold)
    shape_after = resulting_sparse_matrix.shape

    if shape_before[0] == shape_after[0] and shape_before[1] == shape_after[1]:
        break
shape_final = resulting_sparse_matrix.shape

print("Orig shape: ", shape_orig, " Final shape: ", shape_after)
sys.stdout.flush()
