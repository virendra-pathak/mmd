import tarfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
import time
import random

import itertools as it

path_to_million_song_dataset = "millionsongsubset_full.tar.gz"

hash_vector = np.array([2**i for i in range(64)])

duplicate_songs = dict()

#
#   Returns a matrix which columns corresponds to a specific feature:
#   Each row corresponds to a song
#   Each field a for the moment floats
#
#   features:A list containing features
#   dataframe: frame containing all feature data
#   n: number of songs
#
#   returns: Numpy.Matrix(col=feature,row=songs)
#
def extract_fields(features, dataframe, n):
    number_of_features = len(features)
    feature_data_matrix = np.empty((n, number_of_features))
    for i in range(n):
        col_index = 0
        for feature in features:
            feature_data_matrix[i][col_index] = dataframe.iloc[i][feature]
            col_index += 1
    
    # Is this correct?
    #feature_data_matrix = normalize(feature_data_matrix, norm='l2', axis=0)
    feature_data_matrix = scale(feature_data_matrix)
    return feature_data_matrix
#
#   Generates a "random" matrix
#
def generate_random_v(rows, cols):
    v = np.random.choice([-1,1], (rows, cols))
    return v
    
def banding(signature_matrix, num_bands, rows_in_band, num_RV):
    band_start_index = 0
    band_end_index = rows_in_band - 1 
   
    while(band_end_index <= num_RV):
        
        print("starting index: ",  band_start_index, " and band end index: ", band_end_index)
        band = signature_matrix[band_start_index:band_end_index+1]
        hashing(band)
        band_start_index = band_end_index + 1
        band_end_index += rows_in_band

    duplicates = 0
    for song,similiarity_list in duplicate_songs.items():
        if len(similiarity_list) > 0:
            duplicates += (len(similiarity_list))
            #print("Duplicate pairs: ", song, " and ", similiarity_list)

    print("We have found ", duplicates, " duplicate pairs")

def hashing(band):
    candidate_pairs = 0

    hash_buckets = dict()
    
    for j in range(band.shape[1]):
        local_song_signature = band[:, j]
        hash_value = getHashValue(local_song_signature)
        if hash_value not in hash_buckets: 
            hash_buckets[hash_value] = [j]
        else:
            hash_buckets[hash_value].append(j) 
    
    for bucket in hash_buckets.items():
        if len(bucket) > 1:
            candidate_pairs += len(bucket)
        
    find_exact_cosine_distance(hash_buckets)

    print("Candidate pairs on one band: " , candidate_pairs)

def find_exact_cosine_distance(hash_buckets):
  
    for bucket in hash_buckets.items():
        for (i,j) in it.combinations(bucket[1], 2):

            if i not in duplicate_songs:
                duplicate_songs[i] = set([])
             
            if j not in duplicate_songs[i]:
                cosine_value = cosine_similarity(feature_data_matrix[i], feature_data_matrix[j]) 
                if cosine_value < sigma:
                    duplicate_songs[i].update([j])

def cosine_similarity(song1, song2):
    mag1 = np.linalg.norm(song1)
    mag2 = np.linalg.norm(song2)

    cos_angle = np.dot(song1, song2)/(mag1*mag2)

    return 1 - cos_angle

def getHashValue(local_song_signature):
    hashValue = 0
    
    #   print("local song signature shape after transpose: ", local_song_signature.shape)
    hashValue = np.dot(local_song_signature, hash_vector)
    return hashValue
        
def find_duplicates(feature_data_matrix, r, b, sigma):
    dimensions = feature_data_matrix.shape

    time1 = time.time()
    num_of_RV = r*b
    print("number of RV : ", num_of_RV)
    v = generate_random_v(num_of_RV, dimensions[1])
    time2 = time.time()
    print("Time taken to generate random V: ", time2-time1)
 
    print("Rank of matrix: ", np.linalg.matrix_rank(v))
    
    v = v.transpose()
    
    signature_matrix = np.dot(feature_data_matrix, v)
    
    for i in range(signature_matrix.shape[0]):
        for j in range(signature_matrix.shape[1]):
            if signature_matrix[i][j] > 0:
                signature_matrix[i][j] = 1
            else:
                signature_matrix[i][j] = 0
    
    time1 = time.time()
    banding(signature_matrix.transpose(), b, r, num_of_RV)
    time2 = time.time()
    print("Time taken to find duplicates: ", time2 - time1)
    
    return 0

print("Extracting the tarfile")
t = tarfile.open(path_to_million_song_dataset, "r:gz")
members = t.getmembers()

print("Extracting the summary")
t.extract(members[5].name)
summary = pd.HDFStore(members[5].name)

print("Extracting the features...")
time1 = time.time()
feature_data_matrix = extract_fields(['duration', 
                'end_of_fade_in', 
                'key', 
                'loudness', 
                'mode', 'start_of_fade_out',
                'tempo',
                'time_signature'], summary['analysis/songs'], 9999)
sigma = 0.0006092

time2 = time.time()
print("Real time elapsed for extract fields: ", time2-time1)

time1 = time.time()
find_duplicates(feature_data_matrix, 64, 3, sigma)
time2 = time.time()

print("Time taken to find duplicates with generation of random vectors and preprocessing of the data: ", time2-time1)
print("Exits the program")
t.close()
