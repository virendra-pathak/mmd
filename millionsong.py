import tarfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

import time
import random


path_to_million_song_dataset = "millionsongsubset_full.tar.gz"

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

    print(feature_data_matrix)
    
    # Is this correct?
    feature_data_matrix = normalize(feature_data_matrix, norm='l2', axis=0)

    return feature_data_matrix


#
#   Generates a "random" matrix
#
#   Not sure if this is what we looking for.
#
def generate_random_v(rows, cols):
    seed = time.time()

    v = np.empty((rows, cols))

    random.seed(seed) 
    for i in range(rows):
        for j in range(cols):
            v[i][j] = random.choice([-1, 1])

    #print(v)
    return v


def find_duplicates(feature_data_matrix, r, b, sigma):
    dimensions = feature_data_matrix.shape
    time1 = time.time()
    v = generate_random_v(dimensions[1], dimensions[1])
    time2 = time.time()
    print("Time taken to generate random V: ", time2-time1)
  
    #   Uncomment to show v
    #print("Showing the v matrix: ")
    #print(v)
    
    #   Uncomment to show relations
    #   print("Showing relation_matrix: ")
    relation_matrix = np.dot(feature_data_matrix, v)
    print(relation_matrix)
    print("relation_matrix:s shape: ", relation_matrix.shape)

    relations = np.sign(relation_matrix)
    print("Realtions output : ", relations)

    return 0

#find_duplicates(np.empty((1000000, 31)), 20, 3, 5)

t = tarfile.open(path_to_million_song_dataset, "r:gz")
members = t.getmembers()

#for m in members:
#    print(m.name)

'''t.extract(members[7])
df = pd.read_csv(members[7].name, sep='<SEP>', engine='python', header=None)
print(df)'''

#np.unique(df[0], return_counts=True)
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
time2 = time.time()
print("Real time elapsed for extract fields: ", time2-time1)
time1 = time.time()
find_duplicates(feature_data_matrix, 20, 3, 5)
time2 = time.time()

print("Time taken to find duplicates: ", time2-time1)

t.close()
#print("Type summary: ", type(summary['analysis/songs']))
#print("Some field :", summary['analysis/songs'].iloc[0])

#print("Type of summary field: ", type(summary['analysis/songs'].iloc[0]))

#print("Duration field value: ", summary['analysis/songs'].iloc[0]['duration'])

#print(summary['/analysis/songs'])
