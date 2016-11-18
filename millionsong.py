import tarfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import time
import random
path_to_million_song_dataset = "G:\\MS\\TUM\\courses\\Mining Massive Datasets\\millionsongsubset_full.tar.gz"
hash_vector = np.empty((20))
for i in range(20):
    hash_vector[i] = 2**i

print("hash vectore shape : ", hash_vector.shape)
print ("hashx vector : ", hash_vector)

hash_buckets = np.empty((2**20))
for i in range(2**20):
    hash_buckets[i] = 0

print("hash bucket shape : ", hash_buckets.shape)

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
    
def banding(signature_matrix, num_bands, rows_in_band, num_RV):
    band_start_index = 0
    band_end_index = band_start_index + rows_in_band 
    
    while(band_end_index <= num_RV):
        
        print("starting index: ",  band_start_index, " and band end index: ", band_end_index)
        band = signature_matrix[band_start_index:band_end_index]
        hashing(band)
        band_start_index = band_end_index + 1
        band_end_index = band_start_index + rows_in_band
        

def hashing(band):
    candidate_pairs = 0
    for j in range(band.shape[1]):
        local_song_signature = band[:, j]
        #   print("local song signature shape before transpose: ", local_song_signature.shape)
        hashValue = getHashValue(local_song_signature)
        #   print("hash value is: ", hashValue)
        hash_buckets[hashValue] = hash_buckets[hashValue] + 1
        
    for i in range(2**20):
        if(hash_buckets[i] > 1):
            candidate_pairs = candidate_pairs + 1
    
    print("candidate pairs : " , candidate_pairs)


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
    #   Uncomment to show v
    #print("Showing the v matrix: ")
    #print(v)
    
    #   Uncomment to show relations
    #   print("Showing relation_matrix: ")
    v = v.transpose()
    print ("shape of RV matrix : ", v.shape)
    RV_dimensions = v.shape;
    signature_matrix = np.empty((dimensions[0], num_of_RV))
    
    for i in range(signature_matrix.shape[0]):
        for j in range(signature_matrix.shape[1]):
            signature_matrix[i][j] = np.dot(feature_data_matrix[i], v[:, j])
                
    #   signature_matrix = np.dot(feature_data_matrix, v.transpose())
    print(signature_matrix)
    print("signature_matrix:s shape: ", signature_matrix.shape)

    signatures = np.sign(signature_matrix).transpose()
    print("Signatures output : ", signatures)
    
    banding(signatures, b, r, num_of_RV)

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
