import re
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import sys
import matplotlib.pyplot as plt
import random
import time
import math
from scipy import sparse
from sklearn.linear_model import Ridge

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

    
def binning(play_count, bin_size):
    bin_array = [2**i for i in range(bin_size)]
    print("Bin Array : ", bin_array)
    binned_counts = np.digitize(play_count, bin_array, right = False)
    print("Binned Counts : ", binned_counts)
    return binned_counts
        
    
def cold_start(user_song_matrix, min_threshold):
    #We need to remove songs & users with less or equal to min_threshold 
    #print(user_song_matrix)
    #user_song_matrix = user_song_matrix > 0
    user_song_matrix = user_song_matrix.astype(np.int)
    #print(user_song_matrix.toarray())
 
    songs_per_user = np.ravel(np.sum((user_song_matrix > 0), axis=1)) #sum of row
    users_to_delete = np.where(songs_per_user <= min_threshold)[0]
    
    ##set the users i.e rows to 0
    for i in users_to_delete:
            user_song_matrix.data[user_song_matrix.indptr[i]:user_song_matrix.indptr[i+1]]=0

    user_song_matrix.eliminate_zeros()
    mask = np.concatenate(([True], user_song_matrix.indptr[1:] != user_song_matrix.indptr[:-1]))
    user_song_matrix = sp.csr_matrix((user_song_matrix.data, user_song_matrix.indices, user_song_matrix.indptr[mask]))

    ##set the songs i.e columns to 0
    user_song_matrix = sp.csc_matrix(user_song_matrix)
    
    users_per_song = np.ravel(np.sum((user_song_matrix > 0), axis=0)) #sum of column
    songs_to_delete = np.where(users_per_song <= min_threshold)[0]

    for i in songs_to_delete:
            user_song_matrix.data[user_song_matrix.indptr[i]:user_song_matrix.indptr[i+1]]=0

    user_song_matrix.eliminate_zeros()
    mask = np.concatenate(([True], user_song_matrix.indptr[1:] != user_song_matrix.indptr[:-1]))
    user_song_matrix = sp.csc_matrix((user_song_matrix.data, user_song_matrix.indices, user_song_matrix.indptr[mask]))	
    
    return user_song_matrix.tocsr()

def singular_value_decomp(sparse_matrix):
    sparse_matrix = sparse_matrix.asfptype()
    print("Matrix shape : ", sparse_matrix.shape)
    # print("Sparse Matrix : ", sparse_matrix)
    
    U, sigma, Vt = spl.svds(sparse_matrix, k=30)
    print("U shape : ", U.shape, " sigma shape : ", sigma.shape, " V shape : ", Vt.shape)
    
    sigma = np.diag(sigma)
    print("Sigma shape after converting to diagonal matrix : ", sigma.shape)
    Q = U.dot(sigma)
    print("Q shape : ", Q.shape)
    print("Pt shape : ", Vt.shape)
    return Q, Vt

#
#   This functions picks a random set of size: number_of_random_elements
#   
#   Returns: M_s the modified M
#            random_set containing all random pciked elements on the form [(row,col,data)]
#
def pick_random_test_set(M, number_of_random_elements):
    random.seed(time.time())
    
    M_s = sp.coo_matrix(M)
    interval = len(M_s.data)

    random_picked_values = []
    already_picked_values = []
    
    for i in range(number_of_random_elements):
        random_index = random.randint(0, interval-1)

        while random_index in already_picked_values:
            random_index = random.randint(0, interval-1)
        already_picked_values.append(random_index)
        
        random_picked_values.append((M_s.row[random_index], 
                                    M_s.col[random_index], 
                                    M_s.data[random_index]))
        M_s.data[random_index] = 0

    M_s.eliminate_zeros()
    M_s = sp.csr_matrix(M_s)
    return M_s, random_picked_values

#M is in csr format
def calc_loss(M, Q, Pt):
	#TODO we have to optimize it
    lamda1 = 0.3;lamda2 = 0.3

    loss = ((M - Q.dot(Pt)).power(2)).sum()
    loss += lamda1*(np.sum(np.square(spl.norm(Pt, axis=0)))) + lamda2*(np.sum(np.square(spl.norm(Q, axis=1))))
    loss = loss/171072
    return loss

#M is in csr format
def gradient_loss_P(M, Q, Pt):
	lamda1 = 0.3

	#print("shape Q", Q.shape, "Pt", Pt.shape)
	gradient = ((-2 * (M - Q.dot(Pt))).transpose()).dot(Q)
	gradient = gradient.transpose()
	print("shape gradient", gradient.shape)
	gradient += (2*lamda1*Pt)
	#gradient = gradient.transpose()
    #print("shape gradient after 2lamda1", gradient.shape)
	#print(gradient)
	return gradient

# M is in csr format
def gradient_loss_Q(M, Q, Pt):
	lamda2 = 0.3

	#print("shape Q", Q.shape, "Pt", Pt.shape)
	gradient = ((-2 * (M - Q.dot(Pt)))).dot(Pt.transpose())
	gradient += (2*lamda2*Q)
	#print("shape gradient after 2lamda1", gradient.shape)
	return gradient

def gradient_descent(M, Pt, Q):
    lr_rate = 0.0001
    print("M size", M.shape)
    print("initial Q size", Q.shape)
    print("initial Pt size", Pt.shape)
    print("num of element in M", len(M.data))
    print("P shape index", Pt.shape[0], Pt.shape[1])

    Q = sp.csr_matrix(Q)
    Pt = sp.csr_matrix(Pt)

    loss1 = calc_loss(M, Q, Pt)
    print("loss1", loss1)

    #print(M-Q.dot(Pt))
    for iter in range(6):
       temp0 = Pt - (lr_rate * gradient_loss_P(M, Q, Pt))
       temp1 = Q - (lr_rate * gradient_loss_Q(M, Q, Pt))
       Pt = temp0
       Q = temp1
       #print("sum of Pt & Q")
       #print(Pt.sum(), Q.sum())
       #print("final Q size", Q.shape)
       #print("final Pt size", Pt.shape)
       loss2 = calc_loss(M, Q, Pt)
       print("loss2", loss2)
    return None
#----------------------------------------------------
#M is in csr format
def calc_loss_stoc(M, Q, Pt):
	#TODO we have to optimize it
    lamda1 = 0.3;lamda2 = 0.3

    
    loss = (((M - Q.dot(Pt)).power(2)).sum())
    loss += lamda1*(np.sum(np.square(spl.norm(Pt, axis=0)))) + lamda2*(np.sum(np.square(spl.norm(Q, axis=1))))
    loss = loss/171072
    
    return loss

#M is in csr format
def gradient_loss_Px_stoc(r, qx, pi):
	lamda1 = 0.5

	gradient = (-2 * (r - (qx.dot(pi)[0,0]))) * qx
	gradient += 2*lamda1*pi.transpose()
     
	gradient = gradient.transpose()
	#print("shape gradientPx", gradient.shape)
	#gradient += (2*lamda1*Pt)
	#print("shape gradient after 2lamda1", gradient.shape)
	#print(gradient)
	return gradient

# M is in csr format
def gradient_loss_Qi_stoc(r, qx, pi):
	lamda2 = 0.5

	gradient = (-2 * (r - (qx.dot(pi)[0,0]))) * pi
	gradient += 2*lamda2*qx.transpose()
	gradient = gradient.transpose()
	#print("shape gradientQi", gradient.shape)
	#gradient += (2*lamda2*Q)
	#print("shape gradient after 2lamda1", gradient.shape)
	return gradient

def gradient_descent_stoc(M, Pt, Q):
    lr_rate = 0.0001
    print("M size", M.shape)
    print("initial Q size", Q.shape)
    print("initial Pt size", Pt.shape)
    print("num of element in M", len(M.data))
    print("P shape index", Pt.shape[0], Pt.shape[1])

    Q = sp.csr_matrix(Q)
    Pt = sp.csr_matrix(Pt)

    
    loss1 = calc_loss_stoc(M, Q, Pt)
    print("loss1", loss1)
    

    #print(M-Q.dot(Pt))
    #M = M.tocoo();
    
    for iter in range(10):
        
        #r=M.data[d]; x=M.row[d]; i=M.col[d];
        random_x = random.randint(0, M.shape[0])
        print("random_x : ", random_x)
        
        random_i = random.randint(0, M.shape[1])
        print("random_i : ", random_i)
        
        random_r = M[random_x, random_i]
        print("random_r : ", random_r)
    

        temp0 = Pt[:,random_i] - (lr_rate * gradient_loss_Px_stoc(random_r, Q[random_x,:], Pt[:,random_i]))
        #print(Pt[:, random_i].dtype)
        temp1 = Q[random_x,:] - (lr_rate * gradient_loss_Qi_stoc(random_r, Q[random_x,:], Pt[:,random_i]))
      
        Pt[:,random_i] = temp0
        Q[random_x,:] = temp1

        loss2 = calc_loss_stoc(M, Q, Pt)
        print("loss2", loss2)
    M=M.tocsr()
    return None
#----------------------------------------------------

def finding_method(M, Q, P):
    P = sp.csr_matrix(P_stoc)
    Q = sp.csr_matrix(Q)
    M = sp.csr_matrix(M)

    print("P sh:", P.shape)
    print("Q sh:", Q.shape)
        
    m_shape = M.shape
    
    print(type(M[0,:]))

    print("Q.T shape: ", Q.T.shape)
    print("Q shape: ", Q.shape)
    #print(np.einsum('ji,j->ji', Q, Q))

    qTq = np.sum(np.dot(Q.T,Q).diagonal())
    print("qTq: ", qTq)
    #print(qTq.shape)

    pTp = np.sum(np.dot(P.T, P).diagonal())
    print("pTp: ", pTp)
    
    for x in range(m_shape[1]):
        m = M[:,x]
        #print("Q.T shape: ", Q.T.shape)
        #print("m.T shape: ",  m.T.shape)

        #print("Q shape: ", Q.shape)
        #print("m shape: ",  m.shape)
        p_x = (1/pTp) * np.dot(Q.T, m)
        
        #print("P shape first : ", P.shape)
        P = sp.vstack((sp.vstack((P[:x], p_x.T)), P[x+1:]))
        #print("P shape after : ", P.shape)
        #print(P.data[P.indptr[x]:P.indptr[x+1]])

        #print("p_x: ", p_x.shape)

    print("Done with P")

    for i in range(m_shape[0]):
        m = M[i,:]
        q_i = (1/qTq) * np.dot(P.T, m.T)

        Q = sp.vstack((sp.vstack((Q[:i], q_i.T)), Q[i+1:]))

    print(np.dot(Q,P.T).shape)

    np.sum(np.norm('l2', M - np.dot(Q,P.T)), axis=1)
    #print(m.shape)
    #print(l.shape)
    print("Done")
    
def rmse(Mdiff, test_set):
    
    y_sum = 0
    for test in test_set:
        y_sum += (test[2] - Mdiff[test[0]][test[1]])**2

    rmse = np.sqrt(y_sum/len(test_set))

    print("RMSE: ", rmse)

if len(sys.argv) < 2:
    print("To few arguments...")
    sys.exit(1)

min_threshold = 5
bin_size = 10 ## This is configurable by user
size_of_test_set = 200

print("Parsing the triplets...")
users, songs, play_count = parse_triplets(sys.argv[1], 300000, False, 10, True, False)

# Here the logic for the binnig should be placed

binned_play_counts = binning(play_count, bin_size)

# Either pass this binned_play_counts to the following functions that require play_count
# as a parameter, or change the return variable "binned_play_counts" to "play_count"
# to override its value

# Easiest way to create a sparse matrix is done by using the vectors
resulting_sparse_matrix = create_sparse_matrix(users, songs, binned_play_counts, row=True)
print("Done parsing the triplets -> Sparse matrix")

shape_orig = resulting_sparse_matrix.shape
while True:
    shape_before = resulting_sparse_matrix.shape
    resulting_sparse_matrix = cold_start(resulting_sparse_matrix, min_threshold)
    shape_after = resulting_sparse_matrix.shape
    print("Intermediate shape: ", shape_after)
    if shape_before[0] == shape_after[0] and shape_before[1] == shape_after[1]:
        break
shape_final = resulting_sparse_matrix.shape

print("Picking random test set: ")
resulting_sparse_matrix, test_set = pick_random_test_set(resulting_sparse_matrix, size_of_test_set)

#print("Got random test set: ", test_set)

# Initial P & Q values obtained using SVD
Q, Pt = singular_value_decomp(resulting_sparse_matrix)

# Perform AO using P,Q
Pt_orig = Pt
Q_orig = Q
print("Running batch gd");
gradient_descent(resulting_sparse_matrix, Pt, Q)
print("Running stoc gd");
gradient_descent_stoc(resulting_sparse_matrix, Pt_orig, Q_orig)



#find_method(resulting_sparse_matrix, Q, Pt.T)
print("Orig shape: ", shape_orig, " Final shape: ", shape_after)
sys.stdout.flush()
