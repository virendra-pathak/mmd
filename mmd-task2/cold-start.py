import numpy as np
import time
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix

# user_song_matrix is in its sparse form (csr form)

def cold_start(user_song_matrix, min_threshold):
	matrix_size = np.shape(user_song_matrix)
	#print(matrix_size)
	#print (user_song_matrix.toarray())

	#We need to remove songs & users with less or equal to 5 
	songs_per_user = np.ravel(np.sum(user_song_matrix, axis=1)) #sum of row
	#print("songs per user\n", songs_per_user)

	users_to_delete = np.where(songs_per_user < min_threshold)[0]
	#print("users to delete", users_to_delete)
	
	##set the users i.e rows to 0
	for i in users_to_delete:
		user_song_matrix.data[user_song_matrix.indptr[i]:user_song_matrix.indptr[i+1]]=0
	user_song_matrix.eliminate_zeros()
	mask = np.concatenate(([True], user_song_matrix.indptr[1:] != user_song_matrix.indptr[:-1]))
	user_song_matrix = csr_matrix((user_song_matrix.data, user_song_matrix.indices, user_song_matrix.indptr[mask]))

	##set the songs i.e columns to 0
	user_song_matrix = csc_matrix(user_song_matrix)
	users_per_song = np.ravel(np.sum(user_song_matrix, axis=0)) #sum of column
	songs_to_delete = np.where(users_per_song < min_threshold)[0]

	#print("shape of transpose", user_song_matrix.shape)
	#print("songs to delete ", songs_to_delete)
	for i in songs_to_delete:
		user_song_matrix.data[user_song_matrix.indptr[i]:user_song_matrix.indptr[i+1]]=0
	user_song_matrix.eliminate_zeros()
	mask = np.concatenate(([True], user_song_matrix.indptr[1:] != user_song_matrix.indptr[:-1]))
	user_song_matrix = csc_matrix((user_song_matrix.data, user_song_matrix.indices, user_song_matrix.indptr[mask]))
	return user_song_matrix.tocsr()
	


#Main function
row=12
col=4
min_threshold=1
np.random.seed(int(time.time()))
u_s_matrix = np.random.randint(2, size=(row,col))
sparse_us = csr_matrix(u_s_matrix)
#print("u_s_matrix")
#print(u_s_matrix)
#print("sparse_us", sparse_us.toarray())
#print("sparse_us")
#print(sparse_us)
#print("sparse: sum of columns")
#print(np.sum(sparse_us, axis=0))
#print("sparse: sum of rows")
#print(np.sum(sparse_us, axis=1))
#print("size of sparse matrix")
#print(np.shape(sparse_us))

#cold_start(u_s_matrix, min_threshold)
shape_orig=sparse_us.shape
while True:
	shape=sparse_us.shape
	sparse_us = cold_start(sparse_us, min_threshold)
	shape1=sparse_us.shape
	print("shape", shape, "shape1", shape1)
	if shape[0] == shape1[0] and shape[1] == shape1[1]:
		break
shape_final=sparse_us.shape
print("shape_orig", shape_orig, "shape_final", shape_final)

