import numpy as np

def cold_start(user_song_matrix, min_threshold):
	matrix_size = np.shape(user_song_matrix)
	print(matrix_size)
	print (user_song_matrix)

	#We need to remove songs & users with less or equal to 5 
	users_per_song = np.sum(user_song_matrix, axis=0) #sum of column
	songs_per_user = np.sum(user_song_matrix, axis=1) #sum of row
	print ("users per song", users_per_song)
	print("songs per user", songs_per_user)

	songs_to_delete = np.where(users_per_song < min_threshold)[0]
	users_to_delete = np.where(songs_per_user < min_threshold)[0]
	print("songs to delete", songs_to_delete)
	print("users to delete", users_to_delete)

	#reduce the dimension of matrix
	user_song_matrix=np.delete(user_song_matrix, songs_to_delete, 1) #column
	print("shape", np.shape(user_song_matrix))
	print(user_song_matrix)
	user_song_matrix=np.delete(user_song_matrix, users_to_delete, 0) #row
	print("shape", np.shape(user_song_matrix))
	print(user_song_matrix)

#Main function
row=10
col=10
min_threshold=5
u_s_matrix = np.random.randint(2, size=(row,col))
cold_start(u_s_matrix, min_threshold)
