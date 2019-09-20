import glob
import numpy as np
from midi_manipulation import midiToNoteStateMatrix, noteStateMatrixToMidi
import os

num_timesteps = 192  # This is the number of timesteps that we will create at a time
lowest_note = 24  # the index of the lowest note on the piano roll
highest_note = 102  # the index of the highest note on the piano roll
note_range = highest_note - lowest_note  # the note range

# files = glob.glob('{}/*.mid*'.format('midis'))
files = glob.glob('{}/*.mid*'.format('midis'))
songs = []

pos = 0
neg = 0
print(len(files))
counter = 0
for i, file in enumerate(files):
	try:
	    if i%10 == 0:
	        print(i)
	    song = np.asarray(midiToNoteStateMatrix(file))
	    if len(song) < 2 :
	        pos += 1
	        print(pos)
	        os.remove(file)
	    else:
		    neg += 1
		    end = (np.floor(song.shape[0] / num_timesteps)
		           * num_timesteps).astype(np.int16)
		    song = song[:end]
		    
		    song2 = np.hstack([ song[:,1:], np.zeros([song.shape[0],1])])
		    song3 = np.hstack([ song2[:,1:], np.zeros([song.shape[0],1])])

		    noteStateMatrixToMidi(song, name=("mynewdata/sample_" + str(i)))

		    song = np.reshape(
		        song, [int(song.shape[0] / num_timesteps), song.shape[1] * num_timesteps])
		    song2 = np.reshape(
		        song2, [int(song2.shape[0] / num_timesteps), song2.shape[1] * num_timesteps])
		    song3 = np.reshape(
		        song3, [int(song3.shape[0] / num_timesteps), song3.shape[1] * num_timesteps])

		    songs.append(song)
		    songs.append(song2)
		    songs.append(song3)

	except:
		print("exception")


songs2 = []
songs = np.vstack(songs)

for song in songs:
    if not song.reshape(-1, 156)[:10].sum() < 1:
        songs2.append(song)

np.save('songs', songs)
# songs = np.load("songs.npy")
