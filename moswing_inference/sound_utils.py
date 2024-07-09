import librosa
import numpy.typing as npt
import numpy as np

DEFAULT_SR = 8000

def load_sound(sound_path, sr:int) -> npt.NDArray[np.floating]:
    # type of sound path is the same for first arg of librosa load
    sound, _ = librosa.load(sound_path, sr=sr, mono=True)
    return sound

def split_in_seqs(data, subdivs):
    # https://github.com/sharathadavanne/sed-crnn/blob/master/utils.py#L29
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs)] # cut the last exceeded to make it divisible by subdivs
        data = data.reshape((data.shape[0] // subdivs, subdivs, 1)) # make it 3d
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1])) # like above
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1], data.shape[2])) # 4d
    return data # output array 3d (None, subdivs, 1 or data.shape[1]) or array 4d (None, subdivs, data.shape[1], data.shape[2])

def preprocess_file(sound):
    # With current implementation, if the duration does not divisible with 10 (in seconds) 
    # the remainder will not used. for example, the audio of 37 seconds, only first 30 seconds will get through the model 
    data = split_in_seqs(sound, DEFAULT_SR)
    data = split_in_seqs(data, 10)
    data = np.array([librosa.util.normalize(d) for d in data])
    return data
    



    


