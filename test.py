import warnings
warnings.filterwarnings('ignore')
from functions.environment.camera import thread_episode
import os

if __name__=='__main__':
    songid = '0F7FA14euOIX8KcbEturGH'
    duration = 157066
    steps = 10
    print('Reproducing "Old Town Road - Lil Nas X ft. Billy Ray Cyrus".')
    vector, history = thread_episode(os.path.join(os.curdir,'functions','environment','.songs',f'{songid}.mp3'),duration,steps)
    print('final',vector)