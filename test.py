import os 
import numpy as np


for element in os.listdir('data/clips/old'):

  if not element.startswith('.'):
    os.mkdir('data/clips/new/'+element)

    for clip in os.listdir('data/clips/old/'+element):
      if not clip.startswith('.'):

        path = 'data/clips/old/'+element+'/'+clip
        flow_path = 'data/flow_clips/'+element+'/'+'flow_'+clip
        save_path = 'data/clips/new/'+element+'/'+clip
        print(path)
        temp = np.load(path)
        frames = np.shape(temp)[0]
        os.system('python custom.py --path '+path+' --flow_path '+flow_path+' --save_path '+save_path+' --frames '+str(frames))

