import os

CMD = 'for au in 01 02 04 06 07 10 12 14 15 17 23 24; do ./main.py -- --AU=$au --fold=0 --GPU=3 --OF Horizontal --DEMO GFT_Demo --batch_size=1 --mode_data=normal --pretrained_model=./fold_0/OF_Horizontal/AU${au}.pth --mode=test; done'

GFT_frames = '/data/GFT150_trimmed_frames'
GFT_OF = 'GFT150_OF'


for video in os.listdir(GFT_frames):
        in_dir = os.path.join(GFT_frames, video)
        OF_dir = os.path.join(GFT_OF, video)
        run_CMD = CMD.replace(GFT_Demo)

