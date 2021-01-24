import os#, pdb

CMD = 'for au in 01 02 04 06 07 10 12 14 15 17 23 24; do ./main.py -- --AU=$au --fold=0 --GPU=3 --OF Horizontal --DEMO GFT_Demo --batch_size=1 --mode_data=normal --pretrained_model=./fold_0/OF_Horizontal/AU${au}.pth --mode=test; done'

GFT_frames = 'GFT150_trimmed_Frames'
GFT_OF = 'GFT150_trimmed_Frames_OF'
delete_files_without_OF = True


def sanity_checks():
        for video in os.listdir(GFT_frames):
                in_dir = os.path.join(GFT_frames, video)
                OF_dir = os.path.join(GFT_OF, video)
                print('Processing {0}'.format(video))
                # rename files to remove non-number values from filenames
                #for file in os.listdir(OF_dir):
                #       os.rename(os.path.join(OF_dir, file), os.path.join(OF_dir, file.zfill(10)))
                OF_files = os.listdir(OF_dir)
                # check all files have their corresponding OF files otherwise print missing files
                for file in os.listdir(in_dir):
                       if file not in OF_files:
                               print('{0} does not have OF file for {1}'.format(video, file))
                               if delete_files_without_OF:
                                       os.remove(os.path.join(in_dir, file))
                                       print('Deleted {0} for missing OF file'.format(os.path.join(in_dir, file)))
                os.rename(OF_dir, in_dir+'_OF')
        return


def main():
        videos = [dir for dir in os.listdir(GFT_frames) if not dir.endswith('_OF')]
        for video in videos:
                run_CMD = CMD.replace('GFT_Demo', os.path.join(GFT_frames, video))
                print(run_CMD)
                #pdb.set_trace()
                os.system(run_CMD)



if __name__ == '__main__':
       main()
