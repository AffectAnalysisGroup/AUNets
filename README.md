# AUNets
This repo provides a PyTorch implementation for AUNets. AUNets relies on the power of having independent and binaries CNNs for each facial expression. It works with the hollistical facial image *i.e.*, no keypoints or facial aligment needed. 

Project page: https://biomedicalcomputervision.uniandes.edu.co/index.php/research?id=30

## Citation
```
@article{romero2018multi,
  title={Multi-view dynamic facial action unit detection},
  author={Romero, Andr{\'e}s and Le{\'o}n, Juan and Arbel{\'a}ez, Pablo},
  journal={Image and Vision Computing},
  year={2018},
  publisher={Elsevier}
}
```

## Usage (TRAIN)
```bash
$./main.sh -GPU 0 -OF None #It will train AUNets (12 models and 3 folds) from emotionnet weights.
$./main.sh -GPU 0 -OF None -HYDRA true #It will train HydraNets (12 models and 3 folds) from emotionnet weights. 

$./main.sh -GPU 0 -OF None -finetuning imagenet #It will train AUNets (12 models and 3 folds) from imagenet weights. 
$./main.sh -GPU 0 -OF None -HYDRA true -finetuning imagenet #It will train HydraNets (12 models and 3 folds) from imagenet weights. 

# -OF options: None, Alone, Horizontal, Channels, Conv, FC6, FC7. [default=None].
# -finetuning options: emotionnet, imagenet, random. [default=emotionnet].
# -au options: 1,2,4,...,24. [default=all].
# -fold options: 0,1,2. [default=all].
# -mode_data options: normal, aligned. [default=normal].
```

## Usage (DEMO)
```bash
$./main.sh -AU 12 -gpu 0 -fold 0 -OF Horizontal -DEMO Demo

# -DEMO: folder or image location (absolute or relative). 
#        When OF!=None, DEMO must be a folder that contains *ONLY* RGB images. 
#        data_loader.py will assume $DEMO_OF is the folder where OF images are stored. 

# It will output the confidence score for each image in the folder, or for one single image if DEMO is a file.

# Example
$./main.sh -AU 12 -gpu 0 -fold 0 -OF None -DEMO Demo
./main.py -- --AU=12 --fold=0 --GPU=0 --OF None --DEMO Demo --finetuning=emotionnet --mode_data=normal
 [!!] loaded trained model: ./snapshot/models/BP4D/normal/fold_0/AU12/OF_None/emotionnet/02_1800.pth!
AU12 - OF None | Forward  | Demo/000.jpg : 0.00438882643357
AU12 - OF None | Forward  | Demo/012.jpg : 0.00548902712762
AU12 - OF None | Forward  | Demo/024.jpg : 0.00295104249381
AU12 - OF None | Forward  | Demo/036.jpg : 0.00390655593947
AU12 - OF None | Forward  | Demo/048.jpg : 0.00493786809966
```

## VGG16 Architecture
![](misc/arqs.png)

## Network (VGG - Fig (a))
![](misc/VGG16-OF_None.png)

Other Optical Flow stream architectures are found in `misc` folder

## Requirements

- Python 2.7 
- PyTorch 0.3.1
- Tensorflow (only if tensorboard flag)
- Other modules in `requirements.txt`

## Weights (*.pth models*)

Each model can be as heavy as 1GB for a total of 36GB per network variant (12 AUs - 3 folds). You can find them [here](http://157.253.243.8/weights/AUNets).  


## FAQs from code and paper
#How to run to test on images
```
If you are trying to run any setting with optical flow involved(Horizontal is the best performing model) generate the OF images using `OF_BP4D.py`. Makesure that the OF images folder has _OF suffix wrt the folder of original images. Then use the following commond to run on folder with images
`./main.py -m pdb -- --AU=12 --fold=0 --GPU=0 --OF Horizontal --DEMO Demo --mode_data=normal --pretrained_model='./fold_0/OF_Horizontal/AU10.pth' --mode='test'`
```

# Does repeated runs on same data give different results?
```No, this has been validated using the Demo data author provided. Make sure that you are using `pretrained_model` flag to load the model and not `test_model` flag.```

# Is the output smoothed as mentioned in the paper?
```No. The Demo option does not do that. Its not confirm if the given code does that in any of its settings. But the smoothing is performed on the binary decision after thresholding the model output so its largely irrelevant on other datasets than mentioned in the paper.```

## Results using [misc/VGG16-OF_Horizontal.png](misc/VGG16-OF_Horizontal.png)
These results are reported using three-fold cross validation over the BP4D dataset. 

<table>
<thead>
<tr class="header">
<th style="text-align: right;"><strong>AU</strong></th>
<th style="text-align: center;">1</th>
<th style="text-align: center;">2</th>
<th style="text-align: center;">4</th>
<th style="text-align: center;">6</th>
<th style="text-align: center;">7</th>
<th style="text-align: center;">10</th>
<th style="text-align: center;">12</th>
<th style="text-align: center;">14</th>
<th style="text-align: center;">15</th>
<th style="text-align: center;">17</th>
<th style="text-align: center;">23</th>
<th style="text-align: center;">24</th>
<th style="text-align: center;"><strong><em>Av.</em></strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;"><p><strong>F1</strong></p></td>
<td style="text-align: center;">53.4</td>
<td style="text-align: center;">44.7</td>
<td style="text-align: center;">55.8</td>
<td style="text-align: center;">79.2</td>
<td style="text-align: center;">78.1</td>
<td style="text-align: center;">83.1</td>
<td style="text-align: center;">88.4</td>
<td style="text-align: center;">66.6</td>
<td style="text-align: center;">47.5</td>
<td style="text-align: center;">62.0</td>
<td style="text-align: center;">47.3</td>
<td style="text-align: center;">49.7</td>
<td style="text-align: center;"><strong>63.0</strong></td>
</tr>
</tbody>
</table>

____
## Three fold:
Based on [DRML](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhao_Deep_Region_and_CVPR_2016_paper.pdf) paper, we use their exact subject-exclusive three fold testing (These subjects are exclusively for testing on each fold, the remaining subjects are for train/val):
<table>
<thead>
<tr class="header">
<th style="text-align: right;"><strong>Fold</strong></th>
<th style="text-align: center;"><strong><em>Subjects</em></strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: right;"><p><strong>1</strong></p></td>
<td style="text-align: center;">['F001', 'F002', 'F008', 'F009', 'F010', 'F016', 'F018', 'F023', 'M001', 'M004', 'M007', 'M008', 'M012', 'M014']</td>
</tr>
</tbody>
<tr class="odd">
<td style="text-align: right;"><p><strong>2</strong></p></td>
<td style="text-align: center;">['F003', 'F005', 'F011', 'F013', 'F020', 'F022', 'M002', 'M005', 'M010', 'M011', 'M013', 'M016', 'M017', 'M018']</td>
</tr>
</tbody>
<tr class="odd">
<td style="text-align: right;"><p><strong>3</strong></p></td>
<td style="text-align: center;">['F004', 'F006', 'F007', 'F012', 'F014', 'F015', 'F017', 'F019', 'F021', 'M003', 'M006', 'M009', 'M015']</td>
</tr>
</tbody>
</table>
