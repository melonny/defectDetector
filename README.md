### intro  
this project is for graduation  
as performance of single method is bad for industrial defect detection, i'll combine od, ae, and extra conv together, let's try!  
### model structure  
object detection + auto encoder + siamese network

### dataset  
#### OD 
input：customeDataset/OD/train/8    
output：./output/OD/train  
#### AE
input：  
train：customeDataset/AE/train/0.normal  
test：customeDataset/AE/test/0.normal  customeDataset/AE/test/1.abnormal   
output：./ouput/ganomaly/train
#### Siamese
input：customeDataset/siamese/train/0.normal  customeDataset/siamese/train/1.abnormal    
output：./output/siamese/train  
### CLI
train：python train.py --model aesiamese    
test：python test.py --model aesiamese  
