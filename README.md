### 模型组成  
object detection + auto encoder + siamese network

### 数据集格式
#### OD 
输入：customeDataset/OD/train/8    
输出：./output/OD/train  
#### AE
输入：  
训练集：customeDataset/AE/train/0.normal  
测试集：customeDataset/AE/test/0.normal  customeDataset/AE/test/1.abnormal   
输出：./ouput/ganomaly/train
#### Siamese
输入：customeDataset/siamese/train/0.normal  customeDataset/siamese/train/1.abnormal    
输出：./output/siamese/train  
### CLI
训练：  
测试：  