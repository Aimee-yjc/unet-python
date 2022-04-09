# Image Segmentation (UNet)
## datastes
download BraTS2019
```shell
3d_data to 2d_data ( run getdata.py ): 

- flair_train, seg_train = 3d_train_datapath ( flair.nii, seg.nii )

- flair_test, seg_test = 3d_test_datapath ( flair.nii, seg.nii )

- dir = 2d_data_savepath

in my code :
./data/train : 2d_train_datapath
./data/val : 2d_test_datapath


```
## How to use
Dependencies
```shell
python==3.9
cuda=11.3
cudnn=8.0
pytorch=1.10
```
Run main.py
```shell
- def train() : trainning model
  
- def test() : test model and you will see the predicted results of test image
```
My blog : https://blog.csdn.net/weixin_41911781/article/details/124015137
