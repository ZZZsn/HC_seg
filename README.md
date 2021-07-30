# 基于深度学习的超声胎儿头围分割
2021理邦竞赛-胎儿头围测量项目

## **主要代码描述：**

preprocess.py --预处理代码

main.py --训练运行代码，在这里设置与修改参数

solver.py --交叉验证的主体代码，训练验证都在这里，在这里修改使用的模型与损失函数

test.py --测试运行代码，在这里对测试集进行测试

## **数据存放方式**

/data 为赛方发放数据，其中包含以下内容:  
- training_set  训练集数据  
- test_set  测试集数据  
- training_set_pixel_size_and_HC.csv  描述训练集每张图像的像素大小以及头围  
- test_set_pixel_size.csv  描述测试集每张图像的像素大小  

其中训练集将图像与标注分开成两个文件夹：/data/images和/data/labels，预处理后生成的/data/labels_pre
