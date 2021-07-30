# 基于深度学习的超声胎儿头围分割
2021理邦竞赛-胎儿头围测量项目

## **主要代码描述：**

preprocess.py --预处理运行代码

main.py --训练运行代码，在这里设置与修改参数

solver.py --交叉验证的主体代码，训练验证都在这里，在这里修改使用的模型与损失函数

test.py --测试运行代码，在这里对测试集进行测试

## **数据存放方式**

/data 为所有需要的数据，其中包含以下内容:  
- training_set  - 训练集数据  
- test_set  - 测试集数据  
- training_set_pixel_size_and_HC.csv  - 描述训练集每张图像的像素大小以及头围  
- test_set_pixel_size.csv  - 描述测试集每张图像的像素大小  
- train.csv  - 预处理生成的用于分折的csv文档  
- test_set_result.csv  - 在test_set_pixel_size.csv 基础上加入模型**测试结果**的csv文档

训练集里将图像与标注分开成两个文件夹：/data/training_set/images和/data/training_set/labels，预处理后会另外生成/data/training_set/labels_pre  

测试集里将图像放入一个文件夹：/data/test_set/images，测试后将会生成/data/test_set/img_mask（原图上画了椭圆）和/data/test_set/pre_mask（测试结果mask）

## 测试
/model里提供了最终使用的模型，分别基于RegNetY-064、RegNetY-002和MobileNetV2，修改test里的模型路径即可运行  
```onnx_weight = r"./model/HC_mobileV2.onnx" ```
