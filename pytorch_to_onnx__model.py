import torch
import segmentation_models_pytorch as smp

if __name__ == '__main__':
    # 加载pytorch的模型
    model = smp.Unet(encoder_name="timm-mobilenetv3_small_minimal_100", in_channels=1, classes=1)
    weight = r'/data/medai05/EDAN2021/model/HC_mobileV3_min_100.pkl'
    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
    model.eval()

    # 输入示例
    batch_size = 1
    input_size = 256
    input = torch.randn(batch_size, 1, input_size, input_size)
    output = model(input)

    # onnx模型转换
    onnx_path = weight.split('.')[0]+'.onnx'
    torch.onnx.export(model, input, onnx_path,
                      export_params=True, opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={"input": {0: 'batch', 1: 'batch', 2: 'batch', 4: 'batch'},
                                    "output": {0: 'batch', 1: 'batch', 2: 'batch', 4: 'batch'}}
                      )
    # 最后dynamic是设置可能的batch_size
