import torch
import segmentation_models_pytorch as smp

if __name__ == '__main__':
    device = torch.device('cpu')
    # 加载pytorch的模型
    model = smp.Unet(encoder_name="timm-regnety_032", in_channels=1, classes=1)
    weight = r'/data/medai05/EDAN2021/best_unet_score.pkl'
    model.load_state_dict(torch.load(weight, map_location=device))
    model.eval()

    # 输入示例
    input = torch.randn(1, 1, 512, 512).to(device)

    # 模型转换
    traced_model = torch.jit.trace(model, input)
    # 保存转换后的模型
    traced_model.save('/data/medai05/EDAN2021/HC_seg.pt')
