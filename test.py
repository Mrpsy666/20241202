import os
import torch
from torchvision import transforms
from models import BPNet, LeNet
from PIL import Image
# 图像目录
image_dir = 'custom_images'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # 加载 BP 网络模型
    bp_model = BPNet().to(device)
    bp_model.load_state_dict(torch.load('BPNet.pth'))
    bp_model.eval()

    # 加载 LeNet 模型
    lenet_model = LeNet().to(device)
    lenet_model.load_state_dict(torch.load('LeNet.pth'))
    lenet_model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 存储文件名及预测结果
    filenames = []
    bp_preds = []
    lenet_preds = []

    # 遍历 'custom_images' 文件夹下的所有图片
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            filenames.append(filename)
            image_path = os.path.join(image_dir, filename)

            # 加载图片并预处理
            image = Image.open(image_path)
            image = transform(image)
            image = image.unsqueeze(0).to(device)  # 添加批量维度

            with torch.no_grad():
                # 使用 BP 网络进行预测
                bp_output = bp_model(image)
                bp_pred = torch.argmax(bp_output, dim=1).item()
                bp_preds.append(bp_pred)

                # 使用 LeNet 网络进行预测
                lenet_output = lenet_model(image)
                lenet_pred = torch.argmax(lenet_output, dim=1).item()
                lenet_preds.append(lenet_pred)


    print(f"{'Filename':<30} {'BP Network Prediction':<20} {'LeNet Prediction':<20}")
    print("=" * 70)

    for filename, bp_pred, lenet_pred in zip(filenames, bp_preds, lenet_preds):
        print(f"{filename:<30} {bp_pred:<20} {lenet_pred:<20}")


if __name__ == '__main__':
    main()
