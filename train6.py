import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from common1 import Convolution, Relu, Adam, Sigmoid,im2col,col2im

class DenoisingModel:
    def __init__(self, device):
        self.device = device
        # 初始化卷积层（He初始化）
        self.conv1 = self._conv_layer(3, 16, 3, device)
        self.relu1 = Relu()
        self.conv2 = self._conv_layer(16, 32, 3, device)
        self.relu2 = Relu()
        self.conv3 = self._conv_layer(32, 16, 3, device)
        self.relu3 = Relu()
        self.conv4 = self._conv_layer(16, 3, 3, device)
        self.sigmoid = Sigmoid()  # 使用类实现Sigmoid

        self.optimizer = Adam()

    def _conv_layer(self, in_c, out_c, kernel_size, device):
        # He初始化
        scale = (2. / (in_c * kernel_size ** 2)) ** 0.5
        W = torch.randn(out_c, in_c, kernel_size, kernel_size, device=device) * scale
        b = torch.zeros(out_c, device=device)
        return Convolution(W, b, stride=1, pad=1)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.conv4.forward(x)
        x = self.sigmoid.forward(x)  # 确保输出在[0,1]
        return x

    def backward(self, dout):
        dout = self.sigmoid.backward(dout)
        dout = self.conv4.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.conv3.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)

        # 参数更新
        grads = {
            'W1': self.conv1.dW, 'b1': self.conv1.db,
            'W2': self.conv2.dW, 'b2': self.conv2.db,
            'W3': self.conv3.dW, 'b3': self.conv3.db,
            'W4': self.conv4.dW, 'b4': self.conv4.db
        }
        params = {
            'W1': self.conv1.W, 'b1': self.conv1.b,
            'W2': self.conv2.W, 'b2': self.conv2.b,
            'W3': self.conv3.W, 'b3': self.conv3.b,
            'W4': self.conv4.W, 'b4': self.conv4.b
        }
        self.optimizer.update(params, grads)

    def predict(self, x):
        return self.forward(x)


def load_dataset(noisy_dir, clean_dir, device):
    noisy_images = []
    clean_images = []
    target_size = (256, 256)
    for filename in os.listdir(noisy_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 确保读取为RGB格式
            noisy_img = Image.open(os.path.join(noisy_dir, filename)).convert('RGB')
            clean_img = Image.open(os.path.join(clean_dir, filename)).convert('RGB')
            noisy_img = noisy_img.resize(target_size)
            clean_img = clean_img.resize(target_size)

            # 转换为Tensor并归一化
            noisy_tensor = torch.tensor(np.array(noisy_img).transpose(2, 0, 1) / 255.0,
                                        dtype=torch.float32, device=device)
            clean_tensor = torch.tensor(np.array(clean_img).transpose(2, 0, 1) / 255.0,
                                        dtype=torch.float32, device=device)

            noisy_images.append(noisy_tensor)
            clean_images.append(clean_tensor)

    # 堆叠张量并添加批处理维度
    noisy_images = torch.stack(noisy_images)
    clean_images = torch.stack(clean_images)
    return noisy_images, clean_images


def train(model, noisy_images, clean_images, epochs=10, batch_size=16):
    losses = []
    num_samples = len(noisy_images)
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        total_loss = 0.0
        # 打乱数据顺序
        indices = torch.randperm(num_samples)
        noisy_images = noisy_images[indices]
        clean_images = clean_images[indices]

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_noisy = noisy_images[start:end]
            batch_clean = clean_images[start:end]

            # 前向传播
            output = model.forward(batch_noisy)
            loss = torch.mean((output - batch_clean) ** 2)
            total_loss += loss.item()

            # 反向传播
            dout = 2 * (output - batch_clean) / batch_size
            model.backward(dout)

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    return losses


from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_metrics(clean_img, denoised_img):
    """
    计算PSNR和SSIM指标
    参数形状应为HWC格式且范围在[0,1]
    """
    # 转换到0-255范围并转为uint8
    clean = (clean_img * 255).astype(np.uint8)
    denoised = (denoised_img * 255).astype(np.uint8)

    # 计算PSNR
    psnr = peak_signal_noise_ratio(clean, denoised)

    # 计算SSIM（多通道图像需指定channel_axis）
    ssim = structural_similarity(clean, denoised,
                                 channel_axis=2,
                                 data_range=255)
    return psnr, ssim


def test(model, noisy_images, clean_images):
    with torch.no_grad():
        output = model.predict(noisy_images)
        loss = torch.mean((output - clean_images) ** 2).item()

        # 转换为numpy并调整维度
        denoised_np = output.cpu().numpy().transpose(0, 2, 3, 1)
        clean_np = clean_images.cpu().numpy().transpose(0, 2, 3, 1)

        # 计算指标
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = len(denoised_np)

        for i in range(num_samples):
            psnr, ssim = calculate_metrics(clean_np[i], denoised_np[i])
            total_psnr += psnr
            total_ssim += ssim

        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples

        print(f'Test Loss: {loss:.4f}')
        print(f'Average PSNR: {avg_psnr:.2f} dB')
        print(f'Average SSIM: {avg_ssim:.4f}')

        return denoised_np
def plot_comparison(noisy, clean, denoised, num=3):
    plt.figure(figsize=(15, 5 * num))
    for i in range(num):
        plt.subplot(num, 3, i * 3 + 1)
        plt.imshow(noisy[i])
        plt.title('Noisy')
        plt.axis('off')

        plt.subplot(num, 3, i * 3 + 2)
        plt.imshow(clean[i])
        plt.title('Clean')
        plt.axis('off')

        plt.subplot(num, 3, i * 3 + 3)
        plt.imshow(denoised[i])
        plt.title('Denoised')
        plt.axis('off')
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    noisy_dir = 'output_images'
    clean_dir = 'images'
    noisy_images, clean_images = load_dataset(noisy_dir, clean_dir, device)

    # 初始化模型
    model = DenoisingModel(device)

    # 训练
    losses = train(model, noisy_images, clean_images, epochs=150, batch_size=8)

    # 测试
    denoised = test(model, noisy_images, clean_images)

    # 可视化
    plot_comparison(
        noisy_images.cpu().numpy().transpose(0, 2, 3, 1),
        clean_images.cpu().numpy().transpose(0, 2, 3, 1),
        denoised
    )
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curve')
    plt.show()


if __name__ == "__main__":
    main()