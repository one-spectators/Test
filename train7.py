import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from common1 import Convolution, Relu, Adam, Sigmoid,BatchNormalization


class DenoisingModel:
    def __init__(self, device):
        self.device = device

        # 初始化卷积层和BN层
        self.conv1 = self._conv_layer(3, 16, 3, device)
        self.bn1 = BatchNormalization(16, device=device)
        self.relu1 = Relu()

        self.conv2 = self._conv_layer(16, 32, 3, device)
        self.bn2 = BatchNormalization(32, device=device)
        self.relu2 = Relu()

        self.conv3 = self._conv_layer(32, 64, 3, device)
        self.bn3 = BatchNormalization(64, device=device)
        self.relu3 = Relu()

        self.conv4 = self._conv_layer(64, 32, 3, device)
        self.bn4 = BatchNormalization(32, device=device)
        self.relu4 = Relu()

        self.conv5 = self._conv_layer(32, 16, 3, device)
        self.bn5 = BatchNormalization(16, device=device)
        self.relu5 = Relu()

        self.conv6 = self._conv_layer(16, 3, 3, device)
        self.sigmoid = Sigmoid()

        # 优化器需管理BN层参数
        self.optimizer = Adam(lr=0.001)

    def _conv_layer(self, in_c, out_c, kernel_size, device):
        # He初始化逻辑保持不变
        scale = (2. / (in_c * kernel_size ** 2)) ** 0.5
        W = torch.randn(out_c, in_c, kernel_size, kernel_size, device=device) * scale
        b = torch.zeros(out_c, device=device)
        return Convolution(W, b, stride=1, pad=1)

    def forward(self, x):
        # 前向传播流程（插入BN层）
        x = self.conv1.forward(x)
        x = self.bn1.forward(x, train_flg=True)
        x = self.relu1.forward(x)

        x = self.conv2.forward(x)
        x = self.bn2.forward(x, train_flg=True)
        x = self.relu2.forward(x)

        x = self.conv3.forward(x)
        x = self.bn3.forward(x, train_flg=True)
        x = self.relu3.forward(x)

        x = self.conv4.forward(x)
        x = self.bn4.forward(x, train_flg=True)
        x = self.relu4.forward(x)

        x = self.conv5.forward(x)
        x = self.bn5.forward(x, train_flg=True)
        x = self.relu5.forward(x)

        x = self.conv6.forward(x)
        x = self.sigmoid.forward(x)
        return x

    def backward(self, dout):
        dout = self.sigmoid.backward(dout)
        dout = self.conv6.backward(dout)

        dout = self.relu5.backward(dout)
        dout = self.bn5.backward(dout)
        dout = self.conv5.backward(dout)

        dout = self.relu4.backward(dout)
        dout = self.bn4.backward(dout)
        dout = self.conv4.backward(dout)

        dout = self.relu3.backward(dout)
        dout = self.bn3.backward(dout)
        dout = self.conv3.backward(dout)

        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

        # 收集所有参数梯度（包括BN层）
        grads = {
            # 卷积层梯度
            'W1': self.conv1.dW, 'b1': self.conv1.db,
            'W2': self.conv2.dW, 'b2': self.conv2.db,
            'W3': self.conv3.dW, 'b3': self.conv3.db,
            'W4': self.conv4.dW, 'b4': self.conv4.db,
            'W5': self.conv5.dW, 'b5': self.conv5.db,
            'W6': self.conv6.dW, 'b6': self.conv6.db,

            # BN层梯度
            'gamma1': self.bn1.dgamma, 'beta1': self.bn1.dbeta,
            'gamma2': self.bn2.dgamma, 'beta2': self.bn2.dbeta,
            'gamma3': self.bn3.dgamma, 'beta3': self.bn3.dbeta,
            'gamma4': self.bn4.dgamma, 'beta4': self.bn4.dbeta,
            'gamma5': self.bn5.dgamma, 'beta5': self.bn5.dbeta
        }

        # 所有可训练参数（包括BN层参数）
        params = {
            # 卷积层参数
            'W1': self.conv1.W, 'b1': self.conv1.b,
            'W2': self.conv2.W, 'b2': self.conv2.b,
            'W3': self.conv3.W, 'b3': self.conv3.b,
            'W4': self.conv4.W, 'b4': self.conv4.b,
            'W5': self.conv5.W, 'b5': self.conv5.b,
            'W6': self.conv6.W, 'b6': self.conv6.b,

            # BN层参数
            'gamma1': self.bn1.gamma, 'beta1': self.bn1.beta,
            'gamma2': self.bn2.gamma, 'beta2': self.bn2.beta,
            'gamma3': self.bn3.gamma, 'beta3': self.bn3.beta,
            'gamma4': self.bn4.gamma, 'beta4': self.bn4.beta,
            'gamma5': self.bn5.gamma, 'beta5': self.bn5.beta
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

            # ------------------------- 数据增强 -------------------------
            # 1. 随机水平/垂直翻转
            if np.random.rand() > 0.5:
                batch_noisy = torch.flip(batch_noisy, dims=[3])  # 水平翻转
                batch_clean = torch.flip(batch_clean, dims=[3])
            if np.random.rand() > 0.5:
                batch_noisy = torch.flip(batch_noisy, dims=[2])  # 垂直翻转
                batch_clean = torch.flip(batch_clean, dims=[2])

            # 2. 随机裁剪到224x224
            crop_size = 224
            _, _, H, W = batch_noisy.shape
            top = torch.randint(0, H - crop_size, (1,)).item()
            left = torch.randint(0, W - crop_size, (1,)).item()
            batch_noisy = batch_noisy[:, :, top:top + crop_size, left:left + crop_size]
            batch_clean = batch_clean[:, :, top:top + crop_size, left:left + crop_size]

            # 3. 颜色抖动（亮度+对比度）
            # 亮度调整（0.8~1.2）
            # 对比度调整（0.8~1.2）
            contrast = torch.rand(1, device=batch_noisy.device) * 0.4 + 0.8
            mean = torch.mean(batch_noisy, dim=(2, 3), keepdim=True)
            batch_noisy = (batch_noisy - mean) * contrast + mean
            batch_noisy = torch.clamp(batch_noisy, 0.0, 1.0)
            # -----------------------------------------------------------

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