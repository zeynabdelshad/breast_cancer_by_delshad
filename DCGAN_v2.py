import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import sys

# ======================================================================================
# مرحله ۱: پیکربندی و تنظیمات اصلی
# ======================================================================================

# --- مسیرها (این بخش را متناسب با سیستم خود تغییر دهید) ---
# مسیری که تصاویر اصلی و بزرگ (1280x1280) شما در آن قرار دارد
# توجه: این پوشه باید شامل زیرپوشه‌ای باشد که تصاویر در آن هستند (مثلا /.../large_images/class_A/img1.png)
# این ساختار برای سازگاری با ImageFolder لازم است.
ORIGINAL_IMAGES_DIR = '/home/natlal/burst_cancer/dataset_large' 

# مسیری که دیتاست جدید (پچ‌ها) در آن ساخته خواهد شد
PATCHES_DATASET_DIR = '/home/natlal/burst_cancer/patched_dataset'

# مسیری که تصاویر نهایی تولید شده و مدل ذخیره می‌شوند
OUTPUT_DIR = 'GAN_Output'

# --- هایپرپارامترها ---
IMAGE_SIZE = 1280         # اندازه تصویر نهایی
PATCH_SIZE = 64           # اندازه هر پچ
LATENT_DIM = 100          # ابعاد وکتور نویز ورودی به مولد
BATCH_SIZE = 64           # اندازه بچ برای آموزش
NUM_EPOCHS = 50           # تعداد اپک‌های آموزش (برای تست می‌توانید عدد کمتری بگذارید)
LR = 0.0002               # نرخ یادگیری
BETA1 = 0.5               # پارامتر Adam optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ساخت پوشه‌های خروجی
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================================
# مرحله ۲: تابع ساخت دیتاست پچ‌ها
# ======================================================================================

def create_patches_from_directory(input_dir, output_dir, patch_size):
    """
    تصاویر بزرگ در `input_dir` را به پچ‌های کوچک تقسیم کرده و در `output_dir` ذخیره می‌کند.
    این تابع ساختار زیرپوشه‌ها را برای سازگاری با ImageFolder حفظ می‌کند.
    """
    if os.path.exists(output_dir):
        print(f"Directory '{output_dir}' already exists. Skipping patch creation.")
        return

    print(f"Creating patches from '{input_dir}' into '{output_dir}'...")
    os.makedirs(output_dir)
    
    total_patches = 0
    # پیدا کردن زیرپوشه‌ها (کلاس‌ها)
    for class_name in os.listdir(input_dir):
        class_dir_in = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir_in):
            continue
        
        # ساخت زیرپوشه متناظر در خروجی
        class_dir_out = os.path.join(output_dir, class_name)
        os.makedirs(class_dir_out)

        image_files = [f for f in os.listdir(class_dir_in) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for filename in image_files:
            try:
                img_path = os.path.join(class_dir_in, filename)
                img = Image.open(img_path).convert("RGB")
                width, height = img.size

                for i in range(0, height, patch_size):
                    for j in range(0, width, patch_size):
                        if (j + patch_size <= width) and (i + patch_size <= height):
                            box = (j, i, j + patch_size, i + patch_size)
                            patch = img.crop(box)
                            patch.save(os.path.join(class_dir_out, f"patch_{total_patches}.png"))
                            total_patches += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Finished creating {total_patches} patches.")


# ======================================================================================
# مرحله ۳: تعریف مدل‌های Generator و Discriminator
# ======================================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ======================================================================================
# مرحله ۵: تابع تولید و ترکیب تصویر نهایی
# ======================================================================================

def generate_and_stitch_image(generator, device, output_filename):
    """
    با استفاده از مولد آموزش‌دیده، یک تصویر بزرگ تولید و ذخیره می‌کند.
    """
    print("\nStarting final image generation and stitching...")
    generator.eval() # مدل را به حالت ارزیابی ببرید

    patches_per_dim = IMAGE_SIZE // PATCH_SIZE
    num_patches = patches_per_dim * patches_per_dim

    with torch.no_grad():
        noise = torch.randn(num_patches, LATENT_DIM, 1, 1, device=device)
        generated_patches = generator(noise).cpu()

    # ایجاد یک تصویر بزرگ خالی
    final_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))

    patch_idx = 0
    for i in range(patches_per_dim):
        for j in range(patches_per_dim):
            # تبدیل تنسور به تصویر PIL
            patch_tensor = generated_patches[patch_idx]
            patch_img = transforms.ToPILImage()(patch_tensor * 0.5 + 0.5) # Un-normalize
            
            # چسباندن پچ در مکان صحیح
            final_image.paste(patch_img, (j * PATCH_SIZE, i * PATCH_SIZE))
            patch_idx += 1
            
    final_image.save(output_filename)
    print(f"Successfully generated and saved the large image to '{output_filename}'")


# ======================================================================================
# بخش اصلی: اجرای مراحل
# ======================================================================================

if __name__ == '__main__':
    # مرحله ۱: ساخت دیتاست از پچ‌ها
    create_patches_from_directory(ORIGINAL_IMAGES_DIR, PATCHES_DATASET_DIR, PATCH_SIZE)

    # مرحله ۲: آماده‌سازی برای آموزش
    transform = transforms.Compose([
        transforms.Resize(PATCH_SIZE),
        transforms.CenterCrop(PATCH_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    try:
        dataset = datasets.ImageFolder(root=PATCHES_DATASET_DIR, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    except FileNotFoundError:
        print(f"Error: Patched dataset not found at '{PATCHES_DATASET_DIR}'.")
        print("Please ensure the ORIGINAL_IMAGES_DIR is correct and has images.")
        sys.exit(1)


    # ساخت مدل‌ها
    netG = Generator(LATENT_DIM).to(DEVICE)
    netG.apply(weights_init)

    netD = Discriminator().to(DEVICE)
    netD.apply(weights_init)

    print(f"Using device: {DEVICE}")
    print("Generator Model:\n", netG)
    print("\nDiscriminator Model:\n", netD)


    # تعریف تابع هزینه و بهینه‌سازها
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE) # برای مشاهده پیشرفت
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

    # مرحله ۳: حلقه آموزش
    print("\nStarting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataloader, 0):
            # (1) آپدیت شبکه Discriminator
            netD.zero_grad()
            real_cpu = data[0].to(DEVICE)
            b_size = real_cpu.size(0)
            
            label = torch.full((b_size,), 1., dtype=torch.float, device=DEVICE)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake = netG(noise)
            label.fill_(0.)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) آپدیت شبکه Generator
            netG.zero_grad()
            label.fill_(1.) # لیبل‌های فیک برای مولد واقعی هستند
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            # نمایش لاگ
            if i % 50 == 0:
                print(f'[{epoch+1}/{NUM_EPOCHS}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

        # ذخیره نمونه تصاویر تولید شده در هر اپک برای مشاهده پیشرفت
        with torch.no_grad():
            fake_samples = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake_samples, f'{OUTPUT_DIR}/epoch_{epoch+1}_fake_samples.png', normalize=True)

    print("Finished Training.")

    # ذخیره مدل آموزش‌دیده
    torch.save(netG.state_dict(), f'{OUTPUT_DIR}/generator_final.pth')
    print(f"Generator model saved to '{OUTPUT_DIR}/generator_final.pth'")

    # مرحله ۴: تولید تصویر بزرگ نهایی
    generate_and_stitch_image(netG, DEVICE, f'{OUTPUT_DIR}/final_stitched_image.png')