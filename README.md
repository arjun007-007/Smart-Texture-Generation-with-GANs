# Smart Texture Generation with GANs

**Project Description**  
This project provides a **StyleGAN2-like** pipeline to generate realistic textures for 3D assets, intended for quick integration with **Unreal Engine**. It uses PyTorch for training and inference, and can process publicly available texture datasets (e.g., from Poly Haven).

---

## 1. Project Structure

```
smart_texture_gan_project/
├── data/
│   ├── raw/
│   │   └── polyhaven/
│   ├── processed/
│   │   └── polyhaven/
│   ...
├── scripts/
│   ├── preprocess_data.py
│   └── export_to_unreal.py
├── src/
│   ├── inference/
│   │   └── generate_textures.py
│   ├── models/
│   │   ├── generator.py
│   │   └── discriminator.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── trainer_utils.py
│   │   └── train.py
│   └── utils/
│       ├── file_io.py
│       ├── image_ops.py
│       └── visualization.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 2. Setup Instructions

### 2.1 Clone the Repository

```bash
git clone https://github.com/arjun007-007/Smart-Texture-Generation-with-GANs.git
cd smart_texture_gan_project
```

### 2.2 Install Dependencies

Below is a sample `requirements.txt`. If you do not use a virtual environment, run:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```
torch==2.0.1
torchvision==0.15.2
Pillow==9.5.0
matplotlib==3.7.1
numpy==1.23.5
requests==2.30.0
```

(Adjust versions as needed for your environment.)

### 2.3 Download & Organize Dataset

**Option A: Manual Download (Recommended)**
1. Go to [Poly Haven](https://polyhaven.com/textures) and download your chosen texture sets (choose *2K* resolution for a balanced approach).  
2. Copy all **albedo / color** images into `data/raw/polyhaven/`.

**Option B: Automated**  
If you have direct links, you can write a small Python script or use `wget`.  

---

## 3. Preprocessing

Resize images to a consistent resolution and save them in `data/processed/`:

```bash
python scripts/preprocess_data.py \
  --input_dir data/raw/polyhaven \
  --output_dir data/processed/polyhaven \
  --size 512
```

---

## 4. Training

Train your GAN using the processed dataset:

```bash
python src/training/train.py \
  --data_root data/processed/polyhaven \
  --output_dir checkpoints/ \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.0002 \
  --beta1 0.5 \
  --beta2 0.999 \
  --image_size 512 \
  --save_interval 10
```

- After each **save_interval** epochs, a checkpoint file will appear in **`checkpoints/`**.

---

## 5. Generating Textures

Use a trained checkpoint to generate textures:

```bash
python src/inference/generate_textures.py \
  --checkpoint_path checkpoints/checkpoint_final.pth \
  --output_dir assets/generated_textures \
  --num_samples 10 \
  --style_dim 512 \
  --image_size 512
```

Generated textures will be saved to **`assets/generated_textures/`**.

---

## 6. Export to Unreal Engine

Finally, copy your textures into an Unreal Engine project:

```bash
python scripts/export_to_unreal.py \
  --source_dir assets/generated_textures \
  --unreal_project_dir /path/to/MyUnrealProject \
  --texture_folder MyGANTextures
```

You can then open your Unreal project and find your textures in:
```
Content/MyGANTextures
```

---

## 7. Future Improvements
- **Mapping Network**: Incorporate a true mapping network for StyleGAN2 to enable style mixing.  
- **Tileability**: Implement advanced seamless texture techniques in `utils/image_ops.py`.  
- **Evaluation Metrics**: Add FID or Inception Score measurement.  
- **Material Generation**: Extend the pipeline to generate normal/roughness maps.

---

## 8. License & Credits

- Datasets from [Poly Haven](https://polyhaven.com/) are licensed under [CC0].  
- Code is shared under MIT License.  
