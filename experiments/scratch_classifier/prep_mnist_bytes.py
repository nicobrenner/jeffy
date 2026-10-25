"""Convert MNIST images to raw byte files for byte-level LM training."""
import numpy as np
from pathlib import Path
from torchvision import datasets

out_dir = Path('pile_runs/custom_vocab/byte_mnist')
out_dir.mkdir(parents=True, exist_ok=True)

for split, train in [('train', True), ('val', False)]:
    ds = datasets.MNIST(root='/tmp/mnist_data', train=train, download=True)
    pixels = ds.data.numpy().astype(np.uint8).reshape(-1)
    out_path = out_dir / f'{split}.bytes'
    pixels.tofile(str(out_path))
    n_images = len(ds)
    print(f'{split}: {n_images} images, {len(pixels):,} bytes ({len(pixels)/1e6:.1f} MB) -> {out_path}')

print(f'\nImage shape: 28x28 = 784 bytes each')
print(f'Pixel range: 0-255 (grayscale)')
print(f'Ready for byte-level LM training')
