DOWNSCALE = 2
DOWNSCALE_WAY = 'bicubic'
# DOWNSCALE_WAY = 'unknown'

if DOWNSCALE == 2:
    LR_SIZE = 48
    HR_SIZE = 96
elif DOWNSCALE == 3:
    LR_SIZE = 48
    HR_SIZE = 144
elif DOWNSCALE == 4:
    LR_SIZE = 48
    HR_SIZE = 192

WEIGHT_DIR = 'models/weights/srgan'
COMPARISON_DIR = 'results/comparison'
LOG_DIR = f'results/training_{DOWNSCALE_WAY}_x{DOWNSCALE}.log'

