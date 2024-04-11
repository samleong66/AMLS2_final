'''''''''
Customize the experiment parameter
'''''''''
DOWNSCALE = 2
DOWNSCALE_WAY = 'bicubic'
# DOWNSCALE_WAY = 'unknown'

'''''''''
Crop size
'''''''''

if DOWNSCALE == 2:
    LR_SIZE = 48
    HR_SIZE = 96
elif DOWNSCALE == 3:
    LR_SIZE = 48
    HR_SIZE = 144
elif DOWNSCALE == 4:
    LR_SIZE = 48
    HR_SIZE = 192

'''''''''
File path
'''''''''

WEIGHT_DIR = 'A'
COMPARISON_DIR = 'results/comparison'
LOG_DIR = f'results/training_{DOWNSCALE_WAY}_x{DOWNSCALE}.log'

