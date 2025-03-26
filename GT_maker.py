import cv2
import numpy as np
import os

# 저장 경로
save_dir = '/home/ml/HAT/datasets/gt_black'
os.makedirs(save_dir, exist_ok=True)

# 이미지 생성 및 저장
for i in range(1, 7):  # 1부터 6까지
    black_image = np.zeros((4096, 4096, 3), dtype=np.uint8)
    save_path = os.path.join(save_dir, f'black_gt_{i:02d}.png')
    cv2.imwrite(save_path, black_image)
    print(f'Saved: {save_path}')
    
