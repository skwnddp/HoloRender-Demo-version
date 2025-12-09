import numpy as np
import random
import os

# --- 1. 설정 및 광학 파라미터 정의 ---
WAVELENGTH = 532e-9  # 532nm (녹색 레이저 가정)
K = 2 * np.pi / WAVELENGTH
SLM_RESOLUTION = (2048, 2048) # SLM 픽셀 해상도
SLM_PIXEL_PITCH = 8e-6 # SLM 픽셀 간격
VIEWING_DISTANCE = 0.5 # 홀로그램 맺힘 거리 (0.5m)
POINT_COUNT = 100000 # 기차를 표현할 점광원의 수 (가정)

# --- 2. 3D 모델 파싱 및 점광원 생성 (시뮬레이션) ---
def parse_and_generate_points(point_count):
    """
    3D 기차 모델 파일을 읽어와 N개의 점광원 데이터셋을 생성한다고 가정.
    실제로는 OBJ/STL 파서가 필요하지만, 여기서는 임의의 기차 모양을 시뮬레이션.
    Data format: [x, y, z, intensity]
    """
    print("🚂 Parsing 3D Train Model...")
    points = []
    
    # 기차 본체 시뮬레이션 (X: 길이, Y: 높이, Z: 깊이)
    for _ in range(point_count):
        x = random.uniform(-0.5, 0.5)  # 기차 길이
        y = random.uniform(-0.1, 0.3)  # 기차 높이
        z = random.uniform(VIEWING_DISTANCE - 0.1, VIEWING_DISTANCE + 0.1) # 깊이 (맺힘 거리 근처)
        
        # 간단한 강도 변화 (예: 기차 앞부분이 더 밝게)
        intensity = 1.0 - abs(x) * 0.5 
        
        points.append([x, y, z, intensity])

    points = np.array(points, dtype=np.float32)
    print(f"✅ Generated {points.shape[0]} Point Sources for CGH.")
    return points
