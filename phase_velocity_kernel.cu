from numba import cuda
from time import time

@cuda.jit
def calculate_cgh_kernel(train_points, phase_map, slm_res, slm_pitch, k_val):
    """
    GPU 커널: 3D 점광원 데이터를 기반으로 SLM 구동을 위한 위상 맵을 계산.
    Fresnel 근사를 이용한 위상 계산의 병렬 구현.
    """
    
    slm_x, slm_y = cuda.grid(2)
    
    if slm_x < slm_res[0] and slm_y < slm_res[1]:
        # SLM 픽셀의 물리적 위치 (중앙 기준)
        r_x = (slm_x - slm_res[0] / 2) * slm_pitch
        r_y = (slm_y - slm_res[1] / 2) * slm_pitch
        
        # 복소수 진폭 합계 초기화 (실수, 허수)
        # Numba의 복소수 지원이 제한적이므로 실수/허수를 분리하여 계산
        complex_amp_real = 0.0
        complex_amp_imag = 0.0
        
        # 모든 점광원에 대해 합산 (병렬 처리의 핵심)
        for i in range(train_points.shape[0]):
            p_x, p_y, p_z, intensity = train_points[i]
            
            # 거리 R 계산: sqrt((dx)^2 + (dy)^2 + (dz)^2)
            dx = p_x - r_x
            dy = p_y - r_y
            dz = p_z # z축 거리는 공통
            R = cuda.device.sqrt(dx*dx + dy*dy + dz*dz)
            
            # 위상 계산: phase = k * R
            phase = k_val * R
            
            # 복소수 합산: complex_amp += intensity * e^(i * phase)
            complex_amp_real += intensity * cuda.device.cos(phase)
            complex_amp_imag += intensity * cuda.device.sin(phase)
        
        # 최종 위상 계산: atan2(imag, real)
        final_phase = cuda.device.atan2(complex_amp_imag, complex_amp_real)
        
        # 위상 맵에 0 ~ 2*pi 범위로 저장
        phase_map[slm_x, slm_y] = final_phase % (2 * np.pi)


def render_holographic_pattern(train_data, slm_res, slm_pitch, k_val):
    """CGH 렌더링을 위한 메인 실행 함수"""
    start_time = time()

    # 1. GPU 메모리 전송
    d_train_points = cuda.to_device(train_data)
    d_phase_map = cuda.device_array(slm_res, dtype=np.float32)
    
    # 2. 커널 실행 설정 및 호출
    threadsperblock = (16, 16)
    blockspergrid_x = (slm_res[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (slm_res[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    calculate_cgh_kernel[blockspergrid, threadsperblock](
        d_train_points, d_phase_map, SLM_RESOLUTION, SLM_PIXEL_PITCH, K
    )
    cuda.synchronize() # GPU 작업 완료 대기
    
    # 3. 결과(Phase Map)를 CPU로 복사
    phase_map_host = d_phase_map.copy_to_host()
    
    end_time = time()
    print(f"⏱️ CGH Calculation Time: {end_time - start_time:.4f} seconds.")
    
    return phase_map_host
