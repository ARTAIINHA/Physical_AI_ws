import pandas as pd
import math

# 경로 파일 이름 확인
csv_path = '/home/deoduck/route/path_planning_result.csv'

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 첫 두 점의 좌표 추출
X0, Y0 = df['X_coordinate (m)'][0], df['Y_coordinate (m)'][0]
X1, Y1 = df['X_coordinate (m)'][1], df['Y_coordinate (m)'][1]

# 시작 Yaw 각도 계산 (라디안)
start_yaw = math.atan2(Y1 - Y0, X1 - X0)

print(f"P0: ({X0:.2f}, {Y0:.2f})")
print(f"P1: ({X1:.2f}, {Y1:.2f})")
print(f"계산된 시작 Yaw 값 (라디안): {start_yaw:.4f}")
