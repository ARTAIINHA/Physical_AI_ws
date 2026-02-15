import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
import csv
import os

# ====================================================
# [⭐ 20개의 랜덤 원통 장애물 좌표 (Gazebo와 일치해야 함)]
# ====================================================
RANDOM_OBS_COORDS = [
    (-7.2, 5.1), (3.9, -6.8), (1.1, 7.5), (-6.0, 0.4), (7.8, -4.9),
    (4.5, 2.0), (-3.3, 7.1), (0.2, -5.5), (6.5, 3.4), (-4.1, -1.9),
    (1.9, 1.2), (-0.8, -7.0), (5.3, -0.1), (-7.5, 3.8), (2.5, 5.9),
    (-1.5, 4.3), (7.0, -2.8), (-5.8, -6.5), (3.1, -0.9), (-2.2, 2.7)
]
# ====================================================


# --- 1. A* 노드 구조 및 유틸리티 함수 ---
class Node:
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index
    def __lt__(self, other):
        return self.cost < other.cost

def calc_heuristic(n1, n2):
    return math.hypot(n1.x - n2.x, n1.y - n2.y)

def calc_index(node, x_width):
    return node.y * x_width + node.x

def get_motion_model():
    motion = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
              [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
              [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]
    return motion

def is_valid_node(node, x_width, y_width, obstacle_map):
    if node.x < 0 or node.x >= x_width or node.y < 0 or node.y >= y_width:
        return False
    if obstacle_map[node.x][node.y]:
        return False
    return True

def create_obstacle_map(min_x, min_y, x_width, y_width, grid_size, robot_radius, obstacle_list):
    obstacle_map = [[False for _ in range(y_width)] for _ in range(x_width)]
    padding = int(robot_radius / grid_size)
    
    for ox, oy in obstacle_list:
        ix = round((ox - min_x) / grid_size)
        iy = round((oy - min_y) / grid_size)
        
        # 로봇 반경을 고려하여 장애물 영역 확장 (Padding)
        for i in range(-padding, padding + 1):
            for j in range(-padding, padding + 1):
                if 0 <= ix + i < x_width and 0 <= iy + j < y_width:
                    obstacle_map[ix + i][iy + j] = True
    return obstacle_map

def reconstruct_path(goal_node, closed_set, grid_size, min_x, min_y):
    rx, ry = [(goal_node.x * grid_size) + min_x], [(goal_node.y * grid_size) + min_y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append((n.x * grid_size) + min_x)
        ry.append((n.y * grid_size) + min_y)
        parent_index = n.parent_index
    return rx[::-1], ry[::-1]

def a_star_search(start_xy, goal_xy, obstacle_list, grid_size, robot_radius):
    min_x, max_x = -10.0, 10.0
    min_y, max_y = -10.0, 10.0

    x_width = round((max_x - min_x) / grid_size)
    y_width = round((max_y - min_y) / grid_size)
    
    start_node = Node(round((start_xy[0] - min_x) / grid_size), round((start_xy[1] - min_y) / grid_size), 0.0, -1)
    goal_node = Node(round((goal_xy[0] - min_x) / grid_size), round((goal_xy[1] - min_y) / grid_size), 0.0, -1)
    
    obstacle_map = create_obstacle_map(min_x, min_y, x_width, y_width, grid_size, robot_radius, obstacle_list)
    motion = get_motion_model()

    open_set, closed_set = {}, {}
    pq = [] 
    
    start_id = calc_index(start_node, x_width)
    open_set[start_id] = start_node
    heapq.heappush(pq, (start_node.cost + calc_heuristic(start_node, goal_node), start_id))

    while pq:
        f_cost, current_id = heapq.heappop(pq)
        current_node = open_set[current_id]
        
        if current_node.x == goal_node.x and current_node.y == goal_node.y:
            goal_node.parent_index = current_node.parent_index
            goal_node.cost = current_node.cost
            break
        
        del open_set[current_id]
        closed_set[current_id] = current_node
        
        for move in motion:
            neighbor_x = current_node.x + move[0]
            neighbor_y = current_node.y + move[1]
            neighbor_node = Node(neighbor_x, neighbor_y, current_node.cost + move[2], current_id)
            neighbor_id = calc_index(neighbor_node, x_width)
            
            if not is_valid_node(neighbor_node, x_width, y_width, obstacle_map) or neighbor_id in closed_set:
                continue
            
            if neighbor_id in open_set:
                if open_set[neighbor_id].cost > neighbor_node.cost:
                    open_set[neighbor_id] = neighbor_node
            else:
                open_set[neighbor_id] = neighbor_node
                h_cost = calc_heuristic(neighbor_node, goal_node)
                f_cost = neighbor_node.cost + h_cost
                heapq.heappush(pq, (f_cost, neighbor_id))

    if goal_node.parent_index == -1:
        return [], []
    
    return reconstruct_path(goal_node, closed_set, grid_size, min_x, min_y)

def smooth_path(path_x, path_y, alpha=0.5, beta=0.3, iterations=100):
    new_path_x = list(path_x)
    new_path_y = list(path_y)
    
    for _ in range(iterations):
        for i in range(1, len(path_x) - 1):
            new_path_x[i] += alpha * (path_x[i] - new_path_x[i])
            new_path_y[i] += alpha * (path_y[i] - new_path_y[i])

            new_path_x[i] += beta * (new_path_x[i+1] + new_path_x[i-1] - 2.0 * new_path_x[i])
            new_path_y[i] += beta * (new_path_y[i+1] + new_path_y[i-1] - 2.0 * new_path_y[i])

    new_path_x[-1] = path_x[-1]
    new_path_y[-1] = path_y[-1]
            
    return new_path_x, new_path_y
# ----------------------------------------------------

## 📂 CSV 파일 저장 함수
def save_path_to_csv(path_x, path_y, filename="path_planning_result.csv"):
    """계산된 경로를 X, Y 좌표 열을 가진 CSV 파일로 저장합니다."""
    path_data = list(zip(path_x, path_y))
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X_coordinate (m)', 'Y_coordinate (m)'])
            writer.writerows(path_data)
        print(f"✅ 경로 데이터가 '{filename}'에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ CSV 파일 저장 중 오류 발생: {e}")


# ----------------------------------------------------
# 4. 메인 실행 함수 (사용자 입력 필드)
# ----------------------------------------------------
START_XY = [-6.0, -6.0] 
GOAL_XY = [0.0, 7.0]     
GRID_SIZE = 0.1          
ROBOT_RADIUS = 0.6     

# SDF 파일에서 계산된 장애물 점 좌표 생성
def generate_obstacle_points():
    obs_points = []
    
    # 1. 맵 경계 (X축 [-10, 10], Y축 [-10, 10])
    x_coords = np.linspace(-9.5, 9.5, num=int(20/0.5)) 
    y_coords = np.linspace(-9.5, 9.5, num=int(20/0.5))
    
    for x in x_coords:
        obs_points.append([x, 9.5]) # 북쪽 벽
        obs_points.append([x, -9.5]) # 남쪽 벽
    for y in y_coords:
        obs_points.append([9.5, y]) # 동쪽 벽
        obs_points.append([-9.5, y]) # 서쪽 벽

    # 2. 주차 구역 고정 벽 (X=[-4, -2], Y=[5.5, 8.5] 및 X=[2, 4], Y=[5.5, 8.5])
    x_left = np.linspace(-4.0, -2.0, num=int(2.0/0.2))
    y_obs = np.linspace(5.5, 8.5, num=int(3.0/0.2))
    for x in x_left:
        for y in y_obs:
            obs_points.append([x, y])

    x_right = np.linspace(2.0, 4.0, num=int(2.0/0.2))
    for x in x_right:
        for y in y_obs:
            obs_points.append([x, y])
            
    # 3. 4개의 2m 벽 (Wall A, B, C, D)
    WALL_WIDTH = 2.0
    WALL_THICKNESS = 0.1
    RESOLUTION = 0.2
    
    # Wall A: 중심 (-4.0, 0.0)
    x_wall_a = np.linspace(-3.5 - WALL_WIDTH/2, -3.5 + WALL_WIDTH/2, num=int(WALL_WIDTH/RESOLUTION))
    y_wall_a = np.linspace(0.0 - WALL_THICKNESS/2, 0.0 + WALL_THICKNESS/2, num=int(WALL_THICKNESS/RESOLUTION) + 2)
    for x in x_wall_a:
        for y in y_wall_a:
            obs_points.append([x, y])

    # Wall B: 중심 (3.0, 0.0)
    x_wall_b = np.linspace(3.0 - WALL_WIDTH/2, 3.0 + WALL_WIDTH/2, num=int(WALL_WIDTH/RESOLUTION))
    y_wall_b = np.linspace(0.0 - WALL_THICKNESS/2, 0.0 + WALL_THICKNESS/2, num=int(WALL_THICKNESS/RESOLUTION) + 2)
    for x in x_wall_b:
        for y in y_wall_b:
            obs_points.append([x, y])

    # Wall C: 중심 (-1.0, 3.0)
    x_wall_c = np.linspace(-1.0 - WALL_WIDTH/2, -1.0 + WALL_WIDTH/2, num=int(WALL_WIDTH/RESOLUTION))
    y_wall_c = np.linspace(3.0 - WALL_THICKNESS/2, 3.0 + WALL_THICKNESS/2, num=int(WALL_THICKNESS/RESOLUTION) + 2)
    for x in x_wall_c:
        for y in y_wall_c:
            obs_points.append([x, y])
            
    # Wall D: 중심 (4.0, 3.0)
    x_wall_d = np.linspace(4.0 - WALL_WIDTH/2, 4.0 + WALL_WIDTH/2, num=int(WALL_WIDTH/RESOLUTION))
    y_wall_d = np.linspace(3.0 - WALL_THICKNESS/2, 3.0 + WALL_THICKNESS/2, num=int(WALL_THICKNESS/RESOLUTION) + 2)
    for x in x_wall_d:
        for y in y_wall_d:
            obs_points.append([x, y])
    
    # 4. 20개의 랜덤 원통
    for x, y in RANDOM_OBS_COORDS:
        obs_points.append([x, y])
    
    return obs_points

OBSTACLES = generate_obstacle_points()

# A* 경로 탐색 및 평탄화 실행
raw_path_x, raw_path_y = a_star_search(START_XY, GOAL_XY, OBSTACLES, GRID_SIZE, ROBOT_RADIUS)
    
if not raw_path_x:
    print("경로를 찾을 수 없습니다! 장애물과 시작/도착 지점을 확인해주세요.")
else:
    smoothed_path_x, smoothed_path_y = smooth_path(raw_path_x, raw_path_y)
    
    # 1. CSV 파일 저장
    CSV_FILENAME = "path_planning_result.csv" 
    save_path_to_csv(smoothed_path_x, smoothed_path_y, CSV_FILENAME)

    # 2. 이미지 시각화 및 PNG 파일 저장
    plt.figure(figsize=(10, 10))
    
    # 장애물 시각화 (기존 박스 장애물: 주차 구역 좌우 벽)
    left_block = plt.Rectangle((-4, 5.5), 2.0, 3.0, fc='gray', ec='black')
    right_block = plt.Rectangle((2, 5.5), 2.0, 3.0, fc='gray', ec='black')
    plt.gca().add_patch(left_block)
    plt.gca().add_patch(right_block)
    
    # 주차 슬롯 시각화
    slot_frame = plt.Rectangle((-1.0, 5.5), 2.0, 3.0, ec='y', fill=False, linewidth=2, linestyle='--')
    plt.gca().add_patch(slot_frame)
    
    # 4개의 벽 시각화
    # Wall A: (x=-5.0, y=-0.05)에서 시작, 너비 2.0, 높이 0.1
    wall_a_vis = plt.Rectangle((-4.5, -0.05), 2.0, 0.1, fc='gray', ec='black')
    plt.gca().add_patch(wall_a_vis)

    # Wall B: (x=2.0, y=-0.05)에서 시작, 너비 2.0, 높이 0.1
    wall_b_vis = plt.Rectangle((2.0, -0.05), 2.0, 0.1, fc='gray', ec='black')
    plt.gca().add_patch(wall_b_vis)
    
    # Wall C: (x=-2.0, y=2.95)에서 시작, 너비 2.0, 높이 0.1
    wall_c_vis = plt.Rectangle((-2.0, 2.95), 2.0, 0.1, fc='gray', ec='black')
    plt.gca().add_patch(wall_c_vis)
    
    # Wall D: (x=3.0, y=2.95)에서 시작, 너비 2.0, 높이 0.1
    wall_d_vis = plt.Rectangle((3.0, 2.95), 2.0, 0.1, fc='gray', ec='black')
    plt.gca().add_patch(wall_d_vis)
    
    # 20개의 랜덤 원통형 장애물 시각화
    for cx, cy in RANDOM_OBS_COORDS:
        plt.gca().add_patch(plt.Circle((cx, cy), 0.1, fc='blue', ec='black'))
    # ------------------------------------------------

    plt.plot(raw_path_x, raw_path_y, ":g", linewidth=1, label="A* Path (Raw)")
    plt.plot(smoothed_path_x, smoothed_path_y, "-r", linewidth=2, label="Smoothed Path")
    plt.plot(START_XY[0], START_XY[1], "sb", markersize=10, label="Start")
    plt.plot(GOAL_XY[0], GOAL_XY[1], "*b", markersize=10, label="Goal")
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axis("equal")
    plt.title("TurtleBot Path Planning (A* + Smoothing) with 4 Walls and 20 Random Obstacles")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True, alpha=0.5)

    PNG_FILENAME = "path_planning_result.png" 
    plt.savefig(PNG_FILENAME)
    print(f"✅ 경로 시각화 이미지가 '{PNG_FILENAME}'에 저장되었습니다.")
    
    # plt.show()
