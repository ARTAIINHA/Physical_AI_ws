#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion # ROS 2 tf_transformations 사용
import numpy as np
import math
import pandas as pd
import os
import sys

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # ----------------------------------------------------
        # 1. 제어 파라미터 설정 (Configuration)
        # ----------------------------------------------------
        
        # [⭐ Lfc 동적 조절을 위한 파라미터]
        self.K = 0.5            # Lfc 비례 상수 (Lfc = K*v + Lfc_base)
        self.Lfc_base = 0.5     # 최소 전방 주시 거리 (정지 상태 포함)
        self.Lfc_max = 2.0      # 최대 전방 주시 거리 제한
        
        self.WB = 0.160         # 터틀봇4의 축간 거리 (Wheel Base, m)

        # [⭐ 터틀봇4 스펙 및 V-Omega 연동 파라미터]
        self.TARGET_SPEED = 0.35 # 기본 선속도 (m/s). 약간 상향 조정
        self.MAX_OMEGA = 1.5     # 터틀봇4의 최대 각속도 (rad/s)
        self.MIN_SPEED = 0.1     # 감속 시 최소 속도 (m/s)
        self.K_OMEGA = 0.6       # 각속도에 따른 속도 감속 비율 (0.0 ~ 1.0)
        
        self.RATE = 30.0         # 제어 루프 주기 (Hz). 20Hz -> 30Hz로 상향 (정확도 개선)
        
        # ----------------------------------------------------
        # 2. 경로 및 상태 초기화
        # ----------------------------------------------------
        self.ref_path_x = []
        self.ref_path_y = []
        self.path_points = 0
        
        # Odometry 콜백을 통해 업데이트될 값
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_v = 0.0    # Odometry 트위스트에서 실제 속도를 사용
        
        # Lookahead Point 탐색을 위한 Index (가장 가까운 점을 추적)
        self.closest_index = 0
        self.target_index = -1
        self.is_path_loaded = False
        
        # ----------------------------------------------------
        # 3. ROS 2 통신 설정
        # ----------------------------------------------------
        self.publisher_ = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)

        # ----------------------------------------------------
        # 4. 경로 로드 및 Odometry 좌표계 변환
        # ----------------------------------------------------
        # ROS Argument에서 CSV 경로를 받아오도록 수정 (로직 상의 간결함을 위해)
        self.declare_parameter('csv_path', '/home/deoduck/route/path_planning_result.csv')
        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        self.load_path_from_csv(csv_path)

        # ----------------------------------------------------
        # 5. 시간 동기화 및 타이머 설정
        # ----------------------------------------------------
        self.timer = self.create_timer(1.0 / self.RATE, self.control_loop)
        self.get_logger().info('Pure Pursuit node upgraded. Waiting for odom data.')

        
    # ----------------------------------------------------
    # ROS Callback 및 유틸리티 함수
    # ----------------------------------------------------
        
    def odom_callback(self, msg):
        """odom 토픽으로부터 현재 위치, 방향(yaw), 속도(v) 정보를 업데이트합니다."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        quaternion = msg.pose.pose.orientation
        # ROS 2 표준 라이브러리를 사용하여 쿼터니언 변환
        _, _, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.current_yaw = yaw
        
        # Odometry 메시지에서 선속도를 직접 사용
        self.current_v = msg.twist.twist.linear.x

    def load_path_from_csv(self, filename):
        """CSV에서 경로를 읽고 Odometry 좌표계에 맞게 회전 및 이동합니다."""
        if not os.path.exists(filename):
            self.get_logger().error(f'❌ 경로 파일 없음: {filename}. 파일 경로를 확인하세요.')
            return

        try:
            df = pd.read_csv(filename)
            self.ref_path_x = df['X_coordinate (m)'].tolist()
            self.ref_path_y = df['Y_coordinate (m)'].tolist()
            
            if not self.ref_path_x:
                raise ValueError("경로 파일에 데이터가 없습니다.")

            # 1. 경로 Translation: World Frame 시작점 (-6.0, -6.0)을 Odometry Frame의 (0, 0)으로 이동
            path_start_x = self.ref_path_x[0] 
            path_start_y = self.ref_path_y[0]

            translated_x = [x - path_start_x for x in self.ref_path_x]
            translated_y = [y - path_start_y for y in self.ref_path_y]
            
            # 2. 경로 Rotation: 로봇의 시작 Yaw 각도(1.0690)만큼 경로를 역회전 (-1.0690)
            # 이 작업을 해야 경로의 시작 방향이 Odometry Frame의 0도(X축)와 일치합니다.
            initial_yaw = 1.0690 # <--- ⭐계산된 Yaw 값 1.0690을 직접 사용합니다.
            
            cos_yaw = math.cos(-initial_yaw)
            sin_yaw = math.sin(-initial_yaw)
            
            rotated_x = []
            rotated_y = []
            
            for x, y in zip(translated_x, translated_y):
                # 회전 변환 공식 적용
                rx = x * cos_yaw - y * sin_yaw
                ry = x * sin_yaw + y * cos_yaw
                rotated_x.append(rx)
                rotated_y.append(ry)
            
            self.ref_path_x = rotated_x
            self.ref_path_y = rotated_y
            
            self.path_points = len(self.ref_path_x)
            self.is_path_loaded = True
            self.get_logger().info(f'✅ 경로 {self.path_points}개 로드 및 Odom 좌표계 정렬(Rotation/Translation) 완료.')
            
        except Exception as e:
            self.get_logger().error(f'❌ 경로 로딩 중 오류 발생: {e}')
            self.ref_path_x = []
            self.ref_path_y = []
            self.is_path_loaded = False
            
    # ----------------------------------------------------
    # Pure Pursuit 핵심 로직
    # ----------------------------------------------------
    
    def calculate_Lfc(self):
        """속도에 비례하고 경로 끝에서 동적으로 감소하는 Lfc를 계산합니다."""
        # 1. 속도 비례 Lfc 계산
        Lfc_speed_proportional = self.Lfc_base + self.K * abs(self.current_v)
        
        # 2. 최대 Lfc 제한
        Lfc_clamped = min(Lfc_speed_proportional, self.Lfc_max)

        # 3. 경로 끝 감속 로직 (경로의 90% 이후부터 Lfc를 Lfc_base로 선형 감소)
        # Lookahead Point 탐색은 closest_index부터 시작하므로, closest_index를 기준으로 남은 경로를 판단
        if self.closest_index > self.path_points * 0.90:
            remaining_ratio = 1.0 - (self.closest_index / self.path_points)
            # 0.98 지점에서 Lfc_base까지 줄어들도록 설정 (0.08 구간에서 감소)
            Lfc = Lfc_clamped * min(remaining_ratio / 0.08, 1.0)
            Lfc = max(Lfc, self.Lfc_base)
            return Lfc
        
        return Lfc_clamped

    def find_target_point(self):
        """가장 가까운 경로 지점 (Closest Point)을 찾고, Lfc에 따라 목표 지점을 탐색합니다."""
        robot_x, robot_y = self.current_x, self.current_y
        min_dist = float('inf')
        closest_idx = -1
        
        # 1. 경로 전체를 탐색하여 현재 위치에서 가장 가까운 점 (Closest Point) 찾기
        for i in range(self.closest_index, self.path_points):
            dist = math.hypot(self.ref_path_x[i] - robot_x, self.ref_path_y[i] - robot_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
            # 거리가 다시 멀어지기 시작하면 (경로를 지나쳤다고 판단) 탐색 중지
            elif dist > min_dist + 0.1 and i > self.closest_index + 5:
                break
                
        if closest_idx != -1:
            self.closest_index = closest_idx
        
        current_Lfc = self.calculate_Lfc() # 동적 Lfc 사용

        # 2. Closest Point부터 Lookahead distance (Lfc)보다 멀리 있는 목표 지점 (Target Point) 찾기
        target_idx = -1
        target_dist = -1
        
        # Closest Point 이후부터 Lfc 만족하는 첫 번째 점 탐색
        for i in range(self.closest_index, self.path_points):
            dist = math.hypot(self.ref_path_x[i] - robot_x, self.ref_path_y[i] - robot_y)
            if dist > current_Lfc:
                target_idx = i
                target_dist = dist
                break
                
        # 3. 경로의 끝 추종 로직 (Lfc를 만족하는 점이 없는 경우, 마지막 점을 목표로 설정)
        if target_idx == -1 and self.closest_index < self.path_points - 1:
            target_idx = self.path_points - 1
            target_dist = math.hypot(self.ref_path_x[-1] - robot_x, self.ref_path_y[-1] - robot_y)

        self.target_index = target_idx # 상태 업데이트
        return target_idx, target_dist, current_Lfc
        
    
    def calculate_steering_angle(self, target_index, target_dist, Lfc):
        """Pure Pursuit 공식을 사용하여 각속도(Angular Velocity)를 계산합니다."""
        if target_index == -1:
            return 0.0 

        target_x = self.ref_path_x[target_index]
        target_y = self.ref_path_y[target_index]
        
        # 로봇 기준 좌표계로 목표점 변환 (제어에 필요한 상대 위치 및 각도 계산)
        angle_to_target = math.atan2(target_y - self.current_y, target_x - self.current_x)
        alpha = angle_to_target - self.current_yaw 
        
        # alpha를 [-pi, pi] 범위로 정규화
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # Pure Pursuit 공식: omega = (2 * v * sin(alpha)) / Lfc
        # [⭐ 현재 속도(self.current_v)를 사용하여 각속도 계산]
        # v가 0에 가까우면 나누기 0 오류 방지를 위해 TARGET_SPEED 사용
        v_used = max(abs(self.current_v), self.MIN_SPEED)
        
        # 곡률 (Curvature, kappa) = (2 * sin(alpha)) / Lfc
        # omega = v_used * kappa
        omega = (2.0 * v_used * math.sin(alpha)) / Lfc

        # 최대 각속도 제한 적용
        omega = max(min(omega, self.MAX_OMEGA), -self.MAX_OMEGA)

        return omega

    # ----------------------------------------------------
    # 메인 제어 루프
    # ----------------------------------------------------

    def control_loop(self):
        """주기적으로 실행되는 제어 루프입니다."""
        
        twist_stamped = TwistStamped()
        twist = twist_stamped.twist
        
        now = self.get_clock().now()
        twist_stamped.header.stamp = now.to_msg()
        twist_stamped.header.frame_id = 'base_link' 
        
        if not self.is_path_loaded:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher_.publish(twist_stamped)
            return

        target_index, target_dist, Lfc = self.find_target_point()
        
        # -----------------------------------------------------------------
        # 1. 종료 조건 확인
        # -----------------------------------------------------------------
        final_x = self.ref_path_x[-1]
        final_y = self.ref_path_y[-1]
        dist_to_final_goal = math.hypot(final_x - self.current_x, final_y - self.current_y)
        
        # 경로의 95% 이상 추종 + 최종 목표 지점과 Lfc_base 이내 근접 시 종료
        if self.closest_index >= self.path_points * 0.95 and dist_to_final_goal < self.Lfc_base * 0.5:
            self.get_logger().info('✅ 경로 추종 완료! (목표 지점 근접). 정지합니다.')
            self.get_logger().info(f'Final Distance: {dist_to_final_goal:.3f}m')
            
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher_.publish(twist_stamped)
            # 정지 후 타이머를 해제하여 루프 종료
            self.timer.cancel()
            return

        # 2. 각속도 계산
        angular_z = self.calculate_steering_angle(target_index, target_dist, Lfc)
        
        # -----------------------------------------------------------------
        # [⭐ 3. V-Omega 연동 선속도 계산 (급커브 감속)]
        # -----------------------------------------------------------------
        omega_ratio = abs(angular_z) / self.MAX_OMEGA
        # 각속도가 클수록 선속도가 감소하도록 조정: v_dynamic = TARGET_SPEED * (1 - K_OMEGA * omega_ratio)
        v_dynamic = self.TARGET_SPEED * (1.0 - self.K_OMEGA * omega_ratio)
        v_dynamic = max(v_dynamic, self.MIN_SPEED) # 최소 속도 제한

        # 목표를 찾지 못했거나 (Target Index가 마지막 점이지만 거리가 Lfc_base의 2배 이상), 
        # 최종 목표 지점 근처에서 속도를 0으로 수렴
        if target_index == -1 or dist_to_final_goal < Lfc:
            # 최종 목표 근접 시 속도 감속: (Lfc 거리를 기준으로 선형적으로 0으로 감속)
            if dist_to_final_goal < self.Lfc_base:
                 v_dynamic = self.TARGET_SPEED * (dist_to_final_goal / self.Lfc_base)
                 v_dynamic = max(v_dynamic, 0.0)
            
            # 최종 목표를 찾지 못했으나 경로 끝은 아닌 경우 (큰 이탈), 정지하지 않고 최소 속도 유지
            elif target_index == -1:
                v_dynamic = self.MIN_SPEED
        
        # 4. 명령 설정 및 발행
        twist.linear.x = v_dynamic
        twist.angular.z = angular_z
        
        self.publisher_.publish(twist_stamped)
        
        # 디버깅 정보 (옵션)
        # self.get_logger().info(
        #     f'X: {self.current_x:.2f}, Y: {self.current_y:.2f}, '
        #     f'V: {v_dynamic:.2f}, Omega: {angular_z:.2f}, Lfc: {Lfc:.2f}, '
        #     f'Closest: {self.closest_index}, Dist Goal: {dist_to_final_goal:.2f}'
        # )


def main(args=None):
    # tf_transformations 라이브러리가 명시적으로 import되도록 합니다.
    # rclpy.init은 이미 main 함수 시작 부분에 있습니다.
    
    rclpy.init(args=args)
    pure_pursuit = PurePursuit()
    try:
        rclpy.spin(pure_pursuit)
    except KeyboardInterrupt:
        pure_pursuit.get_logger().info('Keyboard Interrupt detected. Stopping robot.')
    except Exception as e:
        pure_pursuit.get_logger().error(f"Error during spin: {e}")
        
    finally:
        # 종료 시 안전하게 정지 명령 발행
        if rclpy.ok() and hasattr(pure_pursuit, 'publisher_'):
            stop_stamped = TwistStamped()
            stop_stamped.header.stamp = pure_pursuit.get_clock().now().to_msg()
            stop_stamped.twist.linear.x = 0.0
            stop_stamped.twist.angular.z = 0.0
            pure_pursuit.publisher_.publish(stop_stamped)
            
        if rclpy.ok():
            pure_pursuit.destroy_node()
            rclpy.shutdown()
            
if __name__ == '__main__':
    main()
