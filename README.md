# 쿠팡 로봇 자율주행 알고리즘 구현 Project

팀 이름: Physical AI
팀장: 김재현(12214170)
팀원: 박찬일(12214176), 송재민(12214178)

## Introduction
CCTV가 도착 지점 인식 후, 최단 경로 생성 및 제어를 통해 자율 주행 및 주차 시행

## 사용 기술
VISION: DMPR-PS
- ResNet 기반 AI
- Gazebo 시뮬레이터 환경에서 직접 데이터 수집 후 데이터셋 생성해 학습(100장 * 5(증강))
- DMPR-PS github 코드 사이트: https://github.com/Teoge/DMPR-PS
- 재학습된 가중치: https://drive.google.com/file/d/1nI6WmVJe0Tws1LpY28qFPaoYQU73t0Bc/view

PLANNING: A* 알고리즘

CONTROL: Pure Pursuit
