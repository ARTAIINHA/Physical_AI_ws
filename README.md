# 쿠팡 로봇 자율주행 알고리즘 구현 Project

## Introduction
CCTV가 도착 지점 인식 후, 최단 경로 생성 및 제어를 통해 자율 주행 및 주차 시행

## 사용 기술

VISION: DMPR-PS
- ResNet 기반 AI
- Gazebo 시뮬레이터 환경에서 직접 데이터 수집 후 데이터셋 생성해 학습(100장 * 5(증강))
- DMPR-PS github 코드 사이트: https://github.com/Teoge/DMPR-PS
- 재학습된 가중치: https://drive.google.com/file/d/1nI6WmVJe0Tws1LpY28qFPaoYQU73t0Bc/view

PLANNING: A* 알고리즘
- 장애물 회피 최적 경로 생성
  
CONTROL: Pure Pursuit
- 차량 스펙에 맞게 경로 보정 후 이동 

## Reference
[DMPR-PS](https://github.com/Teoge/DMPR-PS)
