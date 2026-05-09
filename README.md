# CCTV 기반 자율 로봇 경로 탐색 파이프라인
  

> CCTV 기반 환경 인지를 활용한 자율 로봇 경로 탐색 파이프라인 구현 및 검증.  
> 실험 기간: 2025.11 ~ 12 (약 3주)

| 역할 | 이름 |
|------|------|
| 팀장(인지,판단, 제어) | 김재현 |
| 팀원(인지) | 박찬일 |
| 팀원(판단) | 송재민 |

---

---

## 프로젝트 개요

### 문제 제기
- 자율주행 알고리즘은 특정 환경에서 한 번 성공했다고 해서 실제 환경에서도 안정적으로 동작한다고 보장할 수 없음
- 실제 산업 환경(물류센터 등)은 장애물은 정적이지만, 공간 구조는 장소마다 다름
=> 자율주행 시스템에는 **다양한 정적 환경에서도 성공하는 능력**이 요구됨

### 목표
1. **CCTV 기반 환경 인지**를 통해
2. **생성된 경로를 따라** 자율 로봇이
3. **목표 지점까지 안정적으로 도달**하는 파이프라인을 구현 & 검증

---

## 시스템 파이프라인

```
[CCTV 인지] → [경로 생성 (A*)] → [로봇 제어 (Pure Pursuit)]
```
<img width="1503" height="491" alt="image" src="https://github.com/user-attachments/assets/33d64b2c-2efb-47f6-a780-ecf86175d65f" />

---

## 인지 (Perception)

### DMPR-PS 기반 주차 슬롯 검출
<img width="1474" height="384" alt="image" src="https://github.com/user-attachments/assets/7296389c-b71d-40e5-b960-9da6ff494073" />

- Pre-trained weights 사용 시 시뮬레이터 환경에서 **성능 저하 발생**
- 직접 데이터셋 생성 후 **Full Fine-tuning** 진행
  - MATLAB으로 데이터 라벨링
  - 데이터 증강 적용
  - A6000 GPU로 1000 epoch 학습

### 문제 & 해결
| 문제 | 원인 | 해결 방안 |
|------|------|-----------|
| 1000 epoch 학습 후 원하는 결과 미출력 | 사전학습 데이터와의 분포 차이 | 기존 사전학습 데이터와 유사하도록 데이터셋 변경 |
| 패치 단위 학습으로 인한 모서리 좌표 오차 | 패치 기반 추론 방식 | 오차 범위 추정 및 보정 |

### CCTV 연동
- CCTV 카메라를 RViz에 연동하기 위해 **브릿징** 진행
- 시뮬레이터와 RViz 실시간 연동
- 이미지 좌측 상단을 원점으로 설정, **네 모서리 좌표 출력**

---

## 경로 생성 (Planning)

### A* 알고리즘
<img width="473" height="475" alt="image" src="https://github.com/user-attachments/assets/d2547fb0-bd3d-4eac-bac9-e12ca1ceb1fb" />

- 시작 지점에서 목표 지점까지의 최적 경로를 탐색하는 **휴리스틱 기반** 경로 탐색 알고리즘
- `f(n) = g(n) + h(n)` 을 기준으로 탐색
  - `g(n)`: 현재까지 이동한 실제 비용
  - `h(n)`: 목표까지의 추정 비용
- 탐색 효율성과 최적성을 동시에 만족

### 경로 생성 프로세스
<img width="609" height="614" alt="image" src="https://github.com/user-attachments/assets/2b592571-f0d3-407b-83fa-a086391e45c2" />

1. 고정 장애물 2개 + 랜덤 장애물 24개로 맵 생성
2. 목표 지점 네 모서리 좌표 계산 후 평균 좌표 산출
3. 시작점 / 장애물 위치 / 도착점 좌표를 바탕으로 A*로 경로 생성

### 차체 크기 고려
- 초기 경로 생성 시 차체 크기를 고려하지 않아 **장애물 충돌 발생**
- 차량 크기를 반영한 **경로 재생성**으로 해결

---

## 제어 (Control)

### Pure Pursuit 알고리즘
<img width="465" height="326" alt="image" src="https://github.com/user-attachments/assets/d73d522c-5d35-4155-ac19-9573dcede0b8" />

- 전역 경로를 직관적으로 추종하는 **기하학적 제어 기법**
- 구조가 단순하고 계산 비용이 낮아 실시간 제어 및 반복 실험에 적합
- 차량의 운동 방정식과 경로의 지오메트리만을 사용

| 장점 | 단점 |
|------|------|
| 단순한 구조, 낮은 계산 비용 | 경로가 급격히 꺾이면 진동 & 오차 발생 |
| 부드러운 경로에서 안정적 추종 | - |

---

## 기대 효과

- **다양한 정적 환경에서의 주행 안정성 검증**  
  랜덤 배치 장애물 환경에서 실험하여 일반화 가능성 검증

- **인지–판단–제어 파이프라인의 통합 동작 검증**  
  CCTV 인지 → A* 경로 생성 → Pure Pursuit 제어로 이어지는 통합 시스템 검증

- **Top-Down 시점 기반 인지 방식의 실용성 확인**  
  구조화된 정적 환경(물류센터 등)에서의 인지 방식으로서의 실용성 평가

---

## 링크(실행 영상 포함)

- [Blog](https://blog.naver.com/deoduck92/224211977269)

---

## 개발 일정
<img width="1515" height="705" alt="image" src="https://github.com/user-attachments/assets/83a2b4e1-70c2-45c7-b268-61a05f8f2efd" />


## 참조

- [1] Huang et al., *DMPR-PS: Parking-Slot Detection Using Directional Marking-Point Regression*, IEEE T-ITS, 2021  
- [2] Dan Xiang et al., *Combined improved A and greedy algorithm for path planning of multi-objective mobile robot*, Scientific Reports, vol. 12, 2022  
- [3] 네이버 블로그 – rich0812, "주차 슬롯 검출 알고리즘 정리," 2024  
- [4] *Pure pursuit 알고리즘 기반 모바일 로봇의 경로 추종 성능 분석*, ScienceON, 2022  
- [5] Ahn et al., *Accurate Path Tracking by Adjusting Control Gains*, IEEE Access, 2021
