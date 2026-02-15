import cv2
import numpy as np

# 1) 이미지 읽기
img_path = "map.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"이미지를 못 읽었어: {img_path}")

H, W = img.shape[:2]
print("Image size (W,H) =", (W, H))

# 2) 4개 좌표 (x, y)
points = np.array([
    [48, 226],
    [103, 231],
    [40, 284],
    [107, 295]
], dtype=float)

# 3) 중점(평균)
cx, cy = points.mean(axis=0)
print("Center (pixel) =", (cx, cy))

# 4) 사진 전체 기준 변환
#    - 가로축 = y -> X
#    - 세로축 = x -> Y
#    - 좌상단(0,0) -> (10,10)
#    - 우하단(W,H) -> (-10,-10)
def convert_full_image(x, y, W, H):
    # y: 0(위) -> H(아래)  ==> X: 10 -> -10
    X = 10 - 20 * (y / H)

    # x: 0(왼) -> W(오)    ==> Y: 10 -> -10
    Y = 10 - 20 * (x / W)

    return X, Y

Xc, Yc = convert_full_image(cx, cy, W, H)
print("Center (map coord) =", (Xc, Yc))

# (선택) 확인용 시각화
debug = img.copy()
cv2.circle(debug, (int(round(cx)), int(round(cy))), 5, (0, 255, 0), -1)
cv2.imshow("debug", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()
