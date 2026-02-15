import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/overhead_camera/image',
            self.listener_callback,
            10
        )
        self.get_logger().info('Image Viewer Node started. Waiting for image data...')

    def listener_callback(self, msg):
        try:
            # mono16 인코딩을 OpenCV가 처리할 수 있도록 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            
            # 16비트 이미지를 8비트로 스케일링하여 시각화
            # (16비트 전체 범위 (0-65535)를 8비트 범위 (0-255)로 스케일)
            cv_image_8bit = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            cv2.imshow('OpenCV Image Viewer (Mono16)', cv_image_8bit)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')

def main(args=None):
    rclpy.init(args=args)
    image_viewer = ImageViewer()
    try:
        rclpy.spin(image_viewer)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    image_viewer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

