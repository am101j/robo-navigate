#!/usr/bin/env python3
"""
Object Detection Node
Subscribes to /camera/image_raw and performs object detection using PyTorch on GPU.
Publishes detected objects to /camera/detections as JSON messages.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import json
import numpy as np
import io

# Optional heavy imports — wrap them so node can run in environments without them
TORCH_AVAILABLE = True
PIL_AVAILABLE = True
CVBRIDGE_AVAILABLE = True
OPENCV_AVAILABLE = True
TORCHVISION_WEIGHTS_AVAILABLE = True

try:
    from PIL import Image as PILImage
except Exception:
    PIL_AVAILABLE = False

try:
    import torch
except Exception:
    TORCH_AVAILABLE = False

try:
    import torchvision.transforms as transforms
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    # Weighs API may not be available on older torchvision versions
    try:
        from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
    except Exception:
        TORCHVISION_WEIGHTS_AVAILABLE = False
except Exception:
    TORCH_AVAILABLE = False
    TORCHVISION_WEIGHTS_AVAILABLE = False

try:
    import cv2
except Exception:
    OPENCV_AVAILABLE = False

try:
    from cv_bridge import CvBridge
except Exception:
    CVBRIDGE_AVAILABLE = False


class ObjectDetector(Node):
    """ROS2 node for object detection using PyTorch on GPU"""

    def __init__(self):
        super().__init__('object_detector')
        
        # Initialize CV bridge for image conversion (if available)
        if CVBRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            self.get_logger().warning('cv_bridge not available — will use dummy detections')

        # GPU/CPU setup
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.get_logger().info(f'Using device: {self.device}')
            if not torch.cuda.is_available():
                self.get_logger().warning('CUDA not available, using CPU. Detection will be slower.')
        else:
            self.device = None
            self.get_logger().warning('PyTorch not available — object detection will use dummy detections')

        # Load detection model if possible
        self.model = None
        self.model_loaded = False
        if TORCH_AVAILABLE and PIL_AVAILABLE and TORCHVISION_WEIGHTS_AVAILABLE:
            self.load_model()
        else:
            # If torchvision weights API not available, we may still try a simpler load in load_model
            if TORCH_AVAILABLE and PIL_AVAILABLE:
                self.load_model()

        # Image preprocessing (fallback simple transform)
        try:
            self.transform = transforms.Compose([transforms.ToTensor()]) if TORCH_AVAILABLE else None
        except Exception:
            self.transform = None
        
        # COCO class names (for Faster R-CNN)
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Detection parameters
        self.confidence_threshold = 0.5
        
        # Subscriber for camera images
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        
        # Publisher for detections
        self.publisher_ = self.create_publisher(
            String,
            '/camera/detections',
            10
        )
        
        self.get_logger().info('Object Detector Node started')
        self.get_logger().info('Subscribing to: /camera/image_raw')
        self.get_logger().info('Publishing to: /camera/detections')
        
    def load_model(self):
        """Load the object detection model"""
        if not TORCH_AVAILABLE:
            self.get_logger().warning('Cannot load model: PyTorch not available')
            self.model_loaded = False
            return

        try:
            self.get_logger().info('Loading Faster R-CNN model...')
            if TORCHVISION_WEIGHTS_AVAILABLE:
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.model = fasterrcnn_resnet50_fpn(weights=weights)
                # Use the preprocessing transforms recommended by the weights
                try:
                    self.transform = weights.transforms()
                except Exception:
                    pass
            else:
                # Fallback for older torchvision versions
                self.model = fasterrcnn_resnet50_fpn(pretrained=True)

            if self.device is not None:
                self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            self.get_logger().info('Model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            self.get_logger().warning('Using dummy detections')
            self.model_loaded = False
        finally:
            # Ensure we have some transform available if model load succeeded partially
            try:
                if self.transform is None and TORCH_AVAILABLE:
                    import torchvision.transforms as transforms
                    self.transform = transforms.Compose([transforms.ToTensor()])
            except Exception:
                self.transform = None
    
    def image_callback(self, msg):
        """Callback function for incoming camera images"""
        try:
            # Convert ROS Image message to OpenCV format if possible
            if self.bridge is not None:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            else:
                # If bridge not available, create a dummy image for detection
                width = getattr(msg, 'width', 640)
                height = getattr(msg, 'height', 480)
                cv_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Perform object detection
            detections = self.detect_objects(cv_image)
            
            # Publish detections as JSON
            if detections:
                detection_msg = String()
                detection_msg.data = json.dumps(detections)
                self.publisher_.publish(detection_msg)
                self.get_logger().debug(f'Published {len(detections)} detections')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def detect_objects(self, image):
        """
        Perform object detection on the image
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            list: List of detection dictionaries
        """
        if not self.model_loaded or not TORCH_AVAILABLE or not PIL_AVAILABLE or self.transform is None:
            # Return dummy detections if model or dependencies not available
            return self.dummy_detection(image)
        
        try:
            # Preprocess image for PyTorch
            # Convert to PIL Image then to tensor
            pil_image = PILImage.fromarray(image)
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
            
            # Process predictions
            detections = []
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            height, width = image.shape[:2]
            
            for i in range(len(boxes)):
                if scores[i] >= self.confidence_threshold:
                    box = boxes[i]
                    label_id = int(labels[i])
                    label_name = self.coco_classes[label_id] if label_id < len(self.coco_classes) else f'class_{label_id}'
                    
                    detection = {
                        'class_id': int(label_id),
                        'class_name': label_name,
                        'confidence': float(scores[i]),
                        'bbox': {
                            'x_min': float(box[0] / width),
                            'y_min': float(box[1] / height),
                            'x_max': float(box[2] / width),
                            'y_max': float(box[3] / height)
                        },
                        'bbox_pixels': {
                            'x_min': float(box[0]),
                            'y_min': float(box[1]),
                            'x_max': float(box[2]),
                            'y_max': float(box[3])
                        }
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f'Error in object detection: {str(e)}')
            return []
    
    def dummy_detection(self, image):
        """
        Generate dummy detections for testing when model is not available
        
        Args:
            image: numpy array (H, W, 3)
            
        Returns:
            list: List of dummy detection dictionaries
        """
        height, width = image.shape[:2]
        
        # Generate a few dummy detections
        dummy_detections = [
            {
                'class_id': 1,
                'class_name': 'person',
                'confidence': 0.85,
                'bbox': {
                    'x_min': 0.2,
                    'y_min': 0.3,
                    'x_max': 0.5,
                    'y_max': 0.8
                },
                'bbox_pixels': {
                    'x_min': width * 0.2,
                    'y_min': height * 0.3,
                    'x_max': width * 0.5,
                    'y_max': height * 0.8
                }
            },
            {
                'class_id': 3,
                'class_name': 'car',
                'confidence': 0.72,
                'bbox': {
                    'x_min': 0.6,
                    'y_min': 0.5,
                    'x_max': 0.9,
                    'y_max': 0.9
                },
                'bbox_pixels': {
                    'x_min': width * 0.6,
                    'y_min': height * 0.5,
                    'x_max': width * 0.9,
                    'y_max': height * 0.9
                }
            }
        ]
        
        return dummy_detections


def main(args=None):
    rclpy.init(args=args)
    
    object_detector = ObjectDetector()
    
    try:
        rclpy.spin(object_detector)
    except KeyboardInterrupt:
        pass
    finally:
        object_detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



