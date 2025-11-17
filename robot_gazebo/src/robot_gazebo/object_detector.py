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
from PIL import Image as PILImage
import io
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import cv2
from cv_bridge import CvBridge


class ObjectDetector(Node):
    """ROS2 node for object detection using PyTorch on GPU"""

    def __init__(self):
        super().__init__('object_detector')
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        if not torch.cuda.is_available():
            self.get_logger().warn('CUDA not available, using CPU. Detection will be slower.')
        
        # Load detection model
        self.model = None
        self.model_loaded = False
        self.load_model()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
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
        try:
            self.get_logger().info('Loading Faster R-CNN model...')
            # Load pre-trained Faster R-CNN model
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.model_loaded = True
            self.get_logger().info('Model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            self.get_logger().warn('Using dummy model for demonstration')
            self.model_loaded = False
    
    def image_callback(self, msg):
        """Callback function for incoming camera images"""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
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
        if not self.model_loaded:
            # Return dummy detections if model not loaded
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



