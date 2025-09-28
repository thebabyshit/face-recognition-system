"""Face detection module using MTCNN."""

from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image
import torch


class BoundingBox:
    """Bounding box representation for detected faces."""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float = 1.0):
        """
        Initialize bounding box.
        
        Args:
            x1, y1: Top-left coordinates
            x2, y2: Bottom-right coordinates
            confidence: Detection confidence score
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
    
    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center coordinates."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_list(self) -> List[float]:
        """Convert to list format [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'confidence': self.confidence,
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'center': self.center
        }


class FaceDetector:
    """Face detector using MTCNN (Multi-task CNN)."""
    
    def __init__(
        self,
        min_face_size: int = 20,
        thresholds: List[float] = [0.6, 0.7, 0.7],
        factor: float = 0.709,
        device: str = 'cpu'
    ):
        """
        Initialize face detector.
        
        Args:
            min_face_size: Minimum face size to detect
            thresholds: Detection thresholds for P-Net, R-Net, O-Net
            factor: Scale factor for image pyramid
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.device = device
        
        # Initialize MTCNN (will be loaded when first used)
        self._mtcnn = None
    
    def _load_mtcnn(self):
        """Load MTCNN model lazily."""
        if self._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
                self._mtcnn = MTCNN(
                    min_face_size=self.min_face_size,
                    thresholds=self.thresholds,
                    factor=self.factor,
                    device=self.device,
                    keep_all=True,
                    post_process=False
                )
            except ImportError:
                raise ImportError(
                    "facenet_pytorch is required for MTCNN. "
                    "Install with: pip install facenet-pytorch"
                )
    
    def detect_faces(
        self,
        image: Union[np.ndarray, Image.Image],
        return_landmarks: bool = False
    ) -> List[BoundingBox]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            List of detected face bounding boxes
        """
        self._load_mtcnn()
        
        # Convert to PIL Image if necessary
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Detect faces
        try:
            boxes, probs, landmarks = self._mtcnn.detect(image, landmarks=True)
            
            if boxes is None:
                return []
            
            # Convert to BoundingBox objects
            face_boxes = []
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob > self.thresholds[0]:  # Filter by confidence
                    bbox = BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(prob)
                    )
                    
                    # Add landmarks if requested
                    if return_landmarks and landmarks is not None:
                        bbox.landmarks = landmarks[i]
                    
                    face_boxes.append(bbox)
            
            return face_boxes
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def align_face(
        self,
        image: Union[np.ndarray, Image.Image],
        bbox: BoundingBox,
        output_size: Tuple[int, int] = (112, 112),
        margin: float = 0.2
    ) -> Optional[np.ndarray]:
        """
        Align and crop face from image.
        
        Args:
            image: Input image
            bbox: Face bounding box
            output_size: Output image size (width, height)
            margin: Margin around face (as fraction of face size)
            
        Returns:
            Aligned face image or None if failed
        """
        try:
            # Convert to numpy array if PIL Image
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Calculate expanded bounding box with margin
            face_width = bbox.width
            face_height = bbox.height
            
            margin_x = face_width * margin
            margin_y = face_height * margin
            
            x1 = max(0, int(bbox.x1 - margin_x))
            y1 = max(0, int(bbox.y1 - margin_y))
            x2 = min(image.shape[1], int(bbox.x2 + margin_x))
            y2 = min(image.shape[0], int(bbox.y2 + margin_y))
            
            # Crop face region
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Resize to output size
            face_aligned = cv2.resize(face_crop, output_size)
            
            return face_aligned
            
        except Exception as e:
            print(f"Error in face alignment: {e}")
            return None
    
    def detect_and_align(
        self,
        image: Union[np.ndarray, Image.Image],
        output_size: Tuple[int, int] = (112, 112),
        margin: float = 0.2
    ) -> List[np.ndarray]:
        """
        Detect faces and return aligned face crops.
        
        Args:
            image: Input image
            output_size: Output image size (width, height)
            margin: Margin around face (as fraction of face size)
            
        Returns:
            List of aligned face images
        """
        # Detect faces
        face_boxes = self.detect_faces(image)
        
        # Align each detected face
        aligned_faces = []
        for bbox in face_boxes:
            aligned_face = self.align_face(image, bbox, output_size, margin)
            if aligned_face is not None:
                aligned_faces.append(aligned_face)
        
        return aligned_faces
    
    def get_largest_face(
        self,
        image: Union[np.ndarray, Image.Image],
        output_size: Tuple[int, int] = (112, 112),
        margin: float = 0.2
    ) -> Optional[np.ndarray]:
        """
        Detect faces and return the largest one (aligned).
        
        Args:
            image: Input image
            output_size: Output image size (width, height)
            margin: Margin around face (as fraction of face size)
            
        Returns:
            Aligned face image of the largest detected face, or None
        """
        # Detect faces
        face_boxes = self.detect_faces(image)
        
        if not face_boxes:
            return None
        
        # Find largest face by area
        largest_bbox = max(face_boxes, key=lambda bbox: bbox.area)
        
        # Align the largest face
        return self.align_face(image, largest_bbox, output_size, margin)


def create_face_detector(
    min_face_size: int = 20,
    device: Optional[str] = None
) -> FaceDetector:
    """
    Create a face detector with default settings.
    
    Args:
        min_face_size: Minimum face size to detect
        device: Device to use ('cpu', 'cuda', or None for auto-detect)
        
    Returns:
        Configured FaceDetector instance
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return FaceDetector(
        min_face_size=min_face_size,
        device=device
    )