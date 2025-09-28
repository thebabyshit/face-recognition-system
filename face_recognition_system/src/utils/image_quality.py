"""Image quality assessment utilities for face recognition."""

import cv2
import numpy as np
from typing import Dict, Tuple, Union
from PIL import Image


class ImageQualityAssessor:
    """Assess image quality for face recognition."""
    
    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_range: Tuple[float, float] = (50, 200),
        contrast_threshold: float = 30.0,
        min_face_size: int = 50
    ):
        """
        Initialize quality assessor.
        
        Args:
            blur_threshold: Minimum Laplacian variance for sharpness
            brightness_range: Acceptable brightness range (min, max)
            contrast_threshold: Minimum standard deviation for contrast
            min_face_size: Minimum face size in pixels
        """
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.contrast_threshold = contrast_threshold
        self.min_face_size = min_face_size
    
    def assess_sharpness(self, image: np.ndarray) -> float:
        """
        Assess image sharpness using Laplacian variance.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Sharpness score (higher is sharper)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)
    
    def assess_brightness(self, image: np.ndarray) -> float:
        """
        Assess image brightness.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Average brightness value (0-255)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.mean(gray))
    
    def assess_contrast(self, image: np.ndarray) -> float:
        """
        Assess image contrast using standard deviation.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Contrast score (higher is more contrast)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.std(gray))
    
    def assess_face_size(self, face_bbox: 'BoundingBox') -> float:
        """
        Assess face size adequacy.
        
        Args:
            face_bbox: Face bounding box
            
        Returns:
            Size score (1.0 if adequate, < 1.0 if too small)
        """
        face_area = face_bbox.area
        min_area = self.min_face_size ** 2
        
        if face_area >= min_area:
            return 1.0
        else:
            return float(face_area / min_area)
    
    def assess_pose(self, landmarks: np.ndarray) -> float:
        """
        Assess face pose using facial landmarks.
        
        Args:
            landmarks: Facial landmarks array (5 points: eyes, nose, mouth corners)
            
        Returns:
            Pose score (1.0 for frontal, lower for profile)
        """
        if landmarks is None or len(landmarks) < 5:
            return 0.5  # Unknown pose
        
        try:
            # Extract eye and nose positions
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            
            # Calculate eye distance and nose position relative to eyes
            eye_distance = np.linalg.norm(right_eye - left_eye)
            eye_center = (left_eye + right_eye) / 2
            
            # Calculate nose offset from eye center line
            nose_offset = abs(nose[0] - eye_center[0])
            nose_offset_ratio = nose_offset / eye_distance if eye_distance > 0 else 1.0
            
            # Score based on nose alignment (frontal face has nose centered)
            pose_score = max(0.0, 1.0 - nose_offset_ratio * 2)
            
            return float(pose_score)
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def assess_overall_quality(
        self,
        image: np.ndarray,
        face_bbox: 'BoundingBox' = None,
        landmarks: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Assess overall image quality for face recognition.
        
        Args:
            image: Input image
            face_bbox: Optional face bounding box
            landmarks: Optional facial landmarks
            
        Returns:
            Dictionary with quality scores and overall assessment
        """
        scores = {}
        
        # Basic image quality metrics
        scores['sharpness'] = self.assess_sharpness(image)
        scores['brightness'] = self.assess_brightness(image)
        scores['contrast'] = self.assess_contrast(image)
        
        # Normalize scores to 0-1 range
        scores['sharpness_score'] = min(1.0, scores['sharpness'] / self.blur_threshold)
        
        brightness = scores['brightness']
        if self.brightness_range[0] <= brightness <= self.brightness_range[1]:
            scores['brightness_score'] = 1.0
        else:
            # Calculate distance from acceptable range
            if brightness < self.brightness_range[0]:
                distance = self.brightness_range[0] - brightness
            else:
                distance = brightness - self.brightness_range[1]
            scores['brightness_score'] = max(0.0, 1.0 - distance / 100.0)
        
        scores['contrast_score'] = min(1.0, scores['contrast'] / self.contrast_threshold)
        
        # Face-specific metrics
        if face_bbox is not None:
            scores['face_size_score'] = self.assess_face_size(face_bbox)
        else:
            scores['face_size_score'] = 1.0  # Assume adequate if no bbox provided
        
        if landmarks is not None:
            scores['pose_score'] = self.assess_pose(landmarks)
        else:
            scores['pose_score'] = 1.0  # Assume frontal if no landmarks provided
        
        # Calculate overall quality score (weighted average)
        weights = {
            'sharpness_score': 0.25,
            'brightness_score': 0.20,
            'contrast_score': 0.20,
            'face_size_score': 0.20,
            'pose_score': 0.15
        }
        
        overall_score = sum(
            scores[metric] * weight
            for metric, weight in weights.items()
            if metric in scores
        )
        
        scores['overall_score'] = overall_score
        
        # Quality assessment
        if overall_score >= 0.8:
            scores['quality_level'] = 'excellent'
        elif overall_score >= 0.6:
            scores['quality_level'] = 'good'
        elif overall_score >= 0.4:
            scores['quality_level'] = 'fair'
        else:
            scores['quality_level'] = 'poor'
        
        return scores
    
    def is_acceptable_quality(
        self,
        image: np.ndarray,
        face_bbox: 'BoundingBox' = None,
        landmarks: np.ndarray = None,
        min_overall_score: float = 0.5
    ) -> bool:
        """
        Check if image quality is acceptable for face recognition.
        
        Args:
            image: Input image
            face_bbox: Optional face bounding box
            landmarks: Optional facial landmarks
            min_overall_score: Minimum acceptable overall score
            
        Returns:
            True if quality is acceptable, False otherwise
        """
        scores = self.assess_overall_quality(image, face_bbox, landmarks)
        return scores['overall_score'] >= min_overall_score


def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Apply basic image enhancement techniques.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    enhanced = image.copy()
    
    # Convert to LAB color space for better processing
    if len(enhanced.shape) == 3:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # For grayscale images, apply CLAHE directly
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
    
    return enhanced


def detect_motion_blur(image: np.ndarray, threshold: float = 50.0) -> Tuple[bool, float]:
    """
    Detect motion blur in image using FFT analysis.
    
    Args:
        image: Input image
        threshold: Blur detection threshold
        
    Returns:
        Tuple of (is_blurred, blur_score)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    # Calculate high frequency content
    rows, cols = gray.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create a mask for high frequencies (outer region)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, (center_col, center_row), min(rows, cols) // 4, 255, -1)
    mask = 255 - mask  # Invert to get outer region
    
    # Calculate mean magnitude in high frequency region
    high_freq_magnitude = np.mean(magnitude_spectrum[mask == 255])
    
    # Blur score (lower means more blurred)
    blur_score = float(high_freq_magnitude)
    is_blurred = blur_score < threshold
    
    return is_blurred, blur_score