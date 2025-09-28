"""Face processing utilities and helper functions."""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
from pathlib import Path
import json


def calculate_face_landmarks_quality(landmarks: np.ndarray) -> float:
    """
    Calculate face quality based on landmark positions.
    
    Args:
        landmarks: Facial landmarks array (5 points or 68 points)
        
    Returns:
        Quality score between 0 and 1
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0
    
    try:
        # For 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        if len(landmarks) == 5:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            # Calculate eye distance
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # Calculate face symmetry
            face_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_offset = abs(nose[0] - face_center_x)
            mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
            mouth_offset = abs(mouth_center_x - face_center_x)
            
            # Normalize offsets by eye distance
            if eye_distance > 0:
                nose_symmetry = 1.0 - min(1.0, nose_offset / (eye_distance * 0.5))
                mouth_symmetry = 1.0 - min(1.0, mouth_offset / (eye_distance * 0.5))
            else:
                nose_symmetry = 0.0
                mouth_symmetry = 0.0
            
            # Calculate overall quality
            quality = (nose_symmetry + mouth_symmetry) / 2
            
        else:
            # For 68-point landmarks, use more sophisticated analysis
            quality = calculate_68_point_quality(landmarks)
        
        return float(np.clip(quality, 0.0, 1.0))
        
    except Exception:
        return 0.5  # Default quality if calculation fails


def calculate_68_point_quality(landmarks: np.ndarray) -> float:
    """Calculate quality for 68-point facial landmarks."""
    try:
        # Define landmark groups
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        nose = landmarks[27:36]
        mouth = landmarks[48:68]
        
        # Calculate eye openness
        left_eye_height = np.mean([
            np.linalg.norm(landmarks[37] - landmarks[41]),
            np.linalg.norm(landmarks[38] - landmarks[40])
        ])
        right_eye_height = np.mean([
            np.linalg.norm(landmarks[43] - landmarks[47]),
            np.linalg.norm(landmarks[44] - landmarks[46])
        ])
        
        eye_width = np.linalg.norm(landmarks[36] - landmarks[45])
        
        # Eye aspect ratio (higher is better for open eyes)
        left_ear = left_eye_height / (eye_width / 2) if eye_width > 0 else 0
        right_ear = right_eye_height / (eye_width / 2) if eye_width > 0 else 0
        avg_ear = (left_ear + right_ear) / 2
        
        # Normalize EAR to 0-1 range (typical EAR for open eyes is 0.2-0.4)
        eye_quality = min(1.0, max(0.0, (avg_ear - 0.1) / 0.3))
        
        # Calculate face symmetry
        face_center = np.mean(landmarks, axis=0)
        left_points = landmarks[:17]  # Face contour left side
        right_points = landmarks[16::-1]  # Face contour right side (reversed)
        
        # Calculate symmetry score
        symmetry_distances = []
        for i in range(min(len(left_points), len(right_points))):
            left_dist = np.linalg.norm(left_points[i] - face_center)
            right_dist = np.linalg.norm(right_points[i] - face_center)
            if left_dist > 0 and right_dist > 0:
                symmetry = 1.0 - abs(left_dist - right_dist) / max(left_dist, right_dist)
                symmetry_distances.append(symmetry)
        
        symmetry_quality = np.mean(symmetry_distances) if symmetry_distances else 0.5
        
        # Combine qualities
        overall_quality = (eye_quality * 0.6 + symmetry_quality * 0.4)
        
        return overall_quality
        
    except Exception:
        return 0.5


def crop_face_with_padding(
    image: np.ndarray,
    bbox: 'BoundingBox',
    padding_ratio: float = 0.3,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Crop face from image with padding.
    
    Args:
        image: Input image
        bbox: Face bounding box
        padding_ratio: Padding ratio relative to face size
        target_size: Target size for output (width, height)
        
    Returns:
        Cropped face image
    """
    h, w = image.shape[:2]
    
    # Calculate padding
    face_w = bbox.width
    face_h = bbox.height
    pad_w = int(face_w * padding_ratio)
    pad_h = int(face_h * padding_ratio)
    
    # Calculate crop coordinates
    x1 = max(0, int(bbox.x1 - pad_w))
    y1 = max(0, int(bbox.y1 - pad_h))
    x2 = min(w, int(bbox.x2 + pad_w))
    y2 = min(h, int(bbox.y2 + pad_h))
    
    # Crop image
    cropped = image[y1:y2, x1:x2]
    
    # Resize if target size specified
    if target_size is not None and cropped.size > 0:
        cropped = cv2.resize(cropped, target_size)
    
    return cropped


def calculate_face_angle(landmarks: np.ndarray) -> float:
    """
    Calculate face rotation angle from landmarks.
    
    Args:
        landmarks: Facial landmarks (at least eye positions)
        
    Returns:
        Rotation angle in degrees
    """
    if landmarks is None or len(landmarks) < 2:
        return 0.0
    
    try:
        # Use eye positions to calculate angle
        left_eye = landmarks[0] if len(landmarks) >= 5 else landmarks[36:42].mean(axis=0)
        right_eye = landmarks[1] if len(landmarks) >= 5 else landmarks[42:48].mean(axis=0)
        
        # Calculate angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        
        angle = np.arctan2(dy, dx) * 180.0 / np.pi
        
        return float(angle)
        
    except Exception:
        return 0.0


def rotate_face(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Rotate face image to correct orientation.
    
    Args:
        image: Input face image
        angle: Rotation angle in degrees
        center: Rotation center (default: image center)
        
    Returns:
        Rotated image
    """
    if abs(angle) < 1.0:  # Skip rotation for small angles
        return image
    
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Rotate image
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    
    return rotated


def extract_face_patches(
    image: np.ndarray,
    landmarks: np.ndarray,
    patch_size: int = 32
) -> dict:
    """
    Extract facial patches (eyes, nose, mouth) from face image.
    
    Args:
        image: Face image
        landmarks: Facial landmarks
        patch_size: Size of extracted patches
        
    Returns:
        Dictionary with facial patches
    """
    patches = {}
    
    if landmarks is None or len(landmarks) < 5:
        return patches
    
    try:
        h, w = image.shape[:2]
        half_size = patch_size // 2
        
        # Define patch locations
        if len(landmarks) == 5:
            patch_locations = {
                'left_eye': landmarks[0],
                'right_eye': landmarks[1],
                'nose': landmarks[2],
                'mouth': (landmarks[3] + landmarks[4]) / 2  # Center of mouth
            }
        else:
            # For 68-point landmarks
            patch_locations = {
                'left_eye': landmarks[36:42].mean(axis=0),
                'right_eye': landmarks[42:48].mean(axis=0),
                'nose': landmarks[30],  # Nose tip
                'mouth': landmarks[48:68].mean(axis=0)
            }
        
        # Extract patches
        for name, center in patch_locations.items():
            x, y = int(center[0]), int(center[1])
            
            # Calculate patch boundaries
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(w, x + half_size)
            y2 = min(h, y + half_size)
            
            # Extract patch
            if x2 > x1 and y2 > y1:
                patch = image[y1:y2, x1:x2]
                
                # Resize to target size if needed
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    patch = cv2.resize(patch, (patch_size, patch_size))
                
                patches[name] = patch
        
        return patches
        
    except Exception:
        return patches


def save_face_detection_results(
    results: List[dict],
    output_path: Union[str, Path],
    include_images: bool = False
) -> None:
    """
    Save face detection results to JSON file.
    
    Args:
        results: List of detection results
        output_path: Output file path
        include_images: Whether to include base64-encoded images
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for JSON serialization
    serializable_results = []
    
    for result in results:
        serializable_result = {}
        
        for key, value in result.items():
            if key == 'image' and not include_images:
                continue  # Skip image data
            elif isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            elif hasattr(value, 'to_dict'):
                serializable_result[key] = value.to_dict()
            else:
                serializable_result[key] = value
        
        serializable_results.append(serializable_result)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_face_detection_results(input_path: Union[str, Path]) -> List[dict]:
    """
    Load face detection results from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        List of detection results
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    # Convert lists back to numpy arrays where appropriate
    for result in results:
        for key, value in result.items():
            if key in ['landmarks', 'features'] and isinstance(value, list):
                result[key] = np.array(value)
    
    return results


def batch_process_faces(
    image_paths: List[Union[str, Path]],
    detector,
    output_dir: Union[str, Path],
    target_size: Tuple[int, int] = (112, 112),
    save_crops: bool = True,
    save_results: bool = True
) -> List[dict]:
    """
    Batch process multiple face images.
    
    Args:
        image_paths: List of image file paths
        detector: Face detector instance
        output_dir: Output directory
        target_size: Target size for face crops
        save_crops: Whether to save cropped faces
        save_results: Whether to save detection results
        
    Returns:
        List of processing results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_crops:
        crops_dir = output_dir / 'crops'
        crops_dir.mkdir(exist_ok=True)
    
    results = []
    
    for i, image_path in enumerate(image_paths):
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # Detect faces
            face_boxes = detector.detect_faces(image)
            
            result = {
                'image_path': str(image_path),
                'num_faces': len(face_boxes),
                'faces': []
            }
            
            # Process each detected face
            for j, bbox in enumerate(face_boxes):
                face_info = {
                    'bbox': bbox.to_dict(),
                    'confidence': bbox.confidence
                }
                
                # Crop and save face if requested
                if save_crops:
                    face_crop = crop_face_with_padding(image, bbox, target_size=target_size)
                    if face_crop.size > 0:
                        crop_filename = f"{Path(image_path).stem}_face_{j}.jpg"
                        crop_path = crops_dir / crop_filename
                        cv2.imwrite(str(crop_path), face_crop)
                        face_info['crop_path'] = str(crop_path)
                
                result['faces'].append(face_info)
            
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save results if requested
    if save_results:
        results_path = output_dir / 'detection_results.json'
        save_face_detection_results(results, results_path)
    
    return results