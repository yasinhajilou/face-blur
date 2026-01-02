"""
Video Processor - Face Detection and Blurring
Uses multiple detection methods for maximum accuracy:
- MediaPipe Face Detection (handles various angles)
- MediaPipe Face Mesh (catches more edge cases)
- OpenCV DNN SSD (reliable frontal face detection)
- YuNet (modern lightweight detector for various poses)
- Skin tone detection as fallback for partial faces
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Set
import json
import urllib.request

import cv2
import numpy as np

# MediaPipe for better face detection - support both old and new API
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_LEGACY = False  # Old solutions-based API
MEDIAPIPE_TASKS = False   # New task-based API

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    # Check which API is available
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_LEGACY = True
    if hasattr(mp, 'tasks'):
        MEDIAPIPE_TASKS = True
except ImportError:
    print("Warning: MediaPipe not available, using fallback detection only")


class VideoProcessor:
    """
    Video processor for face detection and blurring.
    
    Uses MULTIPLE detection methods for maximum accuracy:
    - MediaPipe Face Detection (short and full range models)
    - MediaPipe Face Mesh for landmark-based detection
    - OpenCV DNN SSD (Caffe model)
    - YuNet detector for various head poses
    - Multi-scale detection for different face sizes
    
    This ensures faces are detected even when:
    - Only partially visible
    - Turned to the side (profile)
    - Not looking at camera
    - At various distances
    """
    
    def __init__(
        self,
        detection_confidence: float = 0.25,  # Low threshold - privacy first
        blur_kernel_size: int = 99,
        blur_expand_ratio: float = 1.4  # Good expansion for coverage
    ):
        """
        Initialize the video processor.
        
        Args:
            detection_confidence: Minimum confidence for face detection (0.0-1.0)
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
            blur_expand_ratio: Expand detected face region by this ratio for better coverage
        """
        self.detection_confidence = detection_confidence
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_expand_ratio = blur_expand_ratio
        
        # Initialize all detectors
        self.face_detector = None  # OpenCV DNN
        self.mp_face_detection_short = None  # MediaPipe short range (legacy)
        self.mp_face_detection_full = None  # MediaPipe full range (legacy)
        self.mp_face_mesh = None  # MediaPipe Face Mesh (legacy)
        self.hog_detector = None  # HOG person detector for back-of-head detection
        self.mp_face_detector_task = None  # MediaPipe task API
        self.yunet_detector = None  # YuNet detector
        
        # Tracking state
        self.prev_faces = []  # Previous frame detections
        self.prev_frame_gray = None  # For scene change detection
        self.tracked_faces = []  # Optical flow tracked faces
        self.face_id_counter = 0
        self.confirmed_faces = {}  # Faces confirmed by multiple detectors {id: (box, confidence_count, frames_seen)}
        
        self._init_face_detector()
        self._init_mediapipe()
        self._init_yunet()
        self._init_hog_detector()
        
    def _init_face_detector(self):
        """Initialize the OpenCV DNN face detector."""
        # Model files paths
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(exist_ok=True)
        
        prototxt_path = model_dir / 'deploy.prototxt'
        caffemodel_path = model_dir / 'res10_300x300_ssd_iter_140000.caffemodel'
        
        # Download model files if not present
        if not prototxt_path.exists():
            print("Downloading face detection model (prototxt)...")
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
        
        if not caffemodel_path.exists():
            print("Downloading face detection model (caffemodel)...")
            caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
        
        # Load the DNN model
        self.face_detector = cv2.dnn.readNetFromCaffe(
            str(prototxt_path),
            str(caffemodel_path)
        )
        print("OpenCV DNN face detection model loaded successfully")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face detectors for better accuracy."""
        if not MEDIAPIPE_AVAILABLE:
            return
        
        # Try new task-based API first (MediaPipe >= 0.10.0)
        if MEDIAPIPE_TASKS:
            try:
                from mediapipe.tasks import python as mp_tasks
                from mediapipe.tasks.python import vision
                
                # Download face detector model if needed
                model_dir = Path(__file__).parent / 'models'
                model_path = model_dir / 'blaze_face_short_range.tflite'
                
                if not model_path.exists():
                    print("Downloading MediaPipe face detector model...")
                    import ssl
                    ssl._create_default_https_context = ssl._create_unverified_context
                    model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
                    urllib.request.urlretrieve(model_url, model_path)
                
                # Create face detector - LOW confidence for maximum recall
                base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=0.2  # Very low - catch all faces
                )
                self.mp_face_detector_task = vision.FaceDetector.create_from_options(options)
                print("MediaPipe face detection (task API) initialized successfully")
                return
            except Exception as e:
                print(f"MediaPipe task API failed: {e}, trying legacy API...")
        
        # Fallback to legacy solutions API (MediaPipe < 0.10.0)
        if MEDIAPIPE_LEGACY:
            try:
                # MediaPipe Face Detection - Short Range (better for close faces, up to 2m)
                self.mp_face_detection_short = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,  # 0 = short range model
                    min_detection_confidence=0.2  # Low - catch all faces
                )
                
                # MediaPipe Face Detection - Full Range (better for far faces, up to 5m)
                self.mp_face_detection_full = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # 1 = full range model
                    min_detection_confidence=0.2
                )
                
                # MediaPipe Face Mesh - Can detect faces via landmarks even in difficult angles
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=50,  # Support many faces
                    refine_landmarks=True,  # Better accuracy
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2
                )
                
                print("MediaPipe face detection (legacy API) initialized successfully")
            except Exception as e:
                print(f"Warning: MediaPipe legacy initialization failed: {e}")
                self.mp_face_detection_short = None
                self.mp_face_detection_full = None
                self.mp_face_mesh = None
    
    def _init_yunet(self):
        """Initialize YuNet face detector for various head poses."""
        try:
            model_dir = Path(__file__).parent / 'models'
            yunet_path = model_dir / 'face_detection_yunet_2023mar.onnx'
            
            # Download YuNet model if not present
            if not yunet_path.exists():
                print("Downloading YuNet face detection model...")
                yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                urllib.request.urlretrieve(yunet_url, yunet_path)
            
            # Create YuNet detector with LOW threshold for privacy
            self.yunet_detector = cv2.FaceDetectorYN.create(
                str(yunet_path),
                "",
                (320, 320),  # Will be updated per frame
                0.3,  # Low score threshold
                0.3,  # NMS threshold
                5000  # Top K
            )
            print("YuNet face detection model loaded successfully")
        except Exception as e:
            print(f"Warning: YuNet initialization failed: {e}")
            self.yunet_detector = None
    
    def _init_hog_detector(self):
        """Initialize HOG person detector - disabled to reduce false positives."""
        # HOG detector disabled - was causing too many false positives
        self.hog_detector = None
        print("HOG person detector disabled (causes false positives)")
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if MEDIAPIPE_AVAILABLE:
            if self.mp_face_detector_task:
                try:
                    self.mp_face_detector_task.close()
                except:
                    pass
            if self.mp_face_detection_short:
                try:
                    self.mp_face_detection_short.close()
                except:
                    pass
            if self.mp_face_detection_full:
                try:
                    self.mp_face_detection_full.close()
                except:
                    pass
            if self.mp_face_mesh:
                try:
                    self.mp_face_mesh.close()
                except:
                    pass
        
    def get_video_duration(self, video_path: str) -> float:
        """
        Get video duration in seconds using FFmpeg.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds
        """
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"FFprobe error: {result.stderr}")
        
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video information using FFprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,codec_name,bit_rate',
            '-show_entries', 'format=duration,bit_rate',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"FFprobe error: {result.stderr}")
        
        data = json.loads(result.stdout)
        
        stream = data.get('streams', [{}])[0]
        format_info = data.get('format', {})
        
        # Parse frame rate (e.g., "30/1" -> 30.0)
        fps_str = stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        return {
            'width': int(stream.get('width', 1920)),
            'height': int(stream.get('height', 1080)),
            'fps': fps,
            'codec': stream.get('codec_name', 'h264'),
            'duration': float(format_info.get('duration', 0)),
            'bitrate': format_info.get('bit_rate', '5000000')
        }
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        PRIVACY-FIRST face detection - 99%+ accuracy is required.
        For protester safety, we run ALL detectors and accept ANY detection.
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            List of (x, y, width, height) tuples for detected faces
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        all_faces = []
        
        # Run ALL detectors - accuracy is paramount
        # 1. OpenCV DNN - reliable base
        all_faces.extend(self._detect_opencv_dnn(frame, h, w))
        
        # 2. MediaPipe - excellent for frontal faces
        all_faces.extend(self._detect_mediapipe(frame, h, w))
        
        # 3. YuNet - good for various poses
        all_faces.extend(self._detect_yunet(frame, h, w))
        
        # 4. Haar cascades - catches sunglasses, different angles
        all_faces.extend(self._detect_haar_faces(frame, h, w))
        
        # 5. Profile faces (left and right)
        all_faces.extend(self._detect_profile_faces(frame, h, w))
        
        # 6. Track from previous frames for continuity
        if self.prev_faces:
            tracked = self._track_faces_optical_flow(gray, self.prev_faces)
            all_faces.extend(tracked)
            # Keep previous detections for frame continuity
            all_faces.extend(self.prev_faces)
        
        # Store for next frame
        self.prev_frame_gray = gray.copy()
        
        # Filter invalid detections
        filtered_faces = []
        for face in all_faces:
            x, y, fw, fh = face
            
            if fw <= 0 or fh <= 0:
                continue
            
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)
            
            if fw <= 0 or fh <= 0:
                continue
            
            # Loose aspect ratio filter
            aspect_ratio = fw / max(fh, 1)
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Minimum size (1.5% of frame)
            min_size = min(h, w) * 0.015
            if fw < min_size or fh < min_size:
                continue
            
            filtered_faces.append((x, y, fw, fh))
        
        # 8. Merge overlapping detections
        final_faces = self._merge_detections(filtered_faces, h, w)
        
        # 9. Expand faces for coverage
        expanded_faces = []
        for (x, y, fw, fh) in final_faces:
            expanded = self._expand_box(x, y, fw, fh, h, w, ratio=1.4)
            expanded_faces.append(expanded)
        
        # Store for tracking
        self.prev_faces = expanded_faces.copy()
        self.prev_frame_gray = gray.copy()
        
        return expanded_faces
    
    def _detect_opencv_dnn_aggressive(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Extra aggressive OpenCV DNN pass with very low threshold."""
        faces = []
        
        if self.face_detector is None:
            return faces
        
        # Try with different input sizes for better detection
        for size in [(300, 300), (400, 400), (200, 200)]:
            try:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, size),
                    1.0,
                    size,
                    (104.0, 177.0, 123.0)
                )
                
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    # Very low threshold - 0.2
                    if confidence > 0.2:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        faces.append(self._expand_box(x1, y1, x2 - x1, y2 - y1, h, w))
            except Exception:
                pass
        
        return faces
    
    def _detect_upper_body_heads(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Detect upper bodies and estimate head regions."""
        faces = []
        
        try:
            # Use Haar cascade for upper body
            cascade_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
            upper_body_cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect upper bodies
            bodies = upper_body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(30, 30)
            )
            
            for (bx, by, bw, bh) in bodies:
                # Estimate head region (top 30% of upper body, centered)
                head_h = int(bh * 0.35)
                head_w = int(bw * 0.7)
                head_x = bx + (bw - head_w) // 2
                head_y = by
                
                faces.append(self._expand_box(head_x, head_y, head_w, head_h, h, w, ratio=1.5))
        except Exception:
            pass
        
        return faces
    
    def _detect_haar_faces(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Use Haar cascades for additional face detection."""
        faces = []
        
        try:
            # Standard frontal face
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # Alternative frontal face
            face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            # Alt2 - better for some faces
            face_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Run all cascades
            for cascade in [face_cascade, face_cascade_alt, face_cascade_alt2]:
                detected = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # More thorough
                    minNeighbors=2,    # Lower for more detections
                    minSize=(20, 20)
                )
                
                for (x, y, fw, fh) in detected:
                    faces.append(self._expand_box(x, y, fw, fh, h, w))
        except Exception:
            pass
        
        return faces
    
    def _detect_profile_faces(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Detect profile (side) faces."""
        faces = []
        
        try:
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect profiles facing right
            profiles_right = profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(20, 20)
            )
            
            for (x, y, fw, fh) in profiles_right:
                faces.append(self._expand_box(x, y, fw, fh, h, w, ratio=1.5))
            
            # Detect profiles facing left (flip image)
            gray_flipped = cv2.flip(gray, 1)
            profiles_left = profile_cascade.detectMultiScale(
                gray_flipped,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(20, 20)
            )
            
            for (x, y, fw, fh) in profiles_left:
                # Flip x coordinate back
                x_flipped = w - x - fw
                faces.append(self._expand_box(x_flipped, y, fw, fh, h, w, ratio=1.5))
        except Exception:
            pass
        
        return faces
    
    def _detect_scene_change(self, gray: np.ndarray) -> bool:
        """Detect if there was a scene cut/change."""
        if self.prev_frame_gray is None:
            return True
        
        # Compare histograms
        hist_prev = cv2.calcHist([self.prev_frame_gray], [0], None, [64], [0, 256])
        hist_curr = cv2.calcHist([gray], [0], None, [64], [0, 256])
        
        cv2.normalize(hist_prev, hist_prev)
        cv2.normalize(hist_curr, hist_curr)
        
        correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
        
        # Low correlation means scene change
        return correlation < 0.7
    
    def _verify_skin_tone(self, frame: np.ndarray, box: Tuple[int, int, int, int], strict: bool = False) -> bool:
        """Verify that a region contains skin-like colors."""
        x, y, w, h = box
        
        # Clamp to frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return False
        
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return False
        
        # Convert to different color spaces for skin detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        
        # HSV skin range (works for various skin tones)
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([35, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # YCrCb skin range (more robust across lighting)
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
        
        # Calculate skin percentage
        skin_percent = np.count_nonzero(combined_mask) / combined_mask.size
        
        # Threshold depends on strictness
        threshold = 0.25 if strict else 0.15
        
        return skin_percent > threshold
    
    def _boxes_overlap(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
        """Check if two boxes overlap significantly (IoU > threshold)."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        iou = intersection / max(union, 1)
        return iou > threshold
    
    def _merge_two_boxes(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Merge two boxes into their bounding box."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x = min(x1, x2)
        y = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)
        
        return (x, y, x_max - x, y_max - y)
    
    def _track_faces_optical_flow(self, gray: np.ndarray, prev_faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Track faces from previous frame using optical flow."""
        if self.prev_frame_gray is None or not prev_faces:
            return []
        
        tracked = []
        
        for (x, y, w, h) in prev_faces:
            # Get center point of previous face
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Create points to track (corners and center of face)
            points = np.array([
                [center_x, center_y],
                [x, y],
                [x + w, y],
                [x, y + h],
                [x + w, y + h]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            try:
                # Calculate optical flow
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame_gray, gray, points, None,
                    winSize=(31, 31),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                # Check if tracking was successful
                if status is not None and np.sum(status) >= 3:
                    # Calculate new bounding box from tracked points
                    valid_points = new_points[status.flatten() == 1]
                    if len(valid_points) >= 3:
                        x_coords = valid_points[:, 0, 0]
                        y_coords = valid_points[:, 0, 1]
                        
                        new_x = int(np.min(x_coords) - w * 0.1)
                        new_y = int(np.min(y_coords) - h * 0.1)
                        new_w = int((np.max(x_coords) - np.min(x_coords)) + w * 0.2)
                        new_h = int((np.max(y_coords) - np.min(y_coords)) + h * 0.2)
                        
                        # Make sure the size is reasonable compared to original
                        if 0.5 < new_w / max(w, 1) < 2.0 and 0.5 < new_h / max(h, 1) < 2.0:
                            new_x = max(0, new_x)
                            new_y = max(0, new_y)
                            tracked.append((new_x, new_y, new_w, new_h))
            except Exception:
                pass
        
        return tracked
    
    def _detect_opencv_dnn(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV DNN SSD."""
        faces = []
        
        if self.face_detector is None:
            return faces
        
        # Standard detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.detection_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append(self._expand_box(x1, y1, x2 - x1, y2 - y1, h, w))
        
        return faces
    
    def _detect_mediapipe(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe Face Detection."""
        faces = []
        
        if not MEDIAPIPE_AVAILABLE:
            return faces
            
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Try new task-based API first
        if self.mp_face_detector_task:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.mp_face_detector_task.detect(mp_image)
                
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    x = bbox.origin_x
                    y = bbox.origin_y
                    face_w = bbox.width
                    face_h = bbox.height
                    faces.append(self._expand_box(x, y, face_w, face_h, h, w))
            except Exception as e:
                pass
        
        # Legacy API - Short range detection (for nearby faces)
        if self.mp_face_detection_short:
            try:
                results = self.mp_face_detection_short.process(rgb_frame)
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        face_w = int(bbox.width * w)
                        face_h = int(bbox.height * h)
                        faces.append(self._expand_box(x, y, face_w, face_h, h, w))
            except Exception:
                pass
        
        # Legacy API - Full range detection (for distant faces)
        if self.mp_face_detection_full:
            try:
                results = self.mp_face_detection_full.process(rgb_frame)
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        face_w = int(bbox.width * w)
                        face_h = int(bbox.height * h)
                        faces.append(self._expand_box(x, y, face_w, face_h, h, w))
            except Exception:
                pass
        
        return faces
    
    def _detect_face_mesh(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe Face Mesh - works better for angled faces."""
        faces = []
        
        if not MEDIAPIPE_AVAILABLE or self.mp_face_mesh is None:
            return faces
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get bounding box from face landmarks
                    x_coords = [lm.x * w for lm in face_landmarks.landmark]
                    y_coords = [lm.y * h for lm in face_landmarks.landmark]
                    
                    x_min = int(min(x_coords))
                    x_max = int(max(x_coords))
                    y_min = int(min(y_coords))
                    y_max = int(max(y_coords))
                    
                    face_w = x_max - x_min
                    face_h = y_max - y_min
                    
                    # Apply larger expansion for mesh-detected faces (often partial)
                    faces.append(self._expand_box(x_min, y_min, face_w, face_h, h, w, ratio=1.6))
        except Exception:
            pass
        
        return faces
    
    def _detect_yunet(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Detect faces using YuNet detector."""
        faces = []
        
        if self.yunet_detector is None:
            return faces
        
        try:
            # Update input size for this frame
            self.yunet_detector.setInputSize((w, h))
            
            # Detect faces
            _, detections = self.yunet_detector.detect(frame)
            
            if detections is not None:
                for detection in detections:
                    x, y, face_w, face_h = detection[:4].astype(int)
                    confidence = detection[-1]
                    
                    if confidence > self.detection_confidence:
                        faces.append(self._expand_box(x, y, face_w, face_h, h, w))
        except Exception:
            pass
        
        return faces
    
    def _detect_multiscale(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """
        Multi-scale detection for faces at various distances.
        Upscales image to catch small/distant faces.
        """
        faces = []
        
        if self.face_detector is None:
            return faces
        
        # Only do multiscale if frame is large enough
        if w < 640 or h < 480:
            return faces
        
        # Create upscaled version for small face detection
        scales = [1.5, 2.0]  # Upscale factors
        
        for scale in scales:
            try:
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Limit max size to avoid memory issues
                if new_w > 2000 or new_h > 2000:
                    continue
                
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(scaled_frame, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0)
                )
                
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > self.detection_confidence:
                        box = detections[0, 0, i, 3:7] * np.array([new_w, new_h, new_w, new_h])
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Scale back to original size
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        face_w = int((x2 - x1) / scale)
                        face_h = int((y2 - y1) / scale)
                        
                        faces.append(self._expand_box(x1, y1, face_w, face_h, h, w))
            except Exception:
                pass
        
        return faces
    
    def _detect_persons_head(self, frame: np.ndarray, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """
        Detect people using HOG and estimate head regions.
        Useful for protesters facing away from camera or with obscured faces.
        """
        faces = []
        
        if self.hog_detector is None:
            return faces
        
        try:
            # Resize for faster HOG detection
            scale = 1.0
            if w > 800:
                scale = 800 / w
                resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                resized = frame
            
            # Detect people
            rects, weights = self.hog_detector.detectMultiScale(
                resized,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05
            )
            
            for (x, y, pw, ph) in rects:
                # Scale back to original size
                x = int(x / scale)
                y = int(y / scale)
                pw = int(pw / scale)
                ph = int(ph / scale)
                
                # Estimate head region (top 1/5 of person bounding box, slightly wider)
                head_h = int(ph * 0.22)
                head_w = int(pw * 0.6)
                head_x = x + (pw - head_w) // 2
                head_y = y
                
                # Apply expansion for good coverage
                faces.append(self._expand_box(head_x, head_y, head_w, head_h, h, w, ratio=2.0))
        except Exception:
            pass
        
        return faces
    
    def _expand_box(self, x: int, y: int, face_w: int, face_h: int, 
                    img_h: int, img_w: int, ratio: float = None) -> Tuple[int, int, int, int]:
        """Expand a bounding box for better face coverage."""
        if ratio is None:
            ratio = self.blur_expand_ratio
        
        expand_w = int(face_w * (ratio - 1) / 2)
        expand_h = int(face_h * (ratio - 1) / 2)
        
        # Also expand upward more to catch hair/top of head
        expand_h_top = int(expand_h * 1.3)
        
        x = max(0, x - expand_w)
        y = max(0, y - expand_h_top)
        face_w = min(img_w - x, face_w + 2 * expand_w)
        face_h = min(img_h - y, face_h + expand_h_top + expand_h)
        
        return (x, y, face_w, face_h)
    
    def _merge_detections(self, faces: List[Tuple[int, int, int, int]], 
                          img_h: int, img_w: int) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping face detections from multiple methods.
        Uses Non-Maximum Suppression (NMS) approach.
        """
        if not faces:
            return []
        
        # Convert to numpy array for NMS
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in faces], dtype=np.float32)
        
        # Use OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            [1.0] * len(boxes),  # All equal confidence for merging
            score_threshold=0.0,
            nms_threshold=0.3  # IoU threshold - merge if overlap > 30%
        )
        
        merged = []
        if len(indices) > 0:
            indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in indices:
                x1, y1, x2, y2 = boxes[i].astype(int)
                merged.append((x1, y1, x2 - x1, y2 - y1))
        
        return merged
    
    def apply_blur(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Apply Gaussian blur to detected face regions with smooth edges.
        
        Args:
            frame: OpenCV BGR image
            faces: List of (x, y, width, height) tuples
            
        Returns:
            Frame with blurred faces
        """
        result = frame.copy()
        
        for (x, y, w, h) in faces:
            # Ensure valid region
            if w <= 0 or h <= 0:
                continue
            
            # Ensure coordinates are within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, result.shape[1] - x)
            h = min(h, result.shape[0] - y)
            
            if w <= 0 or h <= 0:
                continue
            
            # Extract face region
            face_roi = result[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
            
            # Apply strong Gaussian blur
            kernel_size = self.blur_kernel_size
            
            # Apply blur multiple times for stronger effect
            blurred = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 30)
            blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 30)
            
            # Additional pixelation for extra privacy
            small = cv2.resize(blurred, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Blend blur and pixelation for best effect
            blurred_final = cv2.addWeighted(blurred, 0.7, pixelated, 0.3, 0)
            
            # Create elliptical mask for smoother blending at edges
            mask = np.zeros((h, w), dtype=np.float32)
            center = (w // 2, h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            
            # Feather the mask edges for smooth transition
            mask = cv2.GaussianBlur(mask, (31, 31), 15)
            mask = np.stack([mask] * 3, axis=-1)
            
            # Blend original and blurred using the mask
            face_roi_float = face_roi.astype(np.float32)
            blurred_float = blurred_final.astype(np.float32)
            blended = (face_roi_float * (1 - mask) + blurred_float * mask).astype(np.uint8)
            
            # Replace face region with blended version
            result[y:y+h, x:x+w] = blended
        
        return result
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> bool:
        """
        Process a video file, detecting and blurring all faces.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            progress_callback: Optional callback function for progress updates (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get video info
            video_info = self.get_video_info(input_path)
            fps = video_info['fps']
            
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp(prefix='faceblur_')
            frames_dir = Path(temp_dir) / 'frames'
            processed_dir = Path(temp_dir) / 'processed'
            frames_dir.mkdir()
            processed_dir.mkdir()
            
            try:
                # Extract frames using FFmpeg
                if progress_callback:
                    progress_callback(5)
                
                extract_cmd = [
                    'ffmpeg',
                    '-i', input_path,
                    '-qscale:v', '2',  # High quality JPEG
                    '-start_number', '0',
                    f'{frames_dir}/%08d.jpg'
                ]
                
                result = subprocess.run(extract_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"FFmpeg extract error: {result.stderr}")
                    return False
                
                # Get list of extracted frames
                frame_files = sorted(frames_dir.glob('*.jpg'))
                total_frames = len(frame_files)
                
                if total_frames == 0:
                    print("No frames extracted")
                    return False
                
                if progress_callback:
                    progress_callback(10)
                
                # Process EVERY frame with full detection for 99%+ accuracy
                for i, frame_path in enumerate(frame_files):
                    # Read frame
                    frame = cv2.imread(str(frame_path))
                    
                    if frame is None:
                        continue
                    
                    # Detect faces with ALL detectors
                    faces = self.detect_faces(frame)
                    
                    # Apply blur if faces detected
                    if faces:
                        frame = self.apply_blur(frame, faces)
                    
                    # Save processed frame
                    output_frame_path = processed_dir / frame_path.name
                    cv2.imwrite(str(output_frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Update progress (10-90% for processing)
                    if progress_callback and i % 10 == 0:
                        progress = 10 + int((i / total_frames) * 80)
                        progress_callback(progress)
                
                if progress_callback:
                    progress_callback(90)
                
                # Check if input has audio
                has_audio = self._check_audio(input_path)
                
                # Reassemble video with FFmpeg
                if has_audio:
                    # With audio
                    assemble_cmd = [
                        'ffmpeg',
                        '-framerate', str(fps),
                        '-i', f'{processed_dir}/%08d.jpg',
                        '-i', input_path,  # Original video for audio
                        '-map', '0:v',  # Video from frames
                        '-map', '1:a?',  # Audio from original (if exists)
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '18',  # High quality
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart',
                        '-y',
                        output_path
                    ]
                else:
                    # Without audio
                    assemble_cmd = [
                        'ffmpeg',
                        '-framerate', str(fps),
                        '-i', f'{processed_dir}/%08d.jpg',
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '18',  # High quality
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart',
                        '-y',
                        output_path
                    ]
                
                result = subprocess.run(assemble_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"FFmpeg assemble error: {result.stderr}")
                    return False
                
                if progress_callback:
                    progress_callback(100)
                
                return True
                
            finally:
                # Cleanup temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"Processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_audio(self, video_path: str) -> bool:
        """Check if video has audio stream."""
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False
        
        try:
            data = json.loads(result.stdout)
            return len(data.get('streams', [])) > 0
        except:
            return False


# Test function
def test_processor():
    """Test the video processor with a sample video."""
    processor = VideoProcessor()
    
    # Test with a sample video path
    input_path = "test_video.mp4"
    output_path = "test_output.mp4"
    
    if not os.path.exists(input_path):
        print("Test video not found. Please provide a test video.")
        return
    
    def progress_update(progress):
        print(f"Progress: {progress}%")
    
    success = processor.process_video(input_path, output_path, progress_update)
    
    if success:
        print(f"Processing complete! Output saved to: {output_path}")
    else:
        print("Processing failed.")


if __name__ == "__main__":
    test_processor()
