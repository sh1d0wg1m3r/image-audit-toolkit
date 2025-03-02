# Standard library imports
import json
import os
import re
import threading
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import chardet
import cv2
import numpy as np
import piexif
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageChops, ImageEnhance, ImageTk, UnidentifiedImageError
from scipy.fftpack import dct, idct
from tkinter import (
    Tk, Label, Button, Menu, filedialog, messagebox, Toplevel,
    Text, Scrollbar, RIGHT, Y, END, BOTH, YES, BOTTOM, Frame, Listbox, SINGLE
)

# Constants
ZOOM_FACTOR_INCREMENT = 1.25
DEFAULT_ZOOM_FACTOR = 1.0
ELA_QUALITY = 90
CLONE_DETECTION_BLOCK_SIZE = 32
CLONE_DETECTION_THRESHOLD = 0.9
DCT_BLOCK_SIZE = 8
DCT_THRESHOLD = 10
NOISE_THRESHOLD = 5
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')


class ProgressDialog:
    """Dialog showing progress for long-running operations."""
    
    def __init__(self, parent: Tk, title: str, message: str) -> None:
        """
        Initialize the progress dialog.
        
        Args:
            parent: The parent Tkinter widget
            title: Title of the dialog
            message: Message to display
        """
        self.top = Toplevel(parent)
        self.top.title(title)
        self.top.geometry("300x100")
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center the dialog
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        x = parent_x + (parent_width // 2) - 150
        y = parent_y + (parent_height // 2) - 50
        
        self.top.geometry(f"+{x}+{y}")
        
        # Add message
        Label(self.top, text=message, padx=20, pady=10).pack()
        
        # Add progress indicator
        self.progress_frame = Frame(self.top)
        self.progress_frame.pack(padx=20, pady=10, fill="x")
        
        self.progress_label = Label(self.progress_frame, text="Processing...")
        self.progress_label.pack()
    
    def update_message(self, message: str) -> None:
        """Update the progress message."""
        self.progress_label.config(text=message)
        self.top.update_idletasks()
    
    def close(self) -> None:
        """Close the dialog."""
        self.top.destroy()


class ButtonFrame(Frame):
    """Frame containing navigation and zoom buttons."""
    
    def __init__(self, root: Tk, toolkit) -> None:
        """
        Initialize the button frame.
        
        Args:
            root: The parent Tkinter widget
            toolkit: Reference to the main application for callbacks
        """
        super().__init__(root)
        self.pack(side=BOTTOM, fill='x')

        # Open folder button
        self.open_folder_btn = Button(self, text="Open Folder", command=toolkit.open_folder)
        self.open_folder_btn.pack(side='left', padx=5, pady=5)

        # Open single image button
        self.open_image_btn = Button(self, text="Open Image", command=toolkit.open_image)
        self.open_image_btn.pack(side='left', padx=5, pady=5)

        # Previous image button
        self.prev_btn = Button(self, text="<< Previous", command=toolkit.previous_image)
        self.prev_btn.pack(side='left', padx=5, pady=5)

        # Next image button
        self.next_btn = Button(self, text="Next >>", command=toolkit.next_image)
        self.next_btn.pack(side='left', padx=5, pady=5)

        # Zoom in button
        self.zoom_in_btn = Button(self, text="Zoom In", command=toolkit.zoom_in)
        self.zoom_in_btn.pack(side='right', padx=5, pady=5)

        # Zoom out button
        self.zoom_out_btn = Button(self, text="Zoom Out", command=toolkit.zoom_out)
        self.zoom_out_btn.pack(side='right', padx=5, pady=5)


class ImageAnalyzer:
    """Class for image analysis methods."""
    
    @staticmethod
    def dct_block_analysis(block: np.ndarray) -> float:
        """
        Analyze a block using DCT for artifacts.
        
        Args:
            block: Image block to analyze
            
        Returns:
            float: Energy value from high frequency components
        """
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
        return high_freq_energy
    
    @staticmethod
    def noise_analysis(image: np.ndarray, block_size: int = DCT_BLOCK_SIZE) -> np.ndarray:
        """
        Estimate noise levels across the image.
        
        Args:
            image: Input grayscale image
            block_size: Size of blocks for analysis
            
        Returns:
            np.ndarray: Map of noise levels
        """
        noise_map = np.zeros_like(image, dtype=np.float32)
        height, width = image.shape
        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block = image[y:y+block_size, x:x+block_size]
                noise_map[y:y+block_size, x:x+block_size] = np.std(block)
        return noise_map
    
    @staticmethod
    def detect_manipulated_regions(
        image_path: str, 
        block_size: int = DCT_BLOCK_SIZE, 
        dct_threshold: int = DCT_THRESHOLD, 
        noise_threshold: int = NOISE_THRESHOLD
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Identify manipulated areas using DCT and noise analysis.
        
        Args:
            image_path: Path to the image file
            block_size: Size of blocks for analysis
            dct_threshold: Threshold for DCT energy
            noise_threshold: Threshold for noise detection
            
        Returns:
            Tuple containing:
                - artifact_thresh: Binary map of potential artifacts
                - dct_norm: Normalized DCT energy map
                - noise_norm: Normalized noise map
                
        Raises:
            ValueError: If image cannot be loaded
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dct_map = np.zeros_like(gray, dtype=np.float32)
        noise_map = ImageAnalyzer.noise_analysis(gray, block_size)

        height, width = gray.shape
        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                dct_energy = ImageAnalyzer.dct_block_analysis(block)
                dct_map[y:y+block_size, x:x+block_size] = dct_energy

        dct_norm = cv2.normalize(dct_map, None, 0, 255, cv2.NORM_MINMAX)
        noise_norm = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX)

        artifact_map = cv2.addWeighted(dct_norm, 0.5, noise_norm, 0.5, 0)
        _, artifact_thresh = cv2.threshold(artifact_map, 127, 255, cv2.THRESH_BINARY)

        return artifact_thresh, dct_norm, noise_norm
    
    @staticmethod
    def error_level_analysis(image_path: str, quality: int = ELA_QUALITY) -> Optional[Image.Image]:
        """
        Perform Error Level Analysis to spot inconsistencies.
        
        Args:
            image_path: Path to the image file
            quality: JPEG compression quality
            
        Returns:
            PIL Image with ELA results or None if error occurs
            
        Raises:
            Exception: If ELA processing fails
        """
        try:
            original = Image.open(image_path).convert('RGB')
            temp_path = "temp_ela.jpg"
            original.save(temp_path, "JPEG", quality=quality)
            temp = Image.open(temp_path).convert('RGB')
            ela = ImageChops.difference(original, temp)

            # Enhance the differences
            extrema = ela.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            scale = 255.0 / max_diff if max_diff != 0 else 1
            ela = ImageEnhance.Brightness(ela).enhance(scale)

            os.remove(temp_path)
            return ela
        except Exception as e:
            raise Exception(f"ELA Error: {e}")
    
    @staticmethod
    def detect_clones_optimized(
        image_path: str, 
        block_size: int = CLONE_DETECTION_BLOCK_SIZE, 
        threshold: float = CLONE_DETECTION_THRESHOLD
    ) -> Optional[np.ndarray]:
        """
        Find duplicated regions within the image using a more efficient algorithm.
        
        Args:
            image_path: Path to the image file
            block_size: Size of blocks for analysis
            threshold: Similarity threshold for clone detection
            
        Returns:
            np.ndarray: Image with highlighted cloned regions or None if error occurs
            
        Raises:
            Exception: If clone detection fails
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Use a hash table for block matching
            block_dict = {}
            clone_mask = np.zeros_like(gray)

            # Extract and hash blocks
            for y in range(0, height - block_size + 1, block_size // 2):  # Use overlapping blocks
                for x in range(0, width - block_size + 1, block_size // 2):
                    block = gray[y:y+block_size, x:x+block_size]
                    # Use a simple hash of the block data
                    block_hash = hash(block.tobytes())
                    
                    # Check if similar block exists
                    if block_hash in block_dict:
                        prev_y, prev_x = block_dict[block_hash]
                        
                        # Verify similarity with the actual block data
                        prev_block = gray[prev_y:prev_y+block_size, prev_x:prev_x+block_size]
                        similarity = np.corrcoef(block.flatten(), prev_block.flatten())[0, 1]
                        
                        if similarity > threshold:
                            clone_mask[y:y+block_size, x:x+block_size] = 255
                            clone_mask[prev_y:prev_y+block_size, prev_x:prev_x+block_size] = 255
                    else:
                        block_dict[block_hash] = (y, x)

            # Highlight clone areas in the original image
            clone_mask_bgr = cv2.cvtColor(clone_mask, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(image, 0.7, clone_mask_bgr, 0.3, 0)

            return result
        except Exception as e:
            raise Exception(f"Clone Detection Error: {e}")
    
    @staticmethod
    def analyze_histogram(image_path: str) -> List[str]:
        """
        Check histograms for unusual patterns.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of anomaly descriptions
            
        Raises:
            Exception: If histogram analysis fails
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            anomalies = []
            colors = ('b', 'g', 'r')  # OpenCV uses BGR
            histograms = []
            
            # Calculate histograms for each channel
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms.append(hist)
                
                # Detect anomalies
                channel = color.upper()
                max_freq = np.max(hist)
                min_freq = np.min(hist)
                
                # Check for high spikes
                if max_freq > 10000:
                    anomalies.append(f"High spike in {channel} channel (Max: {int(max_freq)}).")
                
                # Check for gaps
                if min_freq == 0:
                    zero_count = np.sum(hist == 0)
                    if zero_count > 10:  # More than 10 bins with zero values
                        anomalies.append(f"Multiple gaps detected in {channel} channel ({zero_count} empty bins).")
                
                # Check for comb pattern (indicators of processing)
                diffs = np.diff(hist.flatten())
                sign_changes = np.sum(np.diff(np.signbit(diffs)))
                if sign_changes > 100:  # High number of oscillations
                    anomalies.append(f"Comb pattern detected in {channel} channel (possible processing).")

            return anomalies
        except Exception as e:
            raise Exception(f"Histogram Analysis Error: {e}")


class CameraMatcher:
    """Class for camera matching functionality."""
    
    @staticmethod
    def get_image_resolution(image_path: str) -> Optional[Tuple[int, int]]:
        """
        Get the image's width and height.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (width, height) or None if error occurs
        """
        try:
            img = Image.open(image_path)
            return img.size
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            return None
        except Exception as e:
            print(f"Error getting resolution for {image_path}: {e}")
            return None
    
    @staticmethod
    def extract_resolutions_from_yaml(yaml_data: Dict) -> List[str]:
        """
        Pull resolution details from YAML data.
        
        Args:
            yaml_data: Dictionary containing YAML data
            
        Returns:
            List of resolution strings
        """
        resolutions = []
        fields = ['MaxResolution', 'OtherResolutions', 'Resolutions', 'JPEGQualityLevels', 'VideographyNotes', 'Modes']

        for field in fields:
            if field in yaml_data:
                value = yaml_data[field]
                if isinstance(value, str):
                    resolutions.append(value)
                elif isinstance(value, list):
                    resolutions.extend(value)
                elif isinstance(value, dict):
                    for mode_val in value.values():
                        if isinstance(mode_val, str):
                            resolutions.append(mode_val)

        cleaned = []
        for res_str in resolutions:
            if not res_str:
                continue
            split_res = re.split(r'[,;\n]+', res_str)
            for res in split_res:
                res = res.strip()
                if res:
                    cleaned.append(res)
        return cleaned
    
    @staticmethod
    def resolution_string_to_tuple(res_string: str) -> Optional[Tuple[int, int]]:
        """
        Convert resolution string to (width, height).
        
        Args:
            res_string: String containing resolution (e.g., "1920x1080")
            
        Returns:
            Tuple of (width, height) or None if parsing fails
        """
        match = re.match(r'(\d+)\s*x\s*(\d+)', res_string, re.IGNORECASE)
        if match:
            try:
                width = int(match.group(1))
                height = int(match.group(2))
                return width, height
            except ValueError:
                return None
        return None
    
    @staticmethod
    def find_matching_cameras(
        image_resolution_tuple: Tuple[int, int], 
        dataset_folder: str = "dataset",
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Find cameras matching the image resolution.
        
        Args:
            image_resolution_tuple: Tuple of (width, height)
            dataset_folder: Path to the camera dataset folder
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of dictionaries with matching camera information
        """
        matches = []
        if not os.path.isdir(dataset_folder):
            print(f"Dataset folder '{dataset_folder}' not found.")
            return matches

        yaml_files = [f for f in os.listdir(dataset_folder) if f.endswith((".yaml", ".yml"))]
        total_files = len(yaml_files)
        
        for i, filename in enumerate(yaml_files):
            # Update progress
            if progress_callback:
                progress_callback(f"Processing file {i+1} of {total_files}: {filename}")
                
            filepath = os.path.join(dataset_folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] if result['encoding'] else 'utf-8'

                decoded_data = raw_data.decode(encoding, errors='replace')
                yaml_content = yaml.safe_load(decoded_data)

            except Exception as e:
                print(f"Error reading '{filepath}': {e}")
                continue

            if yaml_content and 'Specs' in yaml_content:
                specs = yaml_content['Specs']
                if 'ReviewData' in yaml_content and yaml_content['ReviewData']:
                    specs.update(yaml_content['ReviewData'])

                extracted = CameraMatcher.extract_resolutions_from_yaml(specs)

                for res_str in extracted:
                    yaml_res = CameraMatcher.resolution_string_to_tuple(res_str)
                    if yaml_res and yaml_res == image_resolution_tuple:
                        camera_info = {
                            'Name': yaml_content.get('Name', 'N/A'),
                            'ProductCode': yaml_content.get('ProductCode', 'N/A'),
                            'URL': yaml_content.get('URL', 'N/A'),
                            'ImageURL': yaml_content.get('ImageURL', 'N/A'),
                            'Award': yaml_content.get('Award', 'N/A'),
                            'ShortSpecs': yaml_content.get('ShortSpecs', []),
                            'Specs': specs
                        }
                        matches.append(camera_info)
                        break  # Stop after finding a match

        return matches


class ReportGenerator:
    """Class for generating analysis reports."""
    
    @staticmethod
    def generate_report(
        image_path: str, 
        analyzer: ImageAnalyzer, 
        camera_matcher: CameraMatcher,
        dataset_folder: str = "dataset",
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Create a comprehensive report for the image.
        
        Args:
            image_path: Path to the image file
            analyzer: ImageAnalyzer instance
            camera_matcher: CameraMatcher instance
            dataset_folder: Path to the camera dataset folder
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing the report data
            
        Raises:
            Exception: If report generation fails
        """
        try:
            report = {}
            
            # Basic image information
            if progress_callback:
                progress_callback("Getting basic image information...")
                
            image = Image.open(image_path)
            report['File Name'] = os.path.basename(image_path)
            report['File Path'] = image_path
            report['Format'] = image.format
            report['Size'] = image.size
            report['Mode'] = image.mode

            # EXIF Data
            if progress_callback:
                progress_callback("Extracting EXIF data...")
                
            try:
                exif_dict = piexif.load(image_path)
                if exif_dict and any(exif_dict[ifd] for ifd in exif_dict):
                    readable_exif = {}
                    for ifd in exif_dict:
                        if exif_dict[ifd]:
                            readable_exif[ifd] = {}
                            for tag, value in exif_dict[ifd].items():
                                tag_name = piexif.TAGS[ifd][tag]["name"]
                                try:
                                    if isinstance(value, bytes):
                                        value = value.decode('utf-8', errors='ignore')
                                    readable_exif[tag_name] = value
                                except:
                                    readable_exif[tag_name] = value
                    report['EXIF Data'] = readable_exif
                else:
                    report['EXIF Data'] = "No EXIF data found."
            except piexif.InvalidImageDataError:
                report['EXIF Data'] = "No EXIF data found."
            except Exception as e:
                report['EXIF Data'] = f"Error retrieving EXIF: {e}"

            # Basic Checks
            if progress_callback:
                progress_callback("Performing basic checks...")
                
            manipulations = []
            if 'EXIF Data' in report and report['EXIF Data'] != "No EXIF data found.":
                manipulations.append("EXIF data present.")
            else:
                manipulations.append("No EXIF data. Metadata might be stripped or altered.")

            format = image.format
            if format not in ['JPEG', 'PNG', 'GIF', 'BMP', 'TIFF', 'WEBP']:
                manipulations.append(f"Uncommon format: {format}")

            width, height = image.size
            if width > 10000 or height > 10000:
                manipulations.append(f"Very large dimensions: {width}x{height} pixels.")

            report['Basic Checks'] = manipulations if manipulations else ["No obvious issues detected."]

            # Advanced Detection
            if progress_callback:
                progress_callback("Running manipulation detection...")
                
            try:
                artifact_map, dct_map, noise_map = analyzer.detect_manipulated_regions(image_path)
                artifact_energy = float(np.sum(artifact_map) / 255)
                report['Advanced Detection'] = {
                    "Artifact Energy Sum (White Pixels)": artifact_energy
                }
            except Exception as e:
                report['Advanced Detection'] = f"Failed: {e}"

            # ELA
            if progress_callback:
                progress_callback("Performing Error Level Analysis...")
                
            try:
                ela_image = analyzer.error_level_analysis(image_path)
                if ela_image:
                    ela_path = "temp_report_ela.jpg"
                    ela_image.save(ela_path, "JPEG", quality=90)
                    report['ELA'] = f"ELA image saved at: {os.path.abspath(ela_path)}"
            except Exception as e:
                report['ELA'] = f"Failed: {e}"

            # Clone Detection
            if progress_callback:
                progress_callback("Running clone detection...")
                
            try:
                clone_image = analyzer.detect_clones_optimized(image_path)
                if clone_image is not None:
                    clone_path = "temp_report_clone.jpg"
                    cv2.imwrite(clone_path, clone_image)
                    clone_energy = float(np.sum(cv2.cvtColor(clone_image, cv2.COLOR_BGR2GRAY)) / 255)
                    report['Clone Detection'] = {
                        "Clone Energy Sum (White Pixels)": clone_energy,
                        "Clone Image Path": os.path.abspath(clone_path)
                    }
            except Exception as e:
                report['Clone Detection'] = f"Failed: {e}"

            # Histogram Analysis
            if progress_callback:
                progress_callback("Analyzing image histogram...")
                
            try:
                anomalies = analyzer.analyze_histogram(image_path)
                report['Histogram Analysis'] = anomalies if anomalies else ["No anomalies detected."]
            except Exception as e:
                report['Histogram Analysis'] = f"Failed: {e}"

            # Camera Matching
            if progress_callback:
                progress_callback("Matching camera models...")
                
            try:
                resolution = camera_matcher.get_image_resolution(image_path)
                if resolution:
                    # Use a separate callback for the camera matcher
                    def camera_progress(msg):
                        if progress_callback:
                            progress_callback(f"Camera matching: {msg}")
                            
                    matches = camera_matcher.find_matching_cameras(
                        resolution, 
                        dataset_folder,
                        progress_callback=camera_progress
                    )
                    if matches:
                        report['Matched Cameras'] = matches
                    else:
                        report['Matched Cameras'] = "No matching cameras found."
                else:
                    report['Matched Cameras'] = "Could not determine image resolution."
            except Exception as e:
                report['Matched Cameras'] = f"Error during camera matching: {e}"

            return report
        except Exception as e:
            raise Exception(f"Report generation failed: {e}")


class ImageAuditToolkit:
    """Main application class for the Image Audit Toolkit."""
    
    def __init__(self, root: Tk) -> None:
        """
        Initialize the main application window.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Image Audit Toolkit")
        self.root.geometry("1000x700")

        # Setup variables
        self.image_list: List[str] = []
        self.current_index: int = -1
        self.photo = None
        self.zoom_factor: float = DEFAULT_ZOOM_FACTOR

        # Image display area
        self.image_label = Label(root, text="No image loaded.", bg="grey", fg="white")
        self.image_label.pack(expand=YES, fill=BOTH)

        # Initialize components
        self.analyzer = ImageAnalyzer()
        self.camera_matcher = CameraMatcher()
        
        # Set dataset folder
        self.dataset_folder = "dataset"
        
        # Initialize UI
        self.init_menu()
        self.init_buttons()
        self.bind_keyboard_events()

    def init_menu(self) -> None:
        """Create the menu bar."""
        menu_bar = Menu(self.root)
        self.root.config(menu=menu_bar)

        # File options
        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder", command=self.open_folder)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Audit options
        audit_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Audit", menu=audit_menu)
        audit_menu.add_command(label="View EXIF Data", command=self.view_exif_data)
        audit_menu.add_command(label="Check Manipulation", command=self.check_manipulation)
        audit_menu.add_separator()
        audit_menu.add_command(label="Error Level Analysis (ELA)", command=self.run_ela)
        audit_menu.add_command(label="Clone Detection", command=self.run_clone_detection)
        audit_menu.add_command(label="Histogram Analysis", command=self.run_histogram_analysis)
        audit_menu.add_separator()
        audit_menu.add_command(label="Generate Report", command=self.generate_report)

        # Dataset options
        dataset_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Dataset", menu=dataset_menu)
        dataset_menu.add_command(label="Find Matching Cameras", command=self.find_matching_cameras_ui)

    def init_buttons(self) -> None:
        """Add navigation and zoom buttons."""
        ButtonFrame(self.root, self)

    def bind_keyboard_events(self) -> None:
        """Link keyboard shortcuts to actions."""
        self.root.bind('<Left>', lambda event: self.previous_image())
        self.root.bind('<Right>', lambda event: self.next_image())
        self.root.bind('<plus>', lambda event: self.zoom_in())
        self.root.bind('<KP_Add>', lambda event: self.zoom_in())
        self.root.bind('<minus>', lambda event: self.zoom_out())
        self.root.bind('<KP_Subtract>', lambda event: self.zoom_out())
        self.root.bind('<Escape>', lambda event: self.root.quit())

    # ========================= Image Viewing =========================

    def open_folder(self) -> None:
        """Select a folder and load images."""
        directory = filedialog.askdirectory()
        if directory:
            self.image_list = self.get_image_files_recursive(directory)
            if self.image_list:
                self.current_index = 0
                self.show_image()
            else:
                self.show_message("No images found in this folder.")

    def open_image(self) -> None:
        """Choose a single image to open."""
        image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*" + ";*".join(SUPPORTED_IMAGE_FORMATS))]
        )
        if image_path:
            self.image_list = [image_path]
            self.current_index = 0
            self.show_image()

    def get_image_files_recursive(self, directory: str) -> List[str]:
        """
        Recursively gather supported image files.
        
        Args:
            directory: Root directory to search
            
        Returns:
            List of image file paths
        """
        image_files = []
        for root_dir, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(SUPPORTED_IMAGE_FORMATS):
                    image_files.append(os.path.join(root_dir, file))
        return sorted(image_files)

    def show_image(self) -> None:
        """Display the current image with zoom applied."""
        if 0 <= self.current_index < len(self.image_list):
            image_path = self.image_list[self.current_index]
            try:
                image = Image.open(image_path)

                # Apply zoom
                width, height = image.size
                new_size = (int(width * self.zoom_factor), int(height * self.zoom_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

                self.photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=self.photo, text='')
                self.image_label.image = self.photo
                self.root.title(f"Image Audit Toolkit - {os.path.basename(image_path)}")
            except UnidentifiedImageError:
                self.show_message(f"Cannot open image: {image_path}")
            except Exception as e:
                self.show_message(f"Error opening image:\n{e}")
        else:
            self.show_message("No image to display.")

    def next_image(self) -> None:
        """Move to the next image."""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.zoom_factor = DEFAULT_ZOOM_FACTOR
            self.show_image()

    def previous_image(self) -> None:
        """Move to the previous image."""
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.zoom_factor = DEFAULT_ZOOM_FACTOR
            self.show_image()

    def zoom_in(self) -> None:
        """Increase the zoom level."""
        if self.image_list:
            self.zoom_factor *= ZOOM_FACTOR_INCREMENT
            self.show_image()

    def zoom_out(self) -> None:
        """Decrease the zoom level."""
        if self.image_list:
            self.zoom_factor /= ZOOM_FACTOR_INCREMENT
            self.show_image()

    def show_message(self, message: str) -> None:
        """
        Display a message in the image area.
        
        Args:
            message: Message to display
        """
        self.image_label.config(image='', text=message, bg="grey", fg="white")

    def get_current_image_path(self) -> Optional[str]:
        """
        Get the path of the current image.
        
        Returns:
            Path to the current image or None if no image is loaded
        """
        if 0 <= self.current_index < len(self.image_list):
            return self.image_list[self.current_index]
        return None

    # ========================= Audit Tools =========================

    def view_exif_data(self) -> None:
        """Show EXIF data of the current image."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            exif_dict = piexif.load(image_path)
            if not exif_dict or all(not exif_dict[ifd] for ifd in exif_dict):
                messagebox.showinfo("EXIF Data", "No EXIF data found.")
                return

            readable_exif = {}
            for ifd in exif_dict:
                if exif_dict[ifd]:
                    readable_exif[ifd] = {}
                    for tag, value in exif_dict[ifd].items():
                        tag_name = piexif.TAGS[ifd][tag]["name"]
                        try:
                            if isinstance(value, bytes):
                                value = value.decode('utf-8', errors='ignore')
                            readable_exif[tag_name] = value
                        except:
                            readable_exif[tag_name] = value

            exif_pretty = json.dumps(readable_exif, indent=4)

            # Display EXIF in a new window
            exif_window = Toplevel(self.root)
            exif_window.title("EXIF Data")
            exif_window.geometry("600x400")

            scrollbar = Scrollbar(exif_window)
            scrollbar.pack(side=RIGHT, fill=Y)

            text_area = Text(exif_window, wrap='word', yscrollcommand=scrollbar.set)
            text_area.pack(expand=YES, fill=BOTH)
            scrollbar.config(command=text_area.yview)

            text_area.insert(END, exif_pretty)
            text_area.config(state='disabled')

        except piexif.InvalidImageDataError:
            messagebox.showinfo("EXIF Data", "No EXIF data found.")
        except Exception as e:
            messagebox.showerror("Error", f"Couldn't retrieve EXIF data:\n{e}")

    def check_manipulation(self) -> None:
        """Run basic manipulation detection."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create progress dialog
        progress = ProgressDialog(self.root, "Manipulation Detection", "Analyzing image for manipulation...")
        
        def run_detection():
            try:
                artifact_map, dct_map, noise_map = self.analyzer.detect_manipulated_regions(image_path)
                # Switch back to the main thread for UI updates
                self.root.after(0, lambda: self._show_manipulation_results(image_path, artifact_map, dct_map, noise_map, progress))
            except Exception as e:
                self.root.after(0, lambda: self._show_error("Manipulation Detection Error", f"Detection failed:\n{e}", progress))
        
        # Run detection in a separate thread
        threading.Thread(target=run_detection).start()

    def _show_manipulation_results(self, image_path, artifact_map, dct_map, noise_map, progress=None):
        """Display the results of manipulation detection."""
        if progress:
            progress.close()
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("Error", "Cannot load image for plotting.")
                return

            fig = Figure(figsize=(20, 5))
            axes = fig.subplots(1, 4)

            # Original
            axes[0].set_title("Original Image")
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].axis('off')

            # DCT Map
            axes[1].set_title("DCT Map")
            axes[1].imshow(dct_map, cmap='hot')
            axes[1].axis('off')

            # Noise Map
            axes[2].set_title("Noise Map")
            axes[2].imshow(noise_map, cmap='hot')
            axes[2].axis('off')

            # Artifact Map
            axes[3].set_title("Artifact Map")
            axes[3].imshow(artifact_map, cmap='hot')
            axes[3].axis('off')

            # Show plots
            plot_window = Toplevel(self.root)
            plot_window.title("Manipulation Detection Results")
            plot_window.geometry("1200x400")

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=YES, fill=BOTH)
        except Exception as e:
            messagebox.showerror("Plot Error", f"Error creating visualization:\n{e}")

    def _show_error(self, title, message, progress=None):
        """Show an error message and close the progress dialog."""
        if progress:
            progress.close()
        messagebox.showerror(title, message)

    def run_ela(self) -> None:
        """Perform Error Level Analysis."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create progress dialog
        progress = ProgressDialog(self.root, "Error Level Analysis", "Performing ELA...")
        
        def run_ela():
            try:
                ela_image = self.analyzer.error_level_analysis(image_path)
                # Switch back to the main thread for UI updates
                self.root.after(0, lambda: self._show_ela_results(ela_image, progress))
            except Exception as e:
                self.root.after(0, lambda: self._show_error("ELA Error", f"ELA failed:\n{e}", progress))
        
        # Run ELA in a separate thread
        threading.Thread(target=run_ela).start()

    def _show_ela_results(self, ela_image, progress=None):
        """Display the results of ELA."""
        if progress:
            progress.close()
            
        if ela_image:
            # Show ELA image
            ela_window = Toplevel(self.root)
            ela_window.title("Error Level Analysis (ELA)")
            ela_window.geometry("600x600")

            ela_photo = ImageTk.PhotoImage(ela_image)
            label = Label(ela_window, image=ela_photo)
            label.image = ela_photo
            label.pack(expand=YES, fill=BOTH)
        else:
            messagebox.showinfo("ELA", "No significant ELA results found.")

    def run_clone_detection(self) -> None:
        """Detect cloned areas in the image."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create progress dialog
        progress = ProgressDialog(self.root, "Clone Detection", "Detecting cloned regions...")
        
        def run_detection():
            try:
                clone_image = self.analyzer.detect_clones_optimized(image_path)
                # Switch back to the main thread for UI updates
                self.root.after(0, lambda: self._show_clone_results(clone_image, progress))
            except Exception as e:
                self.root.after(0, lambda: self._show_error("Clone Detection Error", f"Clone detection failed:\n{e}", progress))
        
        # Run detection in a separate thread
        threading.Thread(target=run_detection).start()

    def _show_clone_results(self, clone_image, progress=None):
        """Display the results of clone detection."""
        if progress:
            progress.close()
            
        if clone_image is not None:
            clone_image_rgb = cv2.cvtColor(clone_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(clone_image_rgb)

            # Show clone detection result
            clone_window = Toplevel(self.root)
            clone_window.title("Clone Detection")
            clone_window.geometry("800x600")

            clone_photo = ImageTk.PhotoImage(pil_image)
            label = Label(clone_window, image=clone_photo)
            label.image = clone_photo
            label.pack(expand=YES, fill=BOTH)
        else:
            messagebox.showinfo("Clone Detection", "No cloned regions detected.")

    def run_histogram_analysis(self) -> None:
        """Analyze the image histogram for anomalies."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Create progress dialog
        progress = ProgressDialog(self.root, "Histogram Analysis", "Analyzing histogram...")
        
        def run_analysis():
            try:
                anomalies = self.analyzer.analyze_histogram(image_path)
                image = cv2.imread(image_path)
                
                # Create histogram visualization
                fig = Figure(figsize=(10, 5))
                plot = fig.add_subplot(111)
                colors = ('b', 'g', 'r')
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    plot.plot(hist, color=color, label=f'{color.upper()} Channel')
                plot.set_title("Color Histogram")
                plot.set_xlabel("Pixel Intensity")
                plot.set_ylabel("Frequency")
                plot.legend()
                plot.grid()
                
                # Switch back to the main thread for UI updates
                self.root.after(0, lambda: self._show_histogram_results(fig, anomalies, progress))
            except Exception as e:
                self.root.after(0, lambda: self._show_error("Histogram Analysis Error", f"Histogram analysis failed:\n{e}", progress))
        
        # Run analysis in a separate thread
        threading.Thread(target=run_analysis).start()

    def _show_histogram_results(self, fig, anomalies, progress=None):
        """Display the results of histogram analysis."""
        if progress:
            progress.close()
            
        # Show histogram
        histogram_window = Toplevel(self.root)
        histogram_window.title("Histogram Analysis")
        histogram_window.geometry("800x600")
        
        # Add anomaly findings if any
        if anomalies:
            anomaly_frame = Frame(histogram_window)
            anomaly_frame.pack(pady=10, fill='x')
            
            Label(anomaly_frame, text="Potential Anomalies Detected:", font=('Arial', 12, 'bold')).pack(anchor='w', padx=20)
            for anomaly in anomalies:
                Label(anomaly_frame, text=f"• {anomaly}", font=('Arial', 10)).pack(anchor='w', padx=30)
        else:
            Label(histogram_window, text="No anomalies detected in the histogram.", 
                  font=('Arial', 12)).pack(pady=10)
        
        # Add histogram visualization
        canvas = FigureCanvasTkAgg(fig, master=histogram_window)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=YES, fill=BOTH, padx=20, pady=(0, 20))

    def find_matching_cameras_ui(self) -> None:
        """UI to find and display cameras matching the image resolution."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        resolution = self.camera_matcher.get_image_resolution(image_path)
        if not resolution:
            messagebox.showerror("Error", "Couldn't get image resolution.")
            return

        width, height = resolution
        print(f"Image resolution: {width}x{height}")

        # Warn user about potential delay
        proceed = messagebox.askyesno(
            "Search Warning",
            "Searching for matching cameras might take some time and could make the app unresponsive.\nContinue?"
        )
        if not proceed:
            return

        # Create progress dialog
        progress = ProgressDialog(self.root, "Camera Search", "Searching for matching cameras...")
        
        def update_progress(msg):
            progress.update_message(msg)
        
        def run_search():
            try:
                matches = self.camera_matcher.find_matching_cameras(
                    resolution, 
                    self.dataset_folder,
                    progress_callback=update_progress
                )
                # Switch back to the main thread for UI updates
                self.root.after(0, lambda: self._display_matching_cameras(matches, resolution, progress))
            except Exception as e:
                self.root.after(0, lambda: self._show_error("Camera Search Error", f"Camera search failed:\n{e}", progress))
        
        # Run search in a separate thread
        threading.Thread(target=run_search).start()

    def _display_matching_cameras(self, matches, resolution, progress=None):
        """Show matched cameras in a new window."""
        if progress:
            progress.close()
            
        width, height = resolution
        if not matches:
            messagebox.showinfo("No Matches", f"No cameras found with resolution: {width}x{height}.")
            return

        # New window for matches
        list_window = Toplevel(self.root)
        list_window.title(f"Matching Cameras - {width}x{height}")
        list_window.geometry("500x400")

        # Header
        header = Label(list_window, text=f"Cameras with {width}x{height} resolution:", font=('Arial', 12, 'bold'))
        header.pack(pady=(10, 5), anchor='w', padx=10)
        
        # Count info
        count_info = Label(list_window, text=f"Found {len(matches)} matching camera(s)", font=('Arial', 10))
        count_info.pack(pady=(0, 10), anchor='w', padx=10)

        # Create frame for list and scrollbar
        list_frame = Frame(list_window)
        list_frame.pack(expand=YES, fill=BOTH, padx=10, pady=(0, 10))

        scrollbar = Scrollbar(list_frame)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox = Listbox(list_frame, selectmode=SINGLE, yscrollcommand=scrollbar.set, font=('Arial', 10))
        for idx, camera in enumerate(matches, start=1):
            listbox.insert(END, f"{idx}. {camera['Name']} (Code: {camera['ProductCode']})")
        listbox.pack(expand=YES, fill=BOTH)
        scrollbar.config(command=listbox.yview)

        # Button frame
        button_frame = Frame(list_window)
        button_frame.pack(pady=(0, 10), fill='x')

        # Button to view details
        view_button = Button(button_frame, text="View Selected Camera Details",
                            command=lambda: self.show_camera_details(listbox, matches))
        view_button.pack(side='left', padx=10)
        
        # Button to close
        close_button = Button(button_frame, text="Close", command=list_window.destroy)
        close_button.pack(side='right', padx=10)

    def show_camera_details(self, listbox, matches):
        """Display details of the selected camera."""
        selected = listbox.curselection()
        if not selected:
            messagebox.showwarning("No Selection", "Please choose a camera.")
            return

        index = selected[0]
        camera = matches[index]

        # New window for details
        details_window = Toplevel(self.root)
        details_window.title(f"Camera Details - {camera['Name']}")
        details_window.geometry("700x600")

        # Camera name header
        header = Label(details_window, text=camera['Name'], font=('Arial', 14, 'bold'))
        header.pack(pady=(10, 5), padx=10, anchor='w')
        
        # Product code
        if camera['ProductCode'] != 'N/A':
            product_code = Label(details_window, text=f"Product Code: {camera['ProductCode']}", font=('Arial', 10))
            product_code.pack(padx=10, anchor='w')
        
        # Main content frame
        content_frame = Frame(details_window)
        content_frame.pack(expand=YES, fill=BOTH, padx=10, pady=10)

        scrollbar = Scrollbar(content_frame)
        scrollbar.pack(side=RIGHT, fill=Y)

        text_area = Text(content_frame, wrap='word', yscrollcommand=scrollbar.set, font=('Arial', 10))
        text_area.pack(expand=YES, fill=BOTH)
        scrollbar.config(command=text_area.yview)

        # Format details
        details = "\n"
        
        if camera['URL'] != 'N/A':
            details += f"URL: {camera['URL']}\n"
            
        if camera['ImageURL'] != 'N/A':
            details += f"Image URL: {camera['ImageURL']}\n"
            
        if camera['Award'] != 'N/A':
            details += f"Award: {camera['Award']}\n"
            
        if camera['ShortSpecs']:
            details += "\nShort Specs:\n"
            for spec in camera['ShortSpecs']:
                details += f"  • {spec}\n"
        
        details += "\nSpecifications:\n"
        for key, value in camera['Specs'].items():
            if isinstance(value, list):
                details += f"  • {key}:\n"
                for item in value:
                    details += f"      - {item}\n"
            else:
                details += f"  • {key}: {value}\n"

        text_area.insert(END, details)
        text_area.config(state='disabled')
        
        # Close button
        close_button = Button(details_window, text="Close", command=details_window.destroy)
        close_button.pack(pady=10)

    def generate_report(self) -> None:
        """Create a JSON report of the current image analysis."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Warn user about potential delay
        proceed = messagebox.askyesno(
            "Report Generation Warning",
            "Generating a full report might take some time and could make the app unresponsive.\nContinue?"
        )
        if not proceed:
            return

        # Get save path first
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Report As"
        )
        if not save_path:
            return

        # Create progress dialog
        progress = ProgressDialog(self.root, "Report Generation", "Generating comprehensive report...")
        
        def update_progress(msg):
            progress.update_message(msg)
        
        def run_report_generation():
            try:
                report_generator = ReportGenerator()
                report = report_generator.generate_report(
                    image_path, 
                    self.analyzer, 
                    self.camera_matcher,
                    self.dataset_folder,
                    progress_callback=update_progress
                )
                
                # Save report to file
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=4)
                
                # Switch back to the main thread for UI updates
                self.root.after(0, lambda: self._report_saved(save_path, progress))
            except Exception as e:
                self.root.after(0, lambda: self._show_error("Report Generation Error", f"Couldn't generate report:\n{e}", progress))
        
        # Run report generation in a separate thread
        threading.Thread(target=run_report_generation).start()

    def _report_saved(self, save_path, progress=None):
        """Show confirmation that report was saved."""
        if progress:
            progress.close()
        messagebox.showinfo("Report Saved", f"Report saved to:\n{save_path}")


# ========================= Main Function =========================

def main():
    """Launch the application."""
    root = Tk()
    app = ImageAuditToolkit(root)
    root.mainloop()

if __name__ == "__main__":
    main()
