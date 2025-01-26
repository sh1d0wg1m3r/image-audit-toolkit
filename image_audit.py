import os
import re
import yaml
import cv2
import numpy as np
import json
from PIL import Image, ImageChops, ImageEnhance, ImageTk, UnidentifiedImageError
from tkinter import (
    Tk, Label, Button, Menu, filedialog, messagebox, Toplevel,
    Text, Scrollbar, RIGHT, Y, END, BOTH, YES, BOTTOM, Frame, Listbox, SINGLE
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.fftpack import dct, idct
import piexif
import matplotlib.pyplot as plt
import chardet
import threading

# ========================= Button Frame =========================

class ButtonFrame(Frame):
    def __init__(self, root, toolkit):
        """Set up navigation and zoom buttons."""
        super().__init__(root)
        self.pack(side=BOTTOM, fill='x')

        # Open folder
        self.open_folder_btn = Button(self, text="Open Folder", command=toolkit.open_folder)
        self.open_folder_btn.pack(side='left', padx=5, pady=5)

        # Open single image
        self.open_image_btn = Button(self, text="Open Image", command=toolkit.open_image)
        self.open_image_btn.pack(side='left', padx=5, pady=5)

        # Previous image
        self.prev_btn = Button(self, text="<< Previous", command=toolkit.previous_image)
        self.prev_btn.pack(side='left', padx=5, pady=5)

        # Next image
        self.next_btn = Button(self, text="Next >>", command=toolkit.next_image)
        self.next_btn.pack(side='left', padx=5, pady=5)

        # Zoom in
        self.zoom_in_btn = Button(self, text="Zoom In", command=toolkit.zoom_in)
        self.zoom_in_btn.pack(side='right', padx=5, pady=5)

        # Zoom out
        self.zoom_out_btn = Button(self, text="Zoom Out", command=toolkit.zoom_out)
        self.zoom_out_btn.pack(side='right', padx=5, pady=5)


# ========================= Image Audit Toolkit =========================

class ImageAuditToolkit:
    def __init__(self, root):
        """Initialize the main application window."""
        self.root = root
        self.root.title("Image Audit Toolkit")
        self.root.geometry("1000x700")

        # Setup variables
        self.image_list = []
        self.current_index = -1
        self.photo = None
        self.zoom_factor = 1.0

        # Image display area
        self.image_label = Label(root, text="No image loaded.", bg="grey", fg="white")
        self.image_label.pack(expand=YES, fill=BOTH)

        # Menus and buttons
        self.init_menu()
        self.init_buttons()
        self.bind_keyboard_events()

        # Dataset folder
        self.dataset_folder = "dataset"

    def init_menu(self):
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

    def init_buttons(self):
        """Add navigation and zoom buttons."""
        ButtonFrame(self.root, self)

    def bind_keyboard_events(self):
        """Link keyboard shortcuts to actions."""
        self.root.bind('<Left>', lambda event: self.previous_image())
        self.root.bind('<Right>', lambda event: self.next_image())
        self.root.bind('<plus>', lambda event: self.zoom_in())
        self.root.bind('<KP_Add>', lambda event: self.zoom_in())
        self.root.bind('<minus>', lambda event: self.zoom_out())
        self.root.bind('<KP_Subtract>', lambda event: self.zoom_out())
        self.root.bind('<Escape>', lambda event: self.root.quit())

    # ========================= Image Viewing =========================

    def open_folder(self):
        """Select a folder and load images."""
        directory = filedialog.askdirectory()
        if directory:
            self.image_list = self.get_image_files_recursive(directory)
            if self.image_list:
                self.current_index = 0
                self.show_image()
            else:
                self.show_message("No images found in this folder.")

    def open_image(self):
        """Choose a single image to open."""
        image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.gif;*.webp")]
        )
        if image_path:
            self.image_list = [image_path]
            self.current_index = 0
            self.show_image()

    def get_image_files_recursive(self, directory):
        """Recursively gather supported image files."""
        supported_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
        image_files = []
        for root_dir, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_files.append(os.path.join(root_dir, file))
        return sorted(image_files)

    def show_image(self):
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

    def next_image(self):
        """Move to the next image."""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.zoom_factor = 1.0
            self.show_image()

    def previous_image(self):
        """Move to the previous image."""
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.zoom_factor = 1.0
            self.show_image()

    def zoom_in(self):
        """Increase the zoom level."""
        if self.image_list:
            self.zoom_factor *= 1.25
            self.show_image()

    def zoom_out(self):
        """Decrease the zoom level."""
        if self.image_list:
            self.zoom_factor /= 1.25
            self.show_image()

    def show_message(self, message):
        """Display a message in the image area."""
        self.image_label.config(image='', text=message, bg="grey", fg="white")

    def get_current_image_path(self):
        """Get the path of the current image."""
        if 0 <= self.current_index < len(self.image_list):
            return self.image_list[self.current_index]
        return None

    # ========================= Audit Tools =========================

    def view_exif_data(self):
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

    def check_manipulation(self):
        """Run basic manipulation detection."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            artifact_map, dct_map, noise_map = self.detect_manipulated_regions(image_path)
            self.plot_results(image_path, artifact_map, dct_map, noise_map)
        except Exception as e:
            messagebox.showerror("Error", f"Manipulation check failed:\n{e}")

    def run_ela(self):
        """Perform Error Level Analysis."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            ela_image = self.error_level_analysis(image_path)
            if ela_image:
                # Show ELA image
                ela_window = Toplevel(self.root)
                ela_window.title("Error Level Analysis (ELA)")
                ela_window.geometry("600x600")

                ela_photo = ImageTk.PhotoImage(ela_image)
                label = Label(ela_window, image=ela_photo)
                label.image = ela_photo
                label.pack(expand=YES, fill=BOTH)
        except Exception as e:
            messagebox.showerror("Error", f"ELA failed:\n{e}")

    def run_clone_detection(self):
        """Detect cloned areas in the image."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            clone_image = self.detect_clones(image_path)
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
        except Exception as e:
            messagebox.showerror("Error", f"Clone detection failed:\n{e}")

    def run_histogram_analysis(self):
        """Analyze the image histogram for anomalies."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            anomalies = self.analyze_histogram(image_path)
            if anomalies:
                messagebox.showinfo("Histogram Analysis", "\n".join(anomalies))
        except Exception as e:
            messagebox.showerror("Error", f"Histogram analysis failed:\n{e}")

    # ========================= Manipulation Detection =========================

    def dct_block_analysis(self, block):
        """Analyze a block using DCT for artifacts."""
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
        return high_freq_energy

    def noise_analysis(self, image, block_size=8):
        """Estimate noise levels across the image."""
        noise_map = np.zeros_like(image, dtype=np.float32)
        height, width = image.shape
        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block = image[y:y+block_size, x:x+block_size]
                noise_map[y:y+block_size, x:x+block_size] = np.std(block)
        return noise_map

    def detect_manipulated_regions(self, image_path, block_size=8, dct_threshold=10, noise_threshold=5):
        """Identify manipulated areas using DCT and noise analysis."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Cannot load image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dct_map = np.zeros_like(gray, dtype=np.float32)
        noise_map = self.noise_analysis(gray, block_size)

        height, width = gray.shape
        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                dct_energy = self.dct_block_analysis(block)
                dct_map[y:y+block_size, x:x+block_size] = dct_energy

        dct_norm = cv2.normalize(dct_map, None, 0, 255, cv2.NORM_MINMAX)
        noise_norm = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX)

        artifact_map = cv2.addWeighted(dct_norm, 0.5, noise_norm, 0.5, 0)
        _, artifact_thresh = cv2.threshold(artifact_map, 127, 255, cv2.THRESH_BINARY)

        return artifact_thresh, dct_norm, noise_norm

    def plot_results(self, image_path, artifact_map, dct_map, noise_map):
        """Display analysis results in a new window."""
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

    # ========================= Error Level Analysis =========================

    def error_level_analysis(self, image_path, quality=90):
        """Perform ELA to spot inconsistencies."""
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

    # ========================= Clone Detection =========================

    def detect_clones(self, image_path, block_size=32, threshold=0.9):
        """Find duplicated regions within the image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Cannot load image.")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clone_mask = np.zeros_like(gray)

            height, width = gray.shape
            for y in range(0, height - block_size + 1, block_size):
                for x in range(0, width - block_size + 1, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    for y2 in range(y, height - block_size + 1, block_size):
                        for x2 in range(x, width - block_size + 1, block_size):
                            if y2 == y and x2 == x:
                                continue
                            block2 = gray[y2:y2+block_size, x2:x2+block_size]
                            res = cv2.matchTemplate(block, block2, cv2.TM_CCOEFF_NORMED)
                            similarity = res[0][0]
                            if similarity > threshold:
                                clone_mask[y:y+block_size, x:x+block_size] = 255
                                clone_mask[y2:y2+block_size, x2:x2+block_size] = 255

            clone_mask_bgr = cv2.cvtColor(clone_mask, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(image, 0.7, clone_mask_bgr, 0.3, 0)

            return result
        except Exception as e:
            raise Exception(f"Clone Detection Error: {e}")

    # ========================= Histogram Analysis =========================

    def analyze_histogram(self, image_path):
        """Check histograms for unusual patterns."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Cannot load image.")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            colors = ('r', 'g', 'b')
            histograms = []
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms.append(hist)

            fig = Figure(figsize=(10, 5))
            plot = fig.add_subplot(111)
            for i, color in enumerate(colors):
                plot.plot(histograms[i], color=color, label=f'{color.upper()} Channel')
            plot.set_title("Color Histogram")
            plot.set_xlabel("Pixel Intensity")
            plot.set_ylabel("Frequency")
            plot.legend()
            plot.grid()

            # Show histogram
            histogram_window = Toplevel(self.root)
            histogram_window.title("Histogram Analysis")
            canvas = FigureCanvasTkAgg(fig, master=histogram_window)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=YES, fill=BOTH)

            # Detect anomalies
            anomalies = []
            for i, hist in enumerate(histograms):
                channel = colors[i].upper()
                max_freq = np.max(hist)
                min_freq = np.min(hist)
                if max_freq > 10000:
                    anomalies.append(f"High spike in {channel} channel (Max: {int(max_freq)}).")
                if min_freq == 0:
                    anomalies.append(f"Gap detected in {channel} channel (Min: {int(min_freq)}).")

            return anomalies
        except Exception as e:
            raise Exception(f"Histogram Analysis Error: {e}")

    # ========================= YAML Processing =========================

    def get_image_resolution(self, image_path):
        """Get the image's width and height."""
        try:
            img = Image.open(image_path)
            return img.size
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            return None
        except Exception as e:
            print(f"Error getting resolution for {image_path}: {e}")
            return None

    def extract_resolutions_from_yaml(self, yaml_data):
        """Pull resolution details from YAML data."""
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

    def resolution_string_to_tuple(self, res_string):
        """Convert resolution string to (width, height)."""
        match = re.match(r'(\d+)\s*x\s*(\d+)', res_string, re.IGNORECASE)
        if match:
            try:
                width = int(match.group(1))
                height = int(match.group(2))
                return width, height
            except ValueError:
                return None
        return None

    def find_matching_cameras(self, image_resolution_tuple, dataset_folder="dataset"):
        """Find cameras matching the image resolution."""
        matches = []
        if not os.path.isdir(dataset_folder):
            print(f"Dataset folder '{dataset_folder}' not found.")
            return matches

        for filename in os.listdir(dataset_folder):
            if filename.endswith((".yaml", ".yml")):
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

                    extracted = self.extract_resolutions_from_yaml(specs)

                    for res_str in extracted:
                        yaml_res = self.resolution_string_to_tuple(res_str)
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

    def find_matching_cameras_ui(self):
        """UI to find and display cameras matching the image resolution."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        resolution = self.get_image_resolution(image_path)
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

        # Start search in a new thread
        search_thread = threading.Thread(target=self._find_matching_cameras_thread, args=(resolution,))
        search_thread.start()

    def _find_matching_cameras_thread(self, resolution):
        """Threaded search for matching cameras."""
        try:
            matches = self.find_matching_cameras(resolution, self.dataset_folder)
            self.root.after(0, lambda: self._display_matching_cameras(matches, resolution))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Camera search failed:\n{e}"))

    def _display_matching_cameras(self, matches, resolution):
        """Show matched cameras in a new window."""
        width, height = resolution
        if not matches:
            messagebox.showinfo("No Matches", f"No cameras found with resolution: {width}x{height}.")
            return

        # New window for matches
        list_window = Toplevel(self.root)
        list_window.title("Matching Cameras")
        list_window.geometry("500x400")

        scrollbar = Scrollbar(list_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        listbox = Listbox(list_window, selectmode=SINGLE, yscrollcommand=scrollbar.set)
        for idx, camera in enumerate(matches, start=1):
            listbox.insert(END, f"{idx}. {camera['Name']} (Code: {camera['ProductCode']})")
        listbox.pack(expand=YES, fill=BOTH)
        scrollbar.config(command=listbox.yview)

        # Button to view details
        view_button = Button(list_window, text="View Selected Camera",
                             command=lambda: self.show_camera_details(listbox, matches))
        view_button.pack(pady=10)

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

        scrollbar = Scrollbar(details_window)
        scrollbar.pack(side=RIGHT, fill=Y)

        text_area = Text(details_window, wrap='word', yscrollcommand=scrollbar.set)
        text_area.pack(expand=YES, fill=BOTH)
        scrollbar.config(command=text_area.yview)

        # Format details
        details = (
            f"Name: {camera['Name']}\n"
            f"Product Code: {camera['ProductCode']}\n"
            f"URL: {camera['URL']}\n"
            f"Image URL: {camera['ImageURL']}\n"
            f"Award: {camera['Award']}\n\n"
            "Short Specs:\n" +
            "\n".join(f"  - {spec}" for spec in camera['ShortSpecs']) +
            "\n\nSpecs:\n" +
            "\n".join(
                f"  {key}: {value if not isinstance(value, list) else ', '.join(value)}"
                for key, value in camera['Specs'].items()
            )
        )

        text_area.insert(END, details)
        text_area.config(state='disabled')

    # ========================= Report Generation =========================

    def generate_report(self):
        """Create a JSON report of the current image analysis."""
        image_path = self.get_current_image_path()
        if not image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            report = {}
            image = Image.open(image_path)
            report['File Name'] = os.path.basename(image_path)
            report['File Path'] = image_path
            report['Format'] = image.format
            report['Size'] = image.size
            report['Mode'] = image.mode

            # EXIF Data
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
            manipulations = []
            if exif_dict and any(exif_dict[ifd] for ifd in exif_dict):
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
            try:
                artifact_map, dct_map, noise_map = self.detect_manipulated_regions(image_path)
                artifact_energy = float(np.sum(artifact_map) / 255)
                report['Advanced Detection'] = {
                    "Artifact Energy Sum (White Pixels)": artifact_energy
                }
            except Exception as e:
                report['Advanced Detection'] = f"Failed: {e}"

            # ELA
            try:
                ela_image = self.error_level_analysis(image_path)
                if ela_image:
                    ela_path = "temp_report_ela.jpg"
                    ela_image.save(ela_path, "JPEG", quality=90)
                    report['ELA'] = f"ELA image saved at: {os.path.abspath(ela_path)}"
            except Exception as e:
                report['ELA'] = f"Failed: {e}"

            # Clone Detection
            try:
                clone_image = self.detect_clones(image_path)
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
            try:
                anomalies = self.analyze_histogram(image_path)
                report['Histogram Analysis'] = anomalies if anomalies else ["No anomalies detected."]
            except Exception as e:
                report['Histogram Analysis'] = f"Failed: {e}"

            # Camera Matching
            try:
                resolution = self.get_image_resolution(image_path)
                if resolution:
                    matches = self.find_matching_cameras(resolution, self.dataset_folder)
                    if matches:
                        report['Matched Cameras'] = matches
                    else:
                        report['Matched Cameras'] = "No matching cameras found."
                else:
                    report['Matched Cameras'] = "Could not determine image resolution."
            except Exception as e:
                report['Matched Cameras'] = f"Error during camera matching: {e}"

            # Save report
            save_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Report As"
            )
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=4)
                messagebox.showinfo("Report Saved", f"Report saved to:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Report Error", f"Couldn't generate report:\n{e}")

    # ========================= Main Function =========================

def main():
    """Launch the application."""
    root = Tk()
    app = ImageAuditToolkit(root)
    root.mainloop()

if __name__ == "__main__":
    main()
