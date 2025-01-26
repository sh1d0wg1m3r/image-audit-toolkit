# Image Audit Toolkit

## Hey there!

Welcome to the **Image Audit Toolkit**!  I'm working on this tool to help you analyze and check the integrity of your images. Whether you're a photographer, into digital forensics, or just want to reality check, this toolkit has some cool features to help you spot manipulations and dig into image metadata.

## What It Can Do

- **View Images:**
  - Open single images or whole folders.
  - Easily flip through your images. ( initially started as a slideshow app )
  - Zoom in and out to get a closer look.

- **EXIF Data Viewer:**
  - See all the EXIF metadata for your images.

- **Spot Manipulations:**
  - **Basic Checks:** Look at EXIF data, image format, and size.
  - **Advanced Detection:** Use DCT and noise analysis to find possible edits.
  - **Error Level Analysis (ELA):** Detect weird compression levels.
  - **Clone Detection:** Find duplicated parts in an image.
  - **Histogram Analysis:** Check color histograms for any odd patterns.
  - more will be added as fast as I learn german for my test 

- **Generate Reports:**
  - Create detailed audit reports in JSON, including matched camera info. ( may become unresponsive for a lot of time just wait it out ) 

- **Match Cameras:**
  - Find cameras that match your image resolutions using a built-in database.

## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/sh1d0wg1m3r/image-audit-toolkit.git
cd image-audit-toolkit
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install opencv-python pillow tkinter matplotlib scipy piexif chardet pyyaml
```
( because I hate requirements.txt )

If you come from the future and the dependencies are not installable or have changed ( Python 3.13.1 (tags/v3.13.1:0671451, Dec  3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)] on win32 ) -- my current version 
### 4. Prepare the Dataset

- Make sure there's a `dataset` folder in the root directory.
- This folder has YAML files with camera specs.

## How to Use It

### 1. Launch the App

```bash
python image_audit.py
```


### 2. Navigate the Interface

- **Menu Bar:**
  - **File:** Open images or folders and exit the app.
  - **Audit:** Access tools like the EXIF viewer, manipulation checks, ELA, clone detection, histogram analysis, and report generation.
  - **Dataset:** Find cameras that match your image resolutions. 

- **Buttons:**
  - **Open Folder:** Choose a folder with multiple images.
  - **Open Image:** Select a single image to analyze.
  - **Previous/Next:** Move through your loaded images.
  - **Zoom In/Out:** Get a closer or wider view of the image.

### 3. Create Reports

- After running your analyses, go to `Audit > Generate Report`.
- Pick where to save your JSON report.
- The report will include stuff like EXIF data, manipulation checks, ELA results, clone detection summaries, histogram analysis, and matched camera info.

## Where the Data Comes From

The **Image Audit Toolkit** uses a camera database from the [Open Product Data - Digital Cameras](https://github.com/open-product-data/digital-cameras) repo. This database has detailed specs of various digital cameras, helping the toolkit match your image resolutions to possible camera models.
( I don't see the original licence and have no idea what is permitted if you are the original creator please contact me here: zari@duck.com and I will resolve it as fast as possible )  

## What’s Broken

- **Performance:** Since it's early beta, some features like camera matching and advanced manipulation detection might be slow or glitchy with large datasets.
- **Incomplete Features:** Not everything is fully built yet. Your feedback can help decide what to tackle next.
- **Error Handling:** I've tried to handle errors smoothly, but unexpected issues might still pop up.

## Want to Help?
- **Help is welcome**

## License

[GNU General Public License v3.0](https://github.com/sh1d0wg1m3r/image-audit-toolkit/blob/main/LICENSE)

## Shoutouts

- **Open Product Data - Digital Cameras:** For the camera specs database. [GitHub Repository](https://github.com/open-product-data/digital-cameras)
- **OpenCV, Pillow, Tkinter, Matplotlib, SciPy, piexif, chardet, PyYAML:** The awesome libraries that make this toolkit work.

---

*This README is here to help you understand and use the Image Audit Toolkit. As I keep developing the project, I'll update this document with new features, improvements, and changes.*
