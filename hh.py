import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import halftoning
import histogram
import basic_edge_detection
import advanced_edge_detection


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processing Toolbox")
        self.root.geometry("1200x800")

        # Variables to store images
        self.original_image = None
        self.processed_image = None
        self.detect_type = "SOBEL"
        self.threshold = 128

        # Initialize UI components
        self.init_ui()

    def init_ui(self):
        # Left Frame: Image display and upload
        left_frame = tk.Frame(self.root, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.original_label = tk.Label(left_frame, text="Original Image", bg="gray", fg="white")
        self.original_label.pack(fill=tk.BOTH, expand=True, pady=5)

        self.upload_button = tk.Button(left_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Right Frame: Operations
        right_frame = tk.Frame(self.root, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.processed_label = tk.Label(right_frame, text="Processed Image", bg="gray", fg="white")
        self.processed_label.pack(fill=tk.BOTH, expand=True, pady=5)

        # Middle Frame: Buttons for operations """"
        middle_frame = tk.Frame(self.root, bg="lightgray")
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        operations = {
            "Halftoning": ["Simple (Threshold)", "Advanced (Error Diffusion)"],
            "Histogram": ["View Histogram", "Histogram Equalization"],
            "Basic Edge Detection": ["Sobel Operator", "Prewitt Operator", "Kirsch  Compass"],
            "Advancedv Edge Detection": ["Homogenity Operator", "Difference Operator", "Differenece of Gaussian (7x7)", "Differenece of Gaussian (9x9)", "Contrast-Based","Range", "Variance"],
            "Filtering": ["High-Pass Filter", "Low-Pass Filter", "Median Filter"],
            "Image Operations": ["Invert Image", "Add Images", "Subtract Images"],
            "Histogram-Based Segmentation": ["Manual Technique", "Peak Technique", "Valley Technique" , "Adaptive Technique"]
        }

        for group_name, buttons in operations.items():
            self.create_operation_group(middle_frame, group_name, buttons)

    # def create_operation_group(self, parent, group_name, operations):
    #     """Create a group of related operations."""
    #     frame = ttk.LabelFrame(parent, text=group_name)
    #     frame.pack(fill=tk.X, pady=5)

    #     for operation in operations:
    #         button = tk.Button(frame, text=operation, command=lambda op=operation: self.handle_operation(op))
    #         button.pack(side=tk.LEFT, padx=5, pady=5)
    def create_operation_group(self, parent, group_name, operations):
        """Create a group of related operations with one button per row."""
        frame = ttk.LabelFrame(parent, text=group_name, padding=(10, 5))
        frame.pack(fill=tk.X, pady=5, padx=10)

        for index, operation in enumerate(operations):
            button = tk.Button(
                frame, text=operation, width=25, 
                command=lambda op=operation: self.handle_operation(op)
            )
            # Place each button on a new row
            button.grid(row=index, column=0, padx=5, pady=5, sticky="w")
            


    def upload_image(self):
        """Upload an image."""
        image_path = filedialog.askopenfilename(
            title="Open Image File",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.gif;*.webp"), ("All files", "*.*")]
        )
        if image_path:
            self.original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image, self.original_label)
            self.original_image = Image.open(image_path).convert("L")
            self.original_image = np.array(self.original_image)

    def handle_operation(self, operation):
        """Handle image processing operation."""
        if self.original_image is None:
            return
        #self.original_image = np.array(self.original_image)
        if operation == "Advanced (Error Diffusion)":
            self.processed_image = halftoning.halftone(self.original_image, 128)
        elif operation == "Simple (Threshold)":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "View Histogram":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Histogram Equalization":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Sobel Operator":
            self.processed_image = basic_edge_detection.detect_edges(self.original_image, "SOBEL", 30)
        elif operation == "Prewitt Operator":
            self.processed_image = basic_edge_detection.detect_edges(self.original_image, "PREWITT", 30)
        elif operation == "Kirsch  Compass":
           self.processed_image = basic_edge_detection.detect_edges(self.original_image, "KIRSCH", 30)
        elif operation == "Homogenity Operator":
            self.processed_image = advanced_edge_detection.homogeneity(self.original_image, 30)
        elif operation == "Difference Operator":
            self.processed_image = advanced_edge_detection.difference_edge(self.original_image, 30)
        elif operation == "Differenece of Gaussian (7x7)":
            self.processed_image = advanced_edge_detection.gaussian_difference(self.original_image, 30, "7x7")
        elif operation == "Differenece of Gaussian (9x9)":
            self.processed_image = advanced_edge_detection.gaussian_difference(self.original_image, 30, "9x9")
        elif operation == "Range":
            self.processed_image = advanced_edge_detection.range_filter(self.original_image, 30, 3)
        elif operation == "Variance":
            self.processed_image = advanced_edge_detection.variance_filter(self.original_image, 30, 3)
        elif operation == "Contrast-Based":
            self.processed_image = advanced_edge_detection.homogeneity(self.original_image, self.detect_type, 30)
        elif operation == "High-Pass Filter":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Low-Pass Filter":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Median Filter":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Invert Image":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Subtract Images":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Add Images":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Manual Technique":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Peak Technique":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Valley Technique":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Adaptive Technique":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        else:
            print("No Operation Selected")
        self.display_image(self.processed_image, self.processed_label)

        """def display_image(self, image, label):
        #Display an image in a Tkinter Label.
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)

        label.config(image=image_tk)
        label.image = image_tk  # Keep a reference to avoid garbage collection"""

    def display_image(self, image, label, max_size=(400, 400)):
        """Display an image in a Tkinter Label with a maximum size constraint."""
        # Ensure the image has three channels (convert grayscale to RGB if needed)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize the image to fit within the max_size while maintaining aspect ratio
        height, width = image.shape[:2]
        max_width, max_height = max_size
        scaling_factor = min(max_width / width, max_height / height, 1)  # Scale down if larger

        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Convert to PIL Image and then to ImageTk
        image_pil = Image.fromarray(resized_image)
        image_tk = ImageTk.PhotoImage(image_pil)

        # Update the Tkinter label with the resized image
        label.config(image=image_tk)
        label.image = image_tk  # Keep a reference to avoid garbage collection



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop() 