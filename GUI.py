import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import halftoning
import histogram
import basic_edge_detection
import advanced_edge_detection
import segmentation
import Spatial_Frequency
import image_operations

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processing Toolbox")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)  # Allow resizing of the window

        # Variables to store images and thresholds
        self.original_image = None
        self.processed_image = None
        self.detect_type = "SOBEL"
        self.threshold = tk.IntVar(value=128)  # Default threshold value
        self.high = tk.IntVar(value=255)  # High threshold for manual segmentation
        self.low = tk.IntVar(value=255)    # Low threshold for manual segmentation
        self.segment = tk.IntVar(value=0)  # Default segmentation technique
        self.peak_space = tk.IntVar(value = 10)
        self.value = 255
        self.high_mask = tk.StringVar(value = "First Mask")
        self.low_mask = tk.StringVar(value = "Mask 6")
        # Initialize UI components
        self.init_ui()

    def init_ui(self):
        # Create the top-level frames: Left, Middle, Right
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Frame: Original Image Display
        left_frame = tk.Frame(main_frame, bg="white", width=400)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.original_label = tk.Label(
            left_frame, text="Original Image", bg="gray", fg="white", font=("Helvetica", 14, "bold")
        )
        self.original_label.pack(fill=tk.BOTH, expand=True, pady=0)

        self.upload_button = tk.Button(
            left_frame, text="Upload Image", command=self.upload_image, font=("Helvetica", 12)
        )
        self.upload_button.pack(pady=5)

        # Right Frame: Processed Image Display
        right_frame = tk.Frame(main_frame, bg="white", width=400)
        right_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        self.processed_label = tk.Label(
            right_frame, text="Processed Image", bg="gray", fg="white", font=("Helvetica", 14, "bold")
        )
        self.processed_label.pack(fill=tk.BOTH, expand=True, pady=0)

        # Middle Frame: Scrollable Operations (Horizontal Scroll)
        middle_frame = tk.Frame(main_frame, width=400)
        middle_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Canvas widget for both horizontal and vertical scrolling
        canvas = tk.Canvas(middle_frame)
        canvas.grid(row=0, column=0, sticky="nsew")

        # Vertical and Horizontal Scrollbars
        vertical_scrollbar = tk.Scrollbar(middle_frame, orient="vertical", command=canvas.yview)
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")

        horizontal_scrollbar = tk.Scrollbar(middle_frame, orient="horizontal", command=canvas.xview)
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        canvas.configure(yscrollcommand=vertical_scrollbar.set, xscrollcommand=horizontal_scrollbar.set)

        # Create a frame inside the canvas to hold all operation buttons
        operation_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=operation_frame, anchor="nw")


        # Create a frame inside the canvas to hold all operation buttons
        operation_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=operation_frame, anchor="nw")

        # Add the operations
        operations = {
            "Halftoning": ["Simple (Threshold)", "Advanced (Error Diffusion)"],
            "Histogram": ["View Histogram", "Histogram Equalization"],
            "Basic Edge Detection": ["Sobel Operator", "Prewitt Operator", "Kirsch Compass"],
            "Advanced Edge Detection": [
                "Homogeneity Operator",
                "Difference Operator",
                "Difference of Gaussian (7x7)",
                "Difference of Gaussian (9x9)",
                "Contrast-Based",
                "Range",
                "Variance",
            ],
            "Filtering": ["High-Pass Filter", "Low-Pass Filter", "Median Filter"],
            "Image Operations": ["Invert Image", "Add Images", "Subtract Images"],
            "Histogram-Based Segmentation": [
                "Manual Technique",
                "Peak Technique",
                "Valley Technique",
                "Adaptive Technique",
            ],
        }

        for group_name, buttons in operations.items():
            self.create_operation_group(operation_frame, group_name, buttons)

        # Update the scroll region whenever operations are added
        operation_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # Make the middle frame expand and fill space
        middle_frame.grid_rowconfigure(0, weight=1)
        middle_frame.grid_columnconfigure(0, weight=1)

        # Configure column and row weights for left, middle, and right frames
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=2)  # Middle frame takes more space
        main_frame.grid_columnconfigure(2, weight=1)

    def create_operation_group(self, parent, group_name, operations):
        """Create a group of related operations with organized buttons and threshold input for edge detection."""
        frame = ttk.LabelFrame(parent, text=group_name, padding=(10, 5))
        frame.pack(fill=tk.X, pady=5)

        # Apply larger font for category labels
        frame.config(labelwidget=tk.Label(frame, text=group_name, font=("Helvetica", 14, "bold")))

        for operation in operations:
            # Create a frame for each operation to include button and optional input
            operation_frame = tk.Frame(frame, bg="lightgray")
            operation_frame.pack(fill=tk.X, pady=5, padx=5)

            # Operation Button
            button = tk.Button(
                operation_frame,
                text=operation,
                command=lambda op=operation: self.handle_operation(op),
                font=("Helvetica", 10),
                width=22,
            )
            button.pack(side=tk.LEFT, padx=5)

            # Threshold Input for Edge Detection operations
            if "Edge Detection" in group_name:
                threshold_label = tk.Label(
                    operation_frame, text="Threshold:", font=("Helvetica", 10), bg="lightgray"
                )
                threshold_label.pack(side=tk.LEFT, padx=5)

                threshold_entry = tk.Entry(
                    operation_frame, textvariable=self.threshold, width=5, font=("Helvetica", 10)
                )
                threshold_entry.pack(side=tk.LEFT, padx=5)

            if "Filtering" in group_name:
                if operation == "High-Pass Filter":
                    segment_label = tk.Label(operation_frame, text="High-Pass Mask:", font=("Helvetica", 10), bg="lightgray")
                    segment_label.pack(side=tk.LEFT, padx=5)
                    segment_menu = ttk.Combobox(
                        operation_frame, textvariable=self.high_mask, values=["First Mask" , "Second Mask", "Third Mask"],
                        font=("Helvetica", 10)
                    )
                    segment_menu.pack(side=tk.LEFT, padx=5)
                if operation == "Low-Pass Filter":
                    segment_label = tk.Label(operation_frame, text="Low-Pass Mask:", font=("Helvetica", 10), bg="lightgray")
                    segment_label.pack(side=tk.LEFT, padx=5)
                    segment_menu = ttk.Combobox(
                        operation_frame, textvariable=self.low_mask, values=["Mask 6" , "Mask 9", "Mask 10", "Mask 16"],
                        font=("Helvetica", 10)
                    )
                    segment_menu.pack(side=tk.LEFT, padx=5)

            # Segmentation Thresholds (low, high) and Segment Type
            if "Segmentation" in group_name:
                if operation == "Manual Technique":
                    low_label = tk.Label(operation_frame, text="Low Threshold:", font=("Helvetica", 10), bg="lightgray")
                    low_label.pack(side=tk.LEFT, padx=5)
                    low_entry = tk.Entry(
                        operation_frame, textvariable=self.low, width=5, font=("Helvetica", 10)
                    )
                    low_entry.pack(side=tk.LEFT, padx=5)

                    high_label = tk.Label(operation_frame, text="High Threshold:", font=("Helvetica", 10), bg="lightgray")
                    high_label.pack(side=tk.LEFT, padx=5)
                    high_entry = tk.Entry(
                        operation_frame, textvariable=self.high, width=5, font=("Helvetica", 10)
                    )
                    high_entry.pack(side=tk.LEFT, padx=5)
                else:
                    low_label = tk.Label(operation_frame, text="Peak Space:", font=("Helvetica", 10), bg="lightgray")
                    low_label.pack(side=tk.LEFT, padx=5)
                    low_entry = tk.Entry(
                        operation_frame, textvariable=self.peak_space, width=5, font=("Helvetica", 10)
                    )
                    low_entry.pack(side=tk.LEFT, padx=5)


                # Segment Type (Manual, Peak, Valley, Adaptive)
                segment_label = tk.Label(operation_frame, text="Segment:", font=("Helvetica", 10), bg="lightgray")
                segment_label.pack(side=tk.LEFT, padx=5)
                segment_entry = tk.Entry(
                        operation_frame, textvariable=self.segment, width=5, font=("Helvetica", 10)
                    )
                segment_entry.pack(side=tk.LEFT, padx=5)
                # segment_menu = ttk.Combobox(
                #     operation_frame, textvariable=self.segment, values=[0 , 1],
                #     font=("Helvetica", 10)
                # )
                # segment_menu.pack(side=tk.LEFT, padx=5)

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
        threshold_value = self.threshold.get()  # Retrieve the threshold value
        low_value = self.low.get()  # Retrieve the low threshold for manual technique
        high_value = self.high.get()  # Retrieve the high threshold for manual technique
        segment_value = self.segment.get()  # Retrieve selected segmentation type
        peak_space = self.peak_space.get()
        high_mask = self.high_mask.get()
        low_mask = self.low_mask.get()
        rows, columns = self.original_image.shape
        out_image = np.zeros_like(self.original_image)
        # Perform operations
        if operation == "Advanced (Error Diffusion)":
            self.processed_image = halftoning.error_diffusion(self.original_image)
        elif operation == "Simple (Threshold)":
            self.processed_image = halftoning.simple_threshold(self.original_image)
        elif operation == "View Histogram":
            self.processed_image = histogram.display_histogram(self.original_image)
        elif operation == "Histogram Equalization":
            self.processed_image =  histogram.histogram_equalization(self.original_image)
        elif operation == "Sobel Operator":
            self.processed_image = basic_edge_detection.detect_edges(self.original_image, "SOBEL", threshold_value)
        elif operation == "Prewitt Operator":
            self.processed_image = basic_edge_detection.detect_edges(self.original_image, "PREWITT", threshold_value)
        elif operation == "Kirsch Compass":
           self.processed_image = basic_edge_detection.detect_edges(self.original_image, "KIRSCH", threshold_value)
        elif operation == "Homogeneity Operator":
            self.processed_image = advanced_edge_detection.homogeneity(self.original_image, threshold_value)
        elif operation == "Difference Operator":
            self.processed_image = advanced_edge_detection.difference_edge(self.original_image, threshold_value)
        elif operation == "Difference of Gaussian (7x7)":
            self.processed_image = advanced_edge_detection.gaussian_difference(self.original_image, threshold_value, "7x7")
        elif operation == "Difference of Gaussian (9x9)":
            self.processed_image = advanced_edge_detection.gaussian_difference(self.original_image, threshold_value, "9x9")
        elif operation == "Range":
            self.processed_image = advanced_edge_detection.range_filter(self.original_image, threshold_value, 3)
        elif operation == "Variance":
            self.processed_image = advanced_edge_detection.variance_filter(self.original_image, threshold_value, 3)
        elif operation == "Contrast-Based":
            self.processed_image = advanced_edge_detection.contrast_edge(self.original_image, "LAPLACE", threshold_value)
        elif operation == "High-Pass Filter":
           self.processed_image = Spatial_Frequency.conv(self.original_image, high_mask)
        elif operation == "Low-Pass Filter":
            self.processed_image = Spatial_Frequency.conv(self.original_image, low_mask)
        elif operation == "Median Filter":
            self.processed_image = Spatial_Frequency.median_filter(self.original_image)
        elif operation == "Invert Image":
            self.processed_image = image_operations.invert_image(self.original_image)
        elif operation == "Subtract Images":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Add Images":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Manual Technique":
            self.processed_image = segmentation.manual_threshold_segmentation(self.original_image, high_value, low_value ,self.value,  segment_value)
        elif operation == "Peak Technique":
            segmentation.peak_threshold_segmentation(self.original_image, out_image, self.value ,  segment_value,  rows, columns, peak_space)
            self.processed_image = out_image.astype(np.uint8)
        elif operation == "Valley Technique":
            segmentation.valley_threshold_segmentation(self.original_image, out_image, self.value ,  segment_value,  rows, columns, peak_space)
            self.processed_image = out_image.astype(np.uint8)
        elif operation == "Adaptive Technique":
            segmentation.adaptive_threshold_segmentation(self.original_image, out_image, self.value ,  segment_value, rows, columns,peak_space)
            self.processed_image = out_image.astype(np.uint8)        
        else:
            print("No Operation Selected")
        
        self.display_image(self.processed_image, self.processed_label)

    def display_image(self, image, label, max_size=(400, 400)):
        """Display an image in a Tkinter Label with a maximum size constraint."""
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize the image
        height, width = image.shape[:2]
        max_width, max_height = max_size
        scaling_factor = min(max_width / width, max_height / height, 1)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Convert to PIL Image
        image_pil = Image.fromarray(resized_image)
        image_tk = ImageTk.PhotoImage(image_pil)

        # Update label
        label.config(image=image_tk)
        label.image = image_tk  # Keep reference to avoid garbage collection


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop() 