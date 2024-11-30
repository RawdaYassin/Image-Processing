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
        self.root.resizable(True, True)  # Allow resizing of the window

        # Variables to store images and thresholds
        self.original_image = None
        self.processed_image = None
        self.detect_type = "SOBEL"
        self.threshold = tk.IntVar(value=128)  # Default threshold value

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

        # Middle Frame: Scrollable Operations
        middle_frame = tk.Frame(main_frame, width=400)
        middle_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Canvas widget
        canvas = tk.Canvas(middle_frame)
        canvas.grid(row=0, column=0, sticky="nsew")

        # Scrollbar widget
        scrollbar = tk.Scrollbar(middle_frame, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")

        canvas.configure(yscrollcommand=scrollbar.set)

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
                font=("Helvetica", 12),
                width=25,
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

    def upload_image(self):
        """Upload an image."""
        image_path = filedialog.askopenfilename(
            title="Open Image File",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.gif;*.webp"),
                ("All files", "*.*"),
            ],
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
        # Perform operations
        if operation == "Advanced (Error Diffusion)":
            self.processed_image = halftoning.halftone(self.original_image, 128)
        elif operation == "Simple (Threshold)":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "View Histogram":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Histogram Equalization":
            self.processed_image = cv2.threshold(self.original_image, 128, 255, cv2.THRESH_BINARY)
        elif operation == "Sobel Operator":
            self.processed_image = basic_edge_detection.detect_edges(self.original_image, "SOBEL", threshold_value)
        elif operation == "Prewitt Operator":
            self.processed_image = basic_edge_detection.detect_edges(self.original_image, "PREWITT", threshold_value)
        elif operation == "Kirsch  Compass":
           self.processed_image = basic_edge_detection.detect_edges(self.original_image, "KIRSCH", threshold_value)
        elif operation == "Homogeneity Operator":
            self.processed_image = advanced_edge_detection.homogeneity(self.original_image, threshold_value)
        elif operation == "Difference Operator":
            self.processed_image = advanced_edge_detection.difference_edge(self.original_image, threshold_value)
        elif operation == "Differenece of Gaussian (7x7)":
            self.processed_image = advanced_edge_detection.gaussian_difference(self.original_image, threshold_value, "7x7")
        elif operation == "Differenece of Gaussian (9x9)":
            self.processed_image = advanced_edge_detection.gaussian_difference(self.original_image, threshold_value, "9x9")
        elif operation == "Range":
            self.processed_image = advanced_edge_detection.range_filter(self.original_image, threshold_value, 3)
        elif operation == "Variance":
            self.processed_image = advanced_edge_detection.variance_filter(self.original_image, threshold_value, 3)
        elif operation == "Contrast-Based":
            self.processed_image = advanced_edge_detection.homogeneity(self.original_image, self.detect_type, threshold_value)
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
