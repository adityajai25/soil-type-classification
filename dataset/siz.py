from PIL import Image
import os

def convert_and_resize_images(input_folder, output_folder, target_size=(200, 200)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        input_path = os.path.join(input_folder, file)

        # Check if the file is a PNG or JPG
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".jpg")

            # Open the image
            with Image.open(input_path) as img:
                # Resize the image
                img = img.resize(target_size)

                # Convert to RGB if the image has an alpha channel (transparency)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')

                # Save as JPG
                img.save(output_path, "JPEG")

if __name__ == "__main__":
    input_folder = "Alluvial soil"
    output_folder = "Alluvial soil out"

    convert_and_resize_images(input_folder, output_folder)
