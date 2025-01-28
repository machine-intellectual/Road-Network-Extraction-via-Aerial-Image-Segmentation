import os
import zipfile
root_dir = "datasets/maxtar/helene/quadkey"


def zip_directories_with_pre_post(parent_dirs, zip_name):
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate through each directory in the parent_dirs list
        for parent_dir in parent_dirs:
            # Check if the parent_dir exists and is actually a directory
            if os.path.isdir(parent_dir):
                # Check if 'pre' and 'post' subdirectories exist within the parent directory
                pre_dir = os.path.join(parent_dir, 'pre')
                post_dir = os.path.join(parent_dir, 'post')

                if os.path.isdir(pre_dir) and os.path.isdir(post_dir):
                    # If both 'pre' and 'post' directories are present, add this directory to the zip
                    for root, dirs, files in os.walk(parent_dir):
                        # Add each file and subdirectory to the zip archive
                        # The arcname parameter ensures that the directory structure inside the zip is maintained
                        zipf.write(root, arcname=os.path.relpath(root, parent_dir))
                        for file in files:
                            name = os.path.join(root, file)
                            arcname = os.path.relpath(os.path.join(root, file), root_dir)
                            zipf.write(name, arcname=arcname)


# Example usage

parent_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]  # Replace with the actual list of directories
zip_name = 'maxar.zip'  # Name of the output zip file

zip_directories_with_pre_post(parent_dirs, zip_name)
