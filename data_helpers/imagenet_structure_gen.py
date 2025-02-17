import shutil
from concurrent.futures.process import ProcessPoolExecutor
from classes import IMAGENET2012_CLASSES
import os

INPUT_PATH ="data/train_images"
OUTPUT_PATH = "/home/ruixi/datasets/imagenet_full_size/061417/train"

def copy_file(file_name):
    # Process only JPEG files (handles both .JPEG and .jpeg)
    if file_name.lower().endswith(".jpeg"):
        # image filepath format: <IMAGE_FILENAME>_<SYNSET_ID>.JPEG
        root, _ = os.path.splitext(file_name)
        _, synset_id = os.path.basename(root).rsplit("_", 1)
        label = IMAGENET2012_CLASSES.get(synset_id, "unknown")

        # Create target directory if it doesn't exist
        class_path = os.path.join(OUTPUT_PATH, label)
        os.makedirs(class_path, exist_ok=True)

        # Copy file from input to output path
        original_file_path = os.path.join(INPUT_PATH, file_name)
        target_file_path = os.path.join(class_path, file_name)
        shutil.copy(original_file_path, target_file_path)

def file_generator(path):
    """Yield one file at a time from the input directory."""
    for file_name in os.listdir(path):
        # If there are non-JPEG files, you can filter here
        if file_name.lower().endswith('.jpeg'):
            yield file_name


if __name__ == "__main__":
    files = file_generator(INPUT_PATH)

    # Use ThreadPoolExecutor to run file copies in parallel with 4 workers
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(executor.map(copy_file, files))

