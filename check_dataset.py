import os

base_path = "classification_dataset"

for split in ["train", "valid", "test"]:
    print(f"\n--- {split.upper()} ---")
    for category in ["bird", "drone"]:
        path = os.path.join(base_path, split, category)
        count = len(os.listdir(path))
        print(f"{category}: {count} images")