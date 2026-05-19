import os

# =========================================
# FOLDERS
# =========================================

folders = [
    r"C:\Users\HP\Desktop\New folder\MRI imgs\Lung Cancer MRI NC\train\no_cancer",
    r"C:\Users\HP\Desktop\New folder\MRI imgs\Lung Cancer MRI NC\validate\no_cancer"
]

# =========================================
# RENAME FILES
# =========================================

for folder in folders:

    print(f"\n📂 Processing Folder:\n{folder}\n")

    count = 1

    for filename in os.listdir(folder):

        old_path = os.path.join(folder, filename)

        # Skip folders
        if not os.path.isfile(old_path):
            continue

        # Get extension
        _, ext = os.path.splitext(filename)

        # New filename
        new_name = f"no_cancer_{count}{ext}"

        new_path = os.path.join(folder, new_name)

        # Rename
        os.rename(old_path, new_path)

        print(f"✅ {filename}  -->  {new_name}")

        count += 1

print("\n🎉 All files renamed successfully!")