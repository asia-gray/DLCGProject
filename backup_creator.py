 import shutil

original_file_path = 'art_tensors.csv'
copy_file_path = 'art_tensors_final.csv'

shutil.copyfile(original_file_path, copy_file_path)

print("Copy of CSV file created successfully.")