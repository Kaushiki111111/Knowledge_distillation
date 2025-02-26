# save this as combine_my_data.py
from generate_data import combine_files

input_dir = r"C:\Users\Kaushiki\Downloads\kd\Knowledge_distillation\data"
output_file = r"C:\Users\Kaushiki\Downloads\kd\Knowledge_distillation\data\combined_data.txt"

# Set max_size_mb to limit the file size if needed
combine_files(input_dir, output_file, max_size_mb=10)

print(f"Combined data saved to {output_file}")