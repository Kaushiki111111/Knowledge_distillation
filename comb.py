import os

def combine_all_files(root_dir, output_file, max_size_mb=None):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        current_size = 0
        max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb else float('inf')
        
        # Walk through all directories and files
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.txt'):
                    file_path = os.path.join(dirpath, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            content = infile.read()
                            outfile.write(content)
                            outfile.write('\n\n')
                            
                            current_size += len(content.encode('utf-8'))
                            if current_size >= max_size_bytes:
                                print(f"Reached size limit of {max_size_mb} MB")
                                return
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    print(f"Combined data saved to {output_file}")
    print(f"Total size: {current_size / (1024*1024):.2f} MB")

# Use it like this
combine_all_files(r"C:\Users\Kaushiki\Downloads\kd\Knowledge_distillation\data", r"C:\Users\Kaushiki\Downloads\kd\Knowledge_distillation\combined_data.txt", max_size_mb=100)