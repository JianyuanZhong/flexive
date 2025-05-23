import os
import json
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
import shutil

def create_output_dirs():
    """Create output directories for train and test data"""
    os.makedirs('BIG-Bench-Mistake-Train', exist_ok=True)
    os.makedirs('BIG-Bench-Mistake-Test', exist_ok=True)
    os.makedirs('plots/split_validation', exist_ok=True)

def stratified_split_by_mistake_index(data, test_size=0.2, random_state=42):
    """
    Split data while preserving the distribution of mistake_index.
    Returns train and test sets as lists of dictionaries.
    """
    # Group examples by mistake_index
    grouped_data = defaultdict(list)
    for example in data:
        mistake_index = example.get('mistake_index', None)
        # Store examples grouped by their mistake_index
        grouped_data[mistake_index].append(example)
    
    train_data, test_data = [], []
    
    # For each mistake_index group, perform a split
    for mistake_index, examples in grouped_data.items():
        # Handle very small groups (less than 5 examples)
        if len(examples) < 5:
            # For tiny groups, just include all in training to avoid empty groups
            train_data.extend(examples)
            continue
        
        # Perform the split for this group
        group_train, group_test = train_test_split(
            examples, 
            test_size=test_size,
            random_state=random_state
        )
        
        train_data.extend(group_train)
        test_data.extend(group_test)
    
    # Shuffle both datasets to avoid any ordering bias
    random.seed(random_state)
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    return train_data, test_data

def plot_distribution_comparison(train_indices, test_indices, filename):
    """Plot to compare mistake index distribution between train and test sets"""
    plt.figure(figsize=(12, 7))
    
    # Create bins that include all indices
    all_indices = train_indices + test_indices
    valid_indices = [idx for idx in all_indices if idx != -1]
    
    if not valid_indices:
        bins = [-1.5, -0.5, 0.5]
    else:
        bins = list(np.arange(min(min(valid_indices), -1), max(valid_indices) + 2) - 0.5)
        if -1 in all_indices and -1.5 not in bins:
            bins = [-1.5, -0.5] + [b for b in bins if b > 0]
    
    # Plot histograms side by side
    plt.hist(
        [train_indices, test_indices],
        bins=bins,
        alpha=0.7,
        label=['Train', 'Test'],
        density=True  # Normalize to show proportions instead of counts
    )
    
    plt.xlabel('Mistake Index (-1 = No Mistake)')
    plt.ylabel('Normalized Frequency')
    plt.title(f'Comparison of Mistake Index Distribution:\nTrain vs Test for {filename}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add custom x-tick for null values
    plt.xticks(list(plt.xticks()[0]))
    locs, labels = plt.xticks()
    new_labels = ['No Mistake' if loc == -1 else str(int(loc)) for loc in locs]
    plt.xticks(locs, new_labels)
    
    plt.tight_layout()
    plt.savefig(f'plots/split_validation/{os.path.splitext(filename)[0]}_split_distribution.png')
    plt.close()

def main(test_size=0.2, random_state=42):
    """Main function to split data"""
    create_output_dirs()
    
    # Find all .jsonl files in the BIG-Bench-Mistake directory
    jsonl_files = glob.glob('BIG-Bench-Mistake/*.jsonl')
    
    if not jsonl_files:
        print("No JSONL files found in BIG-Bench-Mistake directory!")
        return
    
    # Process each file independently to maintain file structure
    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        # Load all data from the current file
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    data.append(example)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON in {file_name}")
                    continue
        
        if not data:
            print(f"No valid data found in {file_name}, skipping.")
            continue
        
        # Split the data while preserving mistake_index distribution
        train_data, test_data = stratified_split_by_mistake_index(
            data, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Extract mistake indices for validation plots
        train_indices = [ex.get('mistake_index', None) for ex in train_data]
        test_indices = [ex.get('mistake_index', None) for ex in test_data]
        
        # Convert None to -1 for plotting
        train_indices = [-1 if idx is None else idx for idx in train_indices]
        test_indices = [-1 if idx is None else idx for idx in test_indices]
        
        # Create validation plots
        plot_distribution_comparison(train_indices, test_indices, file_name)
        
        # Write the split data to new files
        train_file = os.path.join('BIG-Bench-Mistake-Train', file_name)
        test_file = os.path.join('BIG-Bench-Mistake-Test', file_name)
        
        with open(train_file, 'w') as f:
            for example in train_data:
                f.write(json.dumps(example) + '\n')
        
        with open(test_file, 'w') as f:
            for example in test_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"Split {file_name}: {len(train_data)} training examples, {len(test_data)} testing examples")
        
        # Calculate and print distribution stats
        train_with_mistakes = len([idx for idx in train_indices if idx != -1])
        test_with_mistakes = len([idx for idx in test_indices if idx != -1])
        
        print(f"  Training: {train_with_mistakes}/{len(train_indices)} with mistakes ({train_with_mistakes/len(train_indices)*100:.1f}%)")
        print(f"  Testing: {test_with_mistakes}/{len(test_indices)} with mistakes ({test_with_mistakes/len(test_indices)*100:.1f}%)")
    
    # Create combined dataset files (optional)
    print("\nCreating combined dataset files...")
    combine_split_files('BIG-Bench-Mistake-Train', 'combined_train.jsonl')
    combine_split_files('BIG-Bench-Mistake-Test', 'combined_test.jsonl')
    
    print("\nData preparation complete!")
    print(f"- Training data saved to BIG-Bench-Mistake-Train/")
    print(f"- Testing data saved to BIG-Bench-Mistake-Test/")
    print(f"- Distribution validation plots saved to plots/split_validation/")

def combine_split_files(directory, output_filename):
    """Combine all JSONL files in a directory into a single file"""
    jsonl_files = glob.glob(f'{directory}/*.jsonl')
    
    if not jsonl_files:
        print(f"No files found in {directory}, skipping combined file creation.")
        return
    
    with open(os.path.join(directory, output_filename), 'w') as outfile:
        for file_path in jsonl_files:
            with open(file_path, 'r') as infile:
                shutil.copyfileobj(infile, outfile)
    
    print(f"Combined {len(jsonl_files)} files into {directory}/{output_filename}")

if __name__ == "__main__":
    # You can adjust these parameters as needed
    main(test_size=0.2, random_state=42) 