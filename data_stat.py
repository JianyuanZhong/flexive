import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np

# Find all .jsonl files in the BIG-Bench-Mistake directory
jsonl_files = glob.glob('BIG-Bench-Mistake/*.jsonl')

# Dictionary to store mistake indices for each file
file_mistake_data = {}  # Will store lists of mistake indices including nulls as -1
all_mistake_data = []   # Will store all indices including nulls as -1

# Process each file
for file_path in jsonl_files:
    file_name = os.path.basename(file_path)
    mistake_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'mistake_index' in data:
                    # Represent null as -1 to include in the plot
                    if data['mistake_index'] is None:
                        mistake_data.append(-1)
                    else:
                        mistake_data.append(data['mistake_index'])
            except json.JSONDecodeError:
                continue
    
    if mistake_data:
        file_mistake_data[file_name] = mistake_data
        all_mistake_data.extend(mistake_data)

# Create directory for output plots
os.makedirs('plots', exist_ok=True)

# Plot distributions for each file
plt.figure(figsize=(15, 10))
for idx, (file_name, indices) in enumerate(file_mistake_data.items()):
    if not indices:
        continue
        
    # Calculate null count and valid count
    null_count = indices.count(-1)
    valid_indices = [idx for idx in indices if idx != -1]
    
    plt.subplot(3, 2, idx + 1)
    
    # Create custom bins that include -1 for null values
    if valid_indices:
        bins = list(np.arange(min(valid_indices), max(valid_indices) + 2) - 0.5)
        # Add a bin for null values if they exist
        if null_count > 0:
            bins = [-1.5, -0.5] + bins
    else:
        # If only null values
        bins = [-1.5, -0.5]
    
    plt.hist(indices, bins=bins, alpha=0.7)
    plt.xlabel('Mistake Index (-1 = No Mistake)')
    plt.ylabel('Frequency')
    plt.title(f'{file_name}\n({len(valid_indices)} with mistakes, {null_count} without mistakes)')
    plt.grid(True, alpha=0.3)
    
    # Add custom x-tick for null values
    plt.xticks(list(plt.xticks()[0]))
    locs, labels = plt.xticks()
    new_labels = ['No Mistake' if loc == -1 else str(int(loc)) for loc in locs]
    plt.xticks(locs, new_labels)
    
    # Save individual plot
    plt.figure(figsize=(10, 6))
    plt.hist(indices, bins=bins, alpha=0.7)
    plt.xlabel('Mistake Index (-1 = No Mistake)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Mistake Indices in {file_name}\n({len(valid_indices)} with mistakes, {null_count} without mistakes)')
    plt.grid(True, alpha=0.3)
    
    # Add custom x-tick for null values in individual plot too
    plt.xticks(list(plt.xticks()[0]))
    locs, labels = plt.xticks()
    new_labels = ['No Mistake' if loc == -1 else str(int(loc)) for loc in locs]
    plt.xticks(locs, new_labels)
    
    plt.tight_layout()
    plt.savefig(f'plots/{os.path.splitext(file_name)[0]}_mistake_distribution.png')
    plt.close()
    
    # Return to the main figure
    plt.figure(1)

plt.tight_layout()
plt.savefig('plots/all_files_mistake_distributions.png')

# Plot combined distribution
if all_mistake_data:
    plt.figure(figsize=(12, 7))
    
    # Calculate null count and valid count for combined data
    null_count = all_mistake_data.count(-1)
    valid_indices = [idx for idx in all_mistake_data if idx != -1]
    
    # Create custom bins that include -1 for null values
    if valid_indices:
        bins = list(np.arange(min(valid_indices), max(valid_indices) + 2) - 0.5)
        # Add a bin for null values if they exist
        if null_count > 0:
            bins = [-1.5, -0.5] + bins
    else:
        # If only null values
        bins = [-1.5, -0.5]
    
    plt.hist(all_mistake_data, bins=bins, alpha=0.7)
    plt.xlabel('Mistake Index (-1 = No Mistake)')
    plt.ylabel('Frequency')
    plt.title(f'Combined Distribution of Mistake Indices across All Files\n({len(valid_indices)} with mistakes, {null_count} without mistakes)')
    plt.grid(True, alpha=0.3)
    
    # Add custom x-tick for null values
    plt.xticks(list(plt.xticks()[0]))
    locs, labels = plt.xticks()
    new_labels = ['No Mistake' if loc == -1 else str(int(loc)) for loc in locs]
    plt.xticks(locs, new_labels)
    
    plt.tight_layout()
    plt.savefig('plots/combined_mistake_distribution.png')

# Print summary statistics
total_examples = len(all_mistake_data)
total_with_mistakes = len([idx for idx in all_mistake_data if idx != -1])
total_without_mistakes = all_mistake_data.count(-1)

print(f"Processed {len(jsonl_files)} files with {total_examples} total examples:")
print(f"- {total_with_mistakes} examples with mistakes ({total_with_mistakes/total_examples*100:.1f}%)")
print(f"- {total_without_mistakes} examples without mistakes ({total_without_mistakes/total_examples*100:.1f}%)")
print(f"Distribution plots saved to 'plots/' directory.") 