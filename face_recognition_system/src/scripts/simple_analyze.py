"""Simple script to analyze the facecap dataset without heavy dependencies."""

import argparse
from pathlib import Path
from collections import defaultdict


def analyze_facecap_dataset(data_root):
    """Analyze facecap dataset structure."""
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    stats = {}
    
    # Check for required files
    required_files = ["labels.txt", "train_list.txt", "val_list.txt", "test_list.txt"]
    missing_files = []
    
    for file_name in required_files:
        file_path = data_root / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return None
    
    # Load labels
    labels_file = data_root / "labels.txt"
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    stats['total_classes'] = len(labels)
    
    # Analyze each split
    for split in ["train", "val", "test"]:
        split_file = data_root / f"{split}_list.txt"
        
        class_counts = defaultdict(int)
        total_samples = 0
        
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        label = int(parts[1])
                        class_counts[label] += 1
                        total_samples += 1
        
        stats[split] = {
            'num_samples': total_samples,
            'num_classes': len(class_counts),
            'class_counts': dict(class_counts),
            'min_samples_per_class': min(class_counts.values()) if class_counts else 0,
            'max_samples_per_class': max(class_counts.values()) if class_counts else 0,
            'avg_samples_per_class': sum(class_counts.values()) / len(class_counts) if class_counts else 0,
        }
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze facecap dataset")
    parser.add_argument(
        "--data-root",
        type=str,
        default="facecap",
        help="Root directory containing facecap data"
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing dataset at: {args.data_root}")
    
    try:
        stats = analyze_facecap_dataset(args.data_root)
        
        if stats is None:
            return 1
        
        # Print results
        print("\n" + "="*50)
        print("FACECAP DATASET ANALYSIS")
        print("="*50)
        
        print(f"\nTotal classes in labels.txt: {stats['total_classes']}")
        
        for split in ["train", "val", "test"]:
            if split in stats:
                split_stats = stats[split]
                print(f"\n{split.upper()} SET:")
                print("-" * 20)
                print(f"  Total samples: {split_stats['num_samples']:,}")
                print(f"  Number of classes: {split_stats['num_classes']}")
                print(f"  Min samples per class: {split_stats['min_samples_per_class']}")
                print(f"  Max samples per class: {split_stats['max_samples_per_class']}")
                print(f"  Avg samples per class: {split_stats['avg_samples_per_class']:.1f}")
        
        # Calculate totals
        total_samples = sum(stats[split]['num_samples'] for split in ["train", "val", "test"] if split in stats)
        print(f"\nTOTAL SAMPLES: {total_samples:,}")
        
        # Check data distribution
        print(f"\nDATA DISTRIBUTION:")
        print("-" * 20)
        for split in ["train", "val", "test"]:
            if split in stats:
                percentage = (stats[split]['num_samples'] / total_samples) * 100
                print(f"  {split}: {percentage:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())