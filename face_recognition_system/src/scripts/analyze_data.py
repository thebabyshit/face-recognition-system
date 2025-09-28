"""Script to analyze the facecap dataset."""

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import analyze_dataset


def main():
    """Main function to analyze dataset."""
    parser = argparse.ArgumentParser(description="Analyze facecap dataset")
    parser.add_argument(
        "--data-root",
        type=str,
        default="facecap",
        help="Root directory containing facecap data"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save analysis results (JSON format)"
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing dataset at: {args.data_root}")
    
    # Analyze dataset
    try:
        stats = analyze_dataset(args.data_root)
        
        # Print results
        print("\n" + "="*50)
        print("DATASET ANALYSIS RESULTS")
        print("="*50)
        
        for split, split_stats in stats.items():
            print(f"\n{split.upper()} SET:")
            print("-" * 20)
            
            if "error" in split_stats:
                print(f"  Error: {split_stats['error']}")
                continue
            
            print(f"  Total samples: {split_stats['num_samples']:,}")
            print(f"  Number of classes: {split_stats['num_classes']}")
            print(f"  Min samples per class: {split_stats['min_samples_per_class']}")
            print(f"  Max samples per class: {split_stats['max_samples_per_class']}")
            print(f"  Avg samples per class: {split_stats['avg_samples_per_class']:.1f}")
        
        # Calculate total statistics
        total_samples = sum(
            s.get('num_samples', 0) for s in stats.values() 
            if 'error' not in s
        )
        print(f"\nTOTAL SAMPLES ACROSS ALL SPLITS: {total_samples:,}")
        
        # Save results if output file specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())