"""Preprocess facecap dataset for training."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.image_quality import ImageQualityAssessor, enhance_image_quality


def process_single_image(args):
    """Process a single image (for multiprocessing)."""
    image_path, output_dir, target_size, enhance, quality_threshold = args
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Assess quality
        assessor = ImageQualityAssessor()
        quality_scores = assessor.assess_overall_quality(image)
        
        # Skip low quality images if threshold is set
        if quality_threshold and quality_scores['overall_score'] < quality_threshold:
            return {
                'path': str(image_path),
                'status': 'skipped_low_quality',
                'quality_score': quality_scores['overall_score']
            }
        
        # Enhance image if requested
        if enhance:
            image = enhance_image_quality(image)
        
        # Resize image
        if target_size:
            image = cv2.resize(image, target_size)
        
        # Create output path
        relative_path = image_path.relative_to(image_path.parents[1])  # Relative to facecap root
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed image
        cv2.imwrite(str(output_path), image)
        
        return {
            'path': str(image_path),
            'output_path': str(output_path),
            'status': 'processed',
            'quality_score': quality_scores['overall_score'],
            'quality_level': quality_scores['quality_level'],
            'original_size': f"{image.shape[1]}x{image.shape[0]}",
            'enhanced': enhance
        }
        
    except Exception as e:
        return {
            'path': str(image_path),
            'status': 'error',
            'error': str(e)
        }


def preprocess_facecap_dataset(
    data_root: Path,
    output_dir: Path,
    target_size: tuple = (112, 112),
    enhance_images: bool = False,
    quality_threshold: float = None,
    num_workers: int = 4,
    max_images_per_person: int = None
):
    """
    Preprocess the entire facecap dataset.
    
    Args:
        data_root: Root directory of facecap dataset
        output_dir: Output directory for processed images
        target_size: Target image size (width, height)
        enhance_images: Whether to enhance image quality
        quality_threshold: Minimum quality threshold (0-1)
        num_workers: Number of worker processes
        max_images_per_person: Maximum images per person (for testing)
    """
    print(f"Preprocessing facecap dataset...")
    print(f"Input: {data_root}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}")
    print(f"Enhance images: {enhance_images}")
    print(f"Quality threshold: {quality_threshold}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_paths = []
    
    # Process each person directory
    for person_dir in sorted(data_root.glob("[0-9]*")):
        if person_dir.is_dir():
            person_images = list(person_dir.glob("*.jpg"))
            
            # Limit images per person if specified
            if max_images_per_person:
                person_images = person_images[:max_images_per_person]
            
            image_paths.extend(person_images)
    
    print(f"Found {len(image_paths)} images to process")
    
    if not image_paths:
        print("No images found!")
        return
    
    # Prepare arguments for multiprocessing
    process_args = [
        (img_path, output_dir, target_size, enhance_images, quality_threshold)
        for img_path in image_paths
    ]
    
    # Process images
    results = []
    
    if num_workers > 1:
        print(f"Processing with {num_workers} workers...")
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image, process_args),
                total=len(process_args),
                desc="Processing images"
            ))
    else:
        print("Processing sequentially...")
        for args in tqdm(process_args, desc="Processing images"):
            results.append(process_single_image(args))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Analyze results
    processed_count = sum(1 for r in results if r['status'] == 'processed')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped_low_quality')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\nProcessing completed:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (low quality): {skipped_count}")
    print(f"  Errors: {error_count}")
    
    # Calculate quality statistics
    quality_scores = [r['quality_score'] for r in results if 'quality_score' in r]
    if quality_scores:
        print(f"\nQuality statistics:")
        print(f"  Mean quality: {np.mean(quality_scores):.3f}")
        print(f"  Min quality: {np.min(quality_scores):.3f}")
        print(f"  Max quality: {np.max(quality_scores):.3f}")
    
    # Save processing report
    report_path = output_dir / 'preprocessing_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'total_images': len(image_paths),
                'processed': processed_count,
                'skipped': skipped_count,
                'errors': error_count,
                'quality_stats': {
                    'mean': float(np.mean(quality_scores)) if quality_scores else None,
                    'min': float(np.min(quality_scores)) if quality_scores else None,
                    'max': float(np.max(quality_scores)) if quality_scores else None,
                }
            },
            'settings': {
                'target_size': target_size,
                'enhance_images': enhance_images,
                'quality_threshold': quality_threshold,
                'num_workers': num_workers
            },
            'results': results
        }, f, indent=2)
    
    print(f"\nProcessing report saved to: {report_path}")
    
    # Copy dataset split files
    for split_file in ['train_list.txt', 'val_list.txt', 'test_list.txt', 'labels.txt']:
        src_path = data_root / split_file
        dst_path = output_dir / split_file
        
        if src_path.exists():
            # For image lists, update paths to point to processed images
            if split_file.endswith('_list.txt'):
                update_split_file(src_path, dst_path, processed_count > 0)
            else:
                # Copy labels.txt as-is
                dst_path.write_text(src_path.read_text())
            
            print(f"Updated {split_file}")


def update_split_file(src_path: Path, dst_path: Path, has_processed_images: bool):
    """Update split file paths to point to processed images."""
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Keep the same format but ensure paths are correct
            updated_lines.append(line + '\n')
    
    with open(dst_path, 'w') as f:
        f.writelines(updated_lines)


def create_quality_report(data_root: Path, output_path: Path, sample_size: int = 1000):
    """Create a quality assessment report for the dataset."""
    print(f"Creating quality report for {data_root}...")
    
    # Find sample images
    image_paths = []
    for person_dir in sorted(data_root.glob("[0-9]*"))[:10]:  # First 10 people
        if person_dir.is_dir():
            person_images = list(person_dir.glob("*.jpg"))[:sample_size//10]
            image_paths.extend(person_images)
    
    if len(image_paths) > sample_size:
        image_paths = image_paths[:sample_size]
    
    print(f"Analyzing {len(image_paths)} sample images...")
    
    assessor = ImageQualityAssessor()
    quality_results = []
    
    for image_path in tqdm(image_paths, desc="Assessing quality"):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            scores = assessor.assess_overall_quality(image)
            scores['image_path'] = str(image_path)
            scores['image_size'] = f"{image.shape[1]}x{image.shape[0]}"
            quality_results.append(scores)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Analyze results
    if quality_results:
        overall_scores = [r['overall_score'] for r in quality_results]
        quality_levels = [r['quality_level'] for r in quality_results]
        
        # Count quality levels
        level_counts = {}
        for level in quality_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        report = {
            'summary': {
                'total_analyzed': len(quality_results),
                'mean_quality': float(np.mean(overall_scores)),
                'std_quality': float(np.std(overall_scores)),
                'min_quality': float(np.min(overall_scores)),
                'max_quality': float(np.max(overall_scores)),
                'quality_distribution': level_counts
            },
            'detailed_results': quality_results
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nQuality Report:")
        print(f"  Mean quality: {report['summary']['mean_quality']:.3f}")
        print(f"  Quality distribution: {level_counts}")
        print(f"  Report saved to: {output_path}")
        
        return report
    
    return None


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess facecap dataset")
    parser.add_argument('--data-root', type=str, default='../facecap',
                        help='Root directory of facecap dataset')
    parser.add_argument('--output-dir', type=str, default='data/processed_facecap',
                        help='Output directory for processed images')
    parser.add_argument('--target-size', type=int, nargs=2, default=[112, 112],
                        help='Target image size (width height)')
    parser.add_argument('--enhance', action='store_true',
                        help='Enhance image quality')
    parser.add_argument('--quality-threshold', type=float, default=None,
                        help='Minimum quality threshold (0-1)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes')
    parser.add_argument('--max-images-per-person', type=int, default=None,
                        help='Maximum images per person (for testing)')
    parser.add_argument('--quality-report-only', action='store_true',
                        help='Only generate quality report')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        return 1
    
    if args.quality_report_only:
        # Generate quality report only
        report_path = output_dir / 'quality_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        create_quality_report(data_root, report_path)
    else:
        # Full preprocessing
        preprocess_facecap_dataset(
            data_root=data_root,
            output_dir=output_dir,
            target_size=tuple(args.target_size),
            enhance_images=args.enhance,
            quality_threshold=args.quality_threshold,
            num_workers=args.num_workers,
            max_images_per_person=args.max_images_per_person
        )
    
    return 0


if __name__ == '__main__':
    exit(main())