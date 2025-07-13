import pandas as pd
import numpy as np
from collections import defaultdict
import os

def convert_grain_annotations_to_training_format(excel_file_path, output_csv_path='grain_training_data.csv'):
    """
    Convert grain quality annotation data to ResNet-50 training format
    
    Args:
        excel_file_path: Path to the _annotations.csv.xlsx file
        output_csv_path: Path for the output CSV file
    
    Returns:
        DataFrame with columns: image_name, count, good, bad
    """
    
    # Read the Excel file
    df = pd.read_excel(excel_file_path)
    
    # Print basic info about the dataset
    print(f"Total annotations: {len(df)}")
    print(f"Unique images: {df['filename'].nunique()}")
    print(f"Average annotations per image: {len(df) / df['filename'].nunique():.2f}")
    
    # Define class mappings
    # Good/Healthy grain classes
    good_classes = {
        'healthy', 'healthy-1-1', 'healthy-1-2', 'healthy-2-1', 'healthy-2-2', 
        'good', 'Healthy'
    }
    
    # Bad/Damaged grain classes  
    bad_classes = {
        'bad', 'damage', 'Damage', 'Damage-Discolor', 'Shriveled', 'shriveled',
        'Fungus', 'Fungus-1', 'Fungus-2', 'fungus', 'Broken', 'broken',
        'Weeveled', 'weeveled', 'Immature', 'immature'
    }
    
    # Foreign material classes (treated as bad)
    foreign_material_classes = {
        'Inorganic foreign material', 'fm-inorganic',
        'Organic foreign material', 'fm-organic',
        'Other Edible Seeds', 'corn'
    }
    
    # Numeric classes (need manual inspection)
    numeric_classes = {0, 1, 2}
    
    print("\nClass distribution:")
    class_counts = df['class'].value_counts()
    print(class_counts)
    
    # Function to categorize grain quality
    def categorize_grain(class_name):
        if pd.isna(class_name):
            return 'unknown'
        
        if class_name in good_classes:
            return 'good'
        elif class_name in bad_classes or class_name in foreign_material_classes:
            return 'bad'
        elif class_name in numeric_classes:
            # You may need to adjust this based on your specific numeric class meanings
            # Assuming: 0=bad, 1=good, 2=bad (adjust as needed)
            if class_name == 1:
                return 'good'
            else:
                return 'bad'
        else:
            # For any unhandled classes, print them for manual review
            print(f"Warning: Unknown class '{class_name}' - treating as 'bad'")
            return 'bad'
    
    # Apply categorization
    df['grain_quality'] = df['class'].apply(categorize_grain)
    
    # Group by image and count grains
    image_stats = defaultdict(lambda: {'good': 0, 'bad': 0, 'unknown': 0})
    
    for _, row in df.iterrows():
        filename = row['filename']
        quality = row['grain_quality']
        image_stats[filename][quality] += 1
    
    # Convert to training format
    training_data = []
    for filename, stats in image_stats.items():
        total_count = stats['good'] + stats['bad'] + stats['unknown']
        good_count = stats['good']
        bad_count = stats['bad'] + stats['unknown']  # Treat unknown as bad
        
        training_data.append({
            'image_name': filename,
            'count': total_count,
            'good': good_count,
            'bad': bad_count
        })
    
    # Create DataFrame
    training_df = pd.DataFrame(training_data)
    
    # Sort by image name for consistency
    training_df = training_df.sort_values('image_name').reset_index(drop=True)
    
    # Print summary statistics
    print(f"\nTraining dataset summary:")
    print(f"Total images: {len(training_df)}")
    print(f"Average grains per image: {training_df['count'].mean():.2f}")
    print(f"Average good grains per image: {training_df['good'].mean():.2f}")
    print(f"Average bad grains per image: {training_df['bad'].mean():.2f}")
    print(f"Images with more good grains: {sum(training_df['good'] > training_df['bad'])}")
    print(f"Images with more bad grains: {sum(training_df['bad'] > training_df['good'])}")
    
    # Quality distribution
    training_df['quality_label'] = training_df.apply(
        lambda row: 'good_dominant' if row['good'] > row['bad'] else 'bad_dominant', axis=1
    )
    print(f"\nQuality distribution:")
    print(training_df['quality_label'].value_counts())
    
    # Save to CSV
    training_df.drop('quality_label', axis=1, inplace=True)  # Remove helper column
    training_df.to_csv(output_csv_path, index=False)
    print(f"\nTraining data saved to: {output_csv_path}")
    
    # Display sample data
    print(f"\nSample training data:")
    print(training_df.head(10))
    
    return training_df

def analyze_class_distribution(excel_file_path):
    """
    Analyze the class distribution in detail to help with categorization
    """
    df = pd.read_excel(excel_file_path)
    
    print("Detailed class analysis:")
    print("=" * 50)
    
    class_counts = df['class'].value_counts()
    
    # Categorize classes for review
    good_classes = {
        'healthy', 'healthy-1-1', 'healthy-1-2', 'healthy-2-1', 'healthy-2-2', 
        'good', 'Healthy'
    }
    
    bad_classes = {
        'bad', 'damage', 'Damage', 'Damage-Discolor', 'Shriveled', 'shriveled',
        'Fungus', 'Fungus-1', 'Fungus-2', 'fungus', 'Broken', 'broken',
        'Weeveled', 'weeveled', 'Immature', 'immature'
    }
    
    foreign_material_classes = {
        'Inorganic foreign material', 'fm-inorganic',
        'Organic foreign material', 'fm-organic',
        'Other Edible Seeds', 'corn'
    }
    
    print("GOOD/HEALTHY CLASSES:")
    for class_name in class_counts.index:
        if class_name in good_classes:
            print(f"  {class_name}: {class_counts[class_name]}")
    
    print("\nBAD/DAMAGED CLASSES:")
    for class_name in class_counts.index:
        if class_name in bad_classes:
            print(f"  {class_name}: {class_counts[class_name]}")
    
    print("\nFOREIGN MATERIAL CLASSES:")
    for class_name in class_counts.index:
        if class_name in foreign_material_classes:
            print(f"  {class_name}: {class_counts[class_name]}")
    
    print("\nNUMERIC/OTHER CLASSES:")
    for class_name in class_counts.index:
        if (class_name not in good_classes and 
            class_name not in bad_classes and 
            class_name not in foreign_material_classes):
            print(f"  {class_name}: {class_counts[class_name]}")

def create_image_quality_report(training_df, output_path='image_quality_report.csv'):
    """
    Create a detailed report of image quality distribution
    """
    # Add quality metrics
    training_df = training_df.copy()
    training_df['good_ratio'] = training_df['good'] / training_df['count']
    training_df['bad_ratio'] = training_df['bad'] / training_df['count']
    training_df['quality_score'] = training_df['good_ratio'] - training_df['bad_ratio']
    training_df['dominant_quality'] = training_df.apply(
        lambda row: 'good' if row['good'] > row['bad'] else 'bad', axis=1
    )
    
    # Sort by quality score
    training_df = training_df.sort_values('quality_score', ascending=False)
    
    # Save detailed report
    training_df.to_csv(output_path, index=False)
    print(f"Detailed quality report saved to: {output_path}")
    
    # Print summary
    print(f"\nQuality Score Distribution:")
    print(f"Best quality images (top 10):")
    print(training_df.head(10)[['image_name', 'count', 'good', 'bad', 'quality_score']])
    
    print(f"\nWorst quality images (bottom 10):")
    print(training_df.tail(10)[['image_name', 'count', 'good', 'bad', 'quality_score']])

# Main execution
if __name__ == "__main__":
    # File paths - update these as needed
    input_file = "_annotations.csv.xlsx"
    output_file = "grain_training_data.csv"
    
    print("Starting grain dataset conversion...")
    print("=" * 60)
    
    # Step 1: Analyze class distribution
    print("STEP 1: Analyzing class distribution")
    analyze_class_distribution(input_file)
    
    print("\n" + "=" * 60)
    
    # Step 2: Convert to training format
    print("STEP 2: Converting to training format")
    training_df = convert_grain_annotations_to_training_format(input_file, output_file)
    
    print("\n" + "=" * 60)
    
    # Step 3: Create detailed quality report
    print("STEP 3: Creating detailed quality report")
    create_image_quality_report(training_df)
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print(f"Files created:")
    print(f"  - {output_file}: Training data for ResNet-50")
    print(f"  - image_quality_report.csv: Detailed quality analysis")
    
    print(f"\nNext steps:")
    print(f"1. Review the class categorization above")
    print(f"2. Adjust class mappings in the script if needed")
    print(f"3. Use '{output_file}' with the ResNet-50 training script")