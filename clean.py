"""
Quick fix script to clean the grain_training_data.csv file
This addresses the NaN filename issue and other data problems
"""

import pandas as pd
import os

def fix_grain_dataset(input_file='grain_training_data.csv', output_file='grain_training_data_fixed.csv'):
    """
    Fix the grain training dataset by removing problematic rows
    """
    
    print("🔧 QUICK FIX: Cleaning grain training dataset")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        print("Please run the dataset converter first.")
        return False
    
    # Load the dataset
    print(f"📁 Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"📊 Original dataset: {len(df)} rows")
    
    # Identify problems
    nan_filenames = df['image_name'].isna().sum()
    empty_filenames = (df['image_name'].astype(str).str.strip() == '').sum()
    invalid_counts = df[['count', 'good', 'bad']].isna().any(axis=1).sum()
    mismatched_counts = (df['count'] != (df['good'] + df['bad'])).sum()
    
    print(f"\n🔍 Problems found:")
    print(f"   📷 NaN filenames: {nan_filenames}")
    print(f"   📝 Empty filenames: {empty_filenames}")
    print(f"   🔢 Invalid numeric values: {invalid_counts}")
    print(f"   ⚖️  Mismatched counts: {mismatched_counts}")
    
    # Clean the dataset step by step
    print(f"\n🧹 Cleaning dataset...")
    
    # Step 1: Remove NaN filenames
    df_clean = df.dropna(subset=['image_name'])
    print(f"   After removing NaN filenames: {len(df_clean)} rows")
    
    # Step 2: Remove empty filenames
    df_clean = df_clean[df_clean['image_name'].astype(str).str.strip() != '']
    print(f"   After removing empty filenames: {len(df_clean)} rows")
    
    # Step 3: Remove 'nan' string filenames
    df_clean = df_clean[df_clean['image_name'].astype(str) != 'nan']
    print(f"   After removing 'nan' filenames: {len(df_clean)} rows")
    
    # Step 4: Remove rows with invalid numeric values
    df_clean = df_clean.dropna(subset=['count', 'good', 'bad'])
    print(f"   After removing NaN numeric values: {len(df_clean)} rows")
    
    # Step 5: Ensure all numeric values are non-negative
    df_clean = df_clean[(df_clean['count'] >= 0) & (df_clean['good'] >= 0) & (df_clean['bad'] >= 0)]
    print(f"   After removing negative values: {len(df_clean)} rows")
    
    # Step 6: Ensure count = good + bad
    df_clean = df_clean[df_clean['count'] == (df_clean['good'] + df_clean['bad'])]
    print(f"   After fixing count mismatches: {len(df_clean)} rows")
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    # Summary
    removed_rows = len(df) - len(df_clean)
    print(f"\n📊 Cleaning summary:")
    print(f"   🔴 Removed: {removed_rows} problematic rows")
    print(f"   🟢 Remaining: {len(df_clean)} clean rows")
    print(f"   📈 Data retention: {len(df_clean)/len(df)*100:.1f}%")
    
    if len(df_clean) == 0:
        print("❌ ERROR: No valid data remaining after cleaning!")
        return False
    
    # Quality analysis of cleaned data
    print(f"\n📈 Cleaned dataset analysis:")
    print(f"   📷 Average grains per image: {df_clean['count'].mean():.1f}")
    print(f"   🟢 Average good grains: {df_clean['good'].mean():.1f}")
    print(f"   🔴 Average bad grains: {df_clean['bad'].mean():.1f}")
    
    good_dominant = (df_clean['good'] > df_clean['bad']).sum()
    bad_dominant = (df_clean['bad'] > df_clean['good']).sum()
    print(f"   ⚖️  Good dominant images: {good_dominant}")
    print(f"   ⚖️  Bad dominant images: {bad_dominant}")
    
    # Save cleaned dataset
    df_clean.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_file}")
    
    # Show sample of cleaned data
    print(f"\n📋 Sample of cleaned data:")
    print(df_clean.head(5)[['image_name', 'count', 'good', 'bad']])
    
    return True

def backup_original_file(original_file='grain_training_data.csv'):
    """
    Create a backup of the original file
    """
    if os.path.exists(original_file):
        backup_file = original_file.replace('.csv', '_backup.csv')
        import shutil
        shutil.copy2(original_file, backup_file)
        print(f"💾 Backup created: {backup_file}")
        return backup_file
    return None

def main():
    """
    Main function to fix the dataset
    """
    
    print("🚀 GRAIN DATASET QUICK FIX")
    print("This script will clean your grain_training_data.csv file")
    print("=" * 50)
    
    original_file = 'grain_training_data.csv'
    fixed_file = 'grain_training_data_fixed.csv'
    
    # Create backup
    backup_file = backup_original_file(original_file)
    
    # Fix the dataset
    success = fix_grain_dataset(original_file, fixed_file)
    
    if success:
        # Replace original with fixed version
        import shutil
        shutil.move(fixed_file, original_file)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Original file cleaned and ready for training")
        print(f"💾 Backup saved as: {backup_file}")
        print(f"\n🚀 Next steps:")
        print(f"1. Run: python complete_pipeline.py")
        print(f"2. Or run: python rtx3050_optimized.py (for RTX 3050)")
        print(f"3. Your dataset is now ready for training!")
        
    else:
        print(f"\n❌ FAILED to clean dataset")
        print(f"Please check your original annotation file and try converting again.")

if __name__ == "__main__":
    main()