import pandas as pd
from collections import defaultdict

def convert_grain_annotations_to_training_format(csv_file_path, output_csv_path='grain_training_data.csv'):
    df = pd.read_csv(csv_file_path)

    print(f"Total annotations: {len(df)}")
    print(f"Unique images: {df['filename'].nunique()}")
    print(f"Average annotations per image: {len(df) / df['filename'].nunique():.2f}")

    def categorize_grain(class_name):
        if pd.isna(class_name):
            return 'unknown'
        if class_name == 'healthy seed':
            return 'good'
        elif class_name in ['bad seed', 'impurity']:
            return 'bad'
        else:
            print(f"Warning: Unknown class '{class_name}' - treating as 'bad'")
            return 'bad'

    df['grain_quality'] = df['class'].apply(categorize_grain)

    image_stats = defaultdict(lambda: {'good': 0, 'bad': 0, 'unknown': 0})
    for _, row in df.iterrows():
        filename = row['filename']
        quality = row['grain_quality']
        image_stats[filename][quality] += 1

    training_data = []
    for filename, stats in image_stats.items():
        total_count = stats['good'] + stats['bad'] + stats['unknown']
        good_count = stats['good']
        bad_count = stats['bad'] + stats['unknown']
        training_data.append({
            'image_name': filename,
            'count': total_count,
            'good': good_count,
            'bad': bad_count
        })

    training_df = pd.DataFrame(training_data)
    training_df = training_df.sort_values('image_name').reset_index(drop=True)

    print(f"\nTraining dataset summary:")
    print(f"Total images: {len(training_df)}")
    print(f"Average grains per image: {training_df['count'].mean():.2f}")
    print(f"Average good grains per image: {training_df['good'].mean():.2f}")
    print(f"Average bad grains per image: {training_df['bad'].mean():.2f}")

    training_df['quality_label'] = training_df.apply(
        lambda row: 'good_dominant' if row['good'] > row['bad'] else 'bad_dominant', axis=1
    )
    print(f"\nQuality distribution:")
    print(training_df['quality_label'].value_counts())

    training_df.drop('quality_label', axis=1, inplace=True)
    training_df.to_csv(output_csv_path, index=False)
    print(f"\nTraining data saved to: {output_csv_path}")
    print(f"\nSample training data:")
    print(training_df.head(10))

    return training_df

def create_image_quality_report(training_df, output_path='image_quality_report.csv'):
    training_df = training_df.copy()
    training_df['good_ratio'] = training_df['good'] / training_df['count']
    training_df['bad_ratio'] = training_df['bad'] / training_df['count']
    training_df['quality_score'] = training_df['good_ratio'] - training_df['bad_ratio']
    training_df['dominant_quality'] = training_df.apply(
        lambda row: 'good' if row['good'] > row['bad'] else 'bad', axis=1
    )

    training_df = training_df.sort_values('quality_score', ascending=False)
    training_df.to_csv(output_path, index=False)
    print(f"Detailed quality report saved to: {output_path}")
    print(f"\nBest quality images (top 10):")
    print(training_df.head(10)[['image_name', 'count', 'good', 'bad', 'quality_score']])
    print(f"\nWorst quality images (bottom 10):")
    print(training_df.tail(10)[['image_name', 'count', 'good', 'bad', 'quality_score']])

if __name__ == "__main__":
    input_file = "_annotations.csv"
    output_file = "grain_training_data1.csv"

    print("Starting grain dataset conversion...")
    print("=" * 60)
    training_df = convert_grain_annotations_to_training_format(input_file, output_file)

    print("\n" + "=" * 60)
    print("STEP 2: Creating detailed quality report")
    create_image_quality_report(training_df)

    print("\nConversion completed successfully!")
