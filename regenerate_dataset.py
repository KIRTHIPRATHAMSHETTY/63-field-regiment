from src.dataset_generator import generate_dataset

if __name__ == '__main__':
    print("Regenerating dataset with new extended parameters...")
    df = generate_dataset(20000, 'data/barbarik_dataset_new.csv')
    print(f"\nDataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['outcome'].value_counts(normalize=True))
    print(f"\nFirst few rows:")
    print(df.head())
    print("\nDataset saved to data/barbarik_dataset_new.csv")
    print("Please rename it to barbarik_dataset.csv after stopping the server.")
