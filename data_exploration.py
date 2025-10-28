import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Data Exploration - Instagram Engagement")
print("-" * 40)

# Create sample dataset (500 posts)
# Later we'll use real data from APIs or Kaggle
np.random.seed(42)

n_posts = 500

# Generate realistic Instagram post data
data = {
    'caption_length': np.random.randint(10, 300, n_posts),
    'hashtag_count': np.random.randint(0, 30, n_posts),
    'emoji_count': np.random.randint(0, 10, n_posts),
    'has_question': np.random.choice([0, 1], n_posts),
    'posting_hour': np.random.randint(6, 23, n_posts),
    'likes': np.random.randint(50, 5000, n_posts),
    'comments': np.random.randint(5, 500, n_posts),
}

df = pd.DataFrame(data)

# Calculate total engagement (our target)
df['engagement'] = df['likes'] + (df['comments'] * 3)

print(f"\nDataset: {len(df)} Instagram posts")
print("\nFirst 5 posts:")
print(df.head())

print("\n\nStatistics:")
print(df.describe())

print("\n\nKey Insights:")
print(f"Average engagement: {df['engagement'].mean():.0f}")
print(f"Average caption length: {df['caption_length'].mean():.0f} chars")
print(f"Average hashtags: {df['hashtag_count'].mean():.1f}")

# Find correlations
print("\n\nWhat correlates with engagement?")
correlations = df.corr()['engagement'].sort_values(ascending=False)
print(correlations)

# Visualize patterns
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Engagement distribution
axes[0, 0].hist(df['engagement'], bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Engagement Distribution')
axes[0, 0].set_xlabel('Engagement')
axes[0, 0].set_ylabel('Frequency')

# 2. Caption length vs engagement
axes[0, 1].scatter(df['caption_length'], df['engagement'], alpha=0.5)
axes[0, 1].set_title('Caption Length vs Engagement')
axes[0, 1].set_xlabel('Caption Length')
axes[0, 1].set_ylabel('Engagement')

# 3. Hashtags vs engagement
axes[1, 0].scatter(df['hashtag_count'], df['engagement'], alpha=0.5, color='green')
axes[1, 0].set_title('Hashtags vs Engagement')
axes[1, 0].set_xlabel('Hashtag Count')
axes[1, 0].set_ylabel('Engagement')

# 4. Best posting times
hourly = df.groupby('posting_hour')['engagement'].mean()
axes[1, 1].plot(hourly.index, hourly.values, marker='o', color='purple')
axes[1, 1].set_title('Best Posting Times')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Avg Engagement')

plt.tight_layout()
plt.savefig('data_exploration.png')
print("\n\nVisualization saved as 'data_exploration.png'")

# Save dataset for next steps
df.to_csv('instagram_data.csv', index=False)
print("Dataset saved as 'instagram_data.csv'")

print("\nDone! Check the PNG file to see patterns.")