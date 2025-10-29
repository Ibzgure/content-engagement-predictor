import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("=" * 50)
print("TRAINING YOUR FIRST AI MODEL")
print("=" * 50)

# Load the data we created yesterday
df = pd.read_csv('instagram_data.csv')
print(f"\nLoaded {len(df)} Instagram posts")

# STEP 1: Prepare the data
print("\n[STEP 1] Preparing data...")

# Features (X) - what we'll use to predict
# These are the inputs to our AI
X = df[['caption_length', 'hashtag_count', 'emoji_count', 
        'has_question', 'posting_hour']]

# Target (y) - what we want to predict
# This is the output
y = df['engagement']

print(f"Features (inputs): {list(X.columns)}")
print(f"Target (output): engagement")

# STEP 2: Split data into training and testing sets
print("\n[STEP 2] Splitting data...")

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} posts")
print(f"Testing set: {len(X_test)} posts")

print("\nWhy split?")
print("- Training set: AI learns patterns from this")
print("- Testing set: We test if AI learned correctly")
print("  (AI has never seen this data before!)")

# STEP 3: Train the model
print("\n[STEP 3] Training the model...")
print("This is where the AI 'learns'...")

model = LinearRegression()
model.fit(X_train, y_train)

print("✓ Model trained!")

# STEP 4: Make predictions
print("\n[STEP 4] Making predictions...")

# Predict on test data
y_pred = model.predict(X_test)

print(f"Made {len(y_pred)} predictions")

# STEP 5: Evaluate accuracy
print("\n[STEP 5] Evaluating accuracy...")

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"  → On average, predictions are off by {mae:.0f} engagement points")

print(f"\nR² Score: {r2:.3f}")
print(f"  → Model explains {r2*100:.1f}% of engagement variance")

if r2 > 0.7:
    print("  → Great! Model is pretty accurate")
elif r2 > 0.5:
    print("  → Decent! Model has learned some patterns")
else:
    print("  → Weak. Model needs improvement")

# STEP 6: Show what the model learned
print("\n[STEP 6] What did the AI learn?")
print("\nFeature Importance (coefficients):")

for feature, coef in zip(X.columns, model.coef_):
    impact = "increases" if coef > 0 else "decreases"
    print(f"  {feature:20s}: {coef:8.2f} → {impact} engagement")

print(f"\nBase engagement (intercept): {model.intercept_:.2f}")

# STEP 7: Test with example posts
print("\n[STEP 7] Testing with example posts...")

examples = [
    {
        'caption_length': 50,
        'hashtag_count': 5,
        'emoji_count': 2,
        'has_question': 1,
        'posting_hour': 18
    },
    {
        'caption_length': 200,
        'hashtag_count': 25,
        'emoji_count': 0,
        'has_question': 0,
        'posting_hour': 9
    },
    {
        'caption_length': 100,
        'hashtag_count': 10,
        'emoji_count': 5,
        'has_question': 1,
        'posting_hour': 20
    }
]

print("\nExample predictions:")
for i, example in enumerate(examples, 1):
    example_df = pd.DataFrame([example])
    prediction = model.predict(example_df)[0]
    
    print(f"\nPost {i}:")
    print(f"  Caption: {example['caption_length']} chars")
    print(f"  Hashtags: {example['hashtag_count']}")
    print(f"  Emojis: {example['emoji_count']}")
    print(f"  Has question: {'Yes' if example['has_question'] else 'No'}")
    print(f"  Posting hour: {example['posting_hour']}:00")
    print(f"  → Predicted engagement: {prediction:.0f}")

# STEP 8: Visualize predictions vs actual
print("\n[STEP 8] Creating visualization...")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', linewidth=2)
plt.xlabel('Actual Engagement')
plt.ylabel('Predicted Engagement')
plt.title(f'Model Predictions vs Reality (R² = {r2:.3f})')
plt.grid(True, alpha=0.3)

# Add text showing accuracy
plt.text(0.05, 0.95, f'Mean Error: {mae:.0f}', 
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=300)
print("✓ Visualization saved as 'model_predictions.png'")

# STEP 9: Save the model (for future use)
import pickle

with open('engagement_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✓ Model saved as 'engagement_model.pkl'")

print("\n" + "=" * 50)
print("SUCCESS! Your first AI model is complete!")
print("=" * 50)

print("\nWhat you built:")
print("✓ Trained a machine learning model")
print("✓ Model can predict engagement from post features")
print("✓ Evaluated accuracy")
print("✓ Saved model for future use")

print("\nNext: Build a simple interface to use this model!")