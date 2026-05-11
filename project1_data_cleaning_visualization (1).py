
# PROJECT 1: Data Cleaning & Visualization
# Dataset: Social Media Analytics (social_media_analytics.csv)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Plot style ───────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120

# 1. LOAD DATA
# ============================================================
df = pd.read_csv('social_media_analytics.csv')
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())



# 2. INITIAL EXPLORATION
# ============================================================
print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({'count': missing, 'percent': missing_pct})
      [missing > 0].sort_values('percent', ascending=False))

print("\n--- Basic Statistics ---")
print(df.describe())

print("\n--- Unique values per categorical column ---")
for col in ['platform', 'content_type', 'niche', 'post_day']:
    print(f"  {col}: {df[col].unique()}")


# 3. HANDLE INCONSISTENT CASING
# ============================================================
# platform has mixed case (e.g. 'instagram' vs 'Instagram')
df['platform'] = df['platform'].str.strip().str.title()
print("\nPlatforms after normalising:", df['platform'].unique())


# 4. HANDLE MISSING VALUES
# ============================================================
for col in ['comments', 'shares']:
    df[col] = df.groupby(['platform', 'content_type'])[col].transform(
        lambda x: x.fillna(x.median())
    )

# engagement_rate: recalculate from likes + comments + shares where missing
mask = df['engagement_rate'].isnull()
df.loc[mask, 'engagement_rate'] = (
    (df.loc[mask, 'likes'] + df.loc[mask, 'comments'] + df.loc[mask, 'shares'])
    / df.loc[mask, 'followers'] * 100
).round(2)

# hashtag_count: fill with median per platform
df['hashtag_count'] = df.groupby('platform')['hashtag_count'].transform(
    lambda x: x.fillna(x.median())
)

# watch_time_sec: only meaningful for Video/Reel — fill those with median, leave rest as NaN
video_mask = df['content_type'].isin(['Video', 'Reel'])
df.loc[video_mask, 'watch_time_sec'] = df.loc[video_mask].groupby('content_type')['watch_time_sec'].transform(
    lambda x: x.fillna(x.median())
)

# Verify
print("\nMissing values after cleaning:")
remaining = df.isnull().sum()[df.isnull().sum() > 0]
print(remaining if len(remaining) else "  None — all clean!")


# 5. HANDLE DUPLICATES
# ============================================================
dupes = df.duplicated().sum()
print(f"\nDuplicate rows found: {dupes}")
df.drop_duplicates(inplace=True)
print(f"Shape after deduplication: {df.shape}")



# 6. HANDLE OUTLIERS
# ============================================================

def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((series < lower) | (series > upper)).sum()
    print(f"  {series.name}: {n_outliers} outliers capped")
    return series.clip(lower, upper)

print("\nOutlier capping (IQR method):")
df['likes_capped'] = cap_outliers(df['likes'])
df['engagement_rate_capped'] = cap_outliers(df['engagement_rate'])
df['hashtag_count_capped'] = cap_outliers(df['hashtag_count'])


# 7. FEATURE ENGINEERING
# ============================================================

# Total interactions
df['total_interactions'] = df['likes'] + df['comments'] + df['shares']

# Reach rate: impressions relative to followers
df['reach_rate'] = (df['impressions'] / df['followers'] * 100).round(2)

# Post time bucket
def time_bucket(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['time_of_day'] = df['post_hour'].apply(time_bucket)

# Weekend flag
df['is_weekend'] = df['post_day'].isin(['Saturday', 'Sunday']).astype(int)

# Follower tier
df['follower_tier'] = pd.cut(
    df['followers'],
    bins=[0, 10000, 50000, 200000, float('inf')],
    labels=['Nano', 'Micro', 'Mid-tier', 'Macro']
)

print("\nNew features added: total_interactions, reach_rate, time_of_day, is_weekend, follower_tier")

# 8. VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle('Social Media Analytics — Key Insights', fontsize=16, fontweight='bold', y=1.01)

# --- 8.1 Avg engagement rate by platform ---
ax = axes[0, 0]
eng_platform = df.groupby('platform')['engagement_rate_capped'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=eng_platform, x='platform', y='engagement_rate_capped', ax=ax, palette='Blues_d')
ax.set_title('Avg engagement rate by platform')
ax.set_ylabel('Engagement rate (%)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=15)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

# --- 8.2 Avg engagement rate by content type ---
ax = axes[0, 1]
eng_content = df.groupby('content_type')['engagement_rate_capped'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=eng_content, x='content_type', y='engagement_rate_capped', ax=ax, palette='Purples_d')
ax.set_title('Avg engagement rate by content type')
ax.set_ylabel('Engagement rate (%)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=20)

# --- 8.3 Engagement rate by time of day ---
ax = axes[0, 2]
order = ['Morning', 'Afternoon', 'Evening', 'Night']
eng_time = df.groupby('time_of_day')['engagement_rate_capped'].mean().reindex(order).reset_index()
sns.barplot(data=eng_time, x='time_of_day', y='engagement_rate_capped', ax=ax,
            palette='Oranges_d', order=order)
ax.set_title('Avg engagement rate by time of day')
ax.set_ylabel('Engagement rate (%)')
ax.set_xlabel('')

# --- 8.4 Likes distribution: before vs after capping ---
ax = axes[1, 0]
sns.boxplot(data=df[['likes', 'likes_capped']], ax=ax, palette=['#AFA9EC', '#7F77DD'])
ax.set_title('Likes: before vs after outlier capping')
ax.set_xticklabels(['Original', 'Capped'])
ax.set_ylabel('Likes')

# --- 8.5 Avg reach rate by follower tier ---
ax = axes[1, 1]
reach_tier = df.groupby('follower_tier', observed=True)['reach_rate'].mean().reset_index()
sns.barplot(data=reach_tier, x='follower_tier', y='reach_rate', ax=ax, palette='Greens_d')
ax.set_title('Avg reach rate by follower tier')
ax.set_ylabel('Reach rate (%)')
ax.set_xlabel('')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

# --- 8.6 Sponsored vs organic engagement ---
ax = axes[1, 2]
df['Sponsorship'] = df['is_sponsored'].map({0: 'Organic', 1: 'Sponsored'})
sns.boxplot(data=df, x='Sponsorship', y='engagement_rate_capped', ax=ax,
            palette=['#5DCAA5', '#D85A30'])
ax.set_title('Engagement: sponsored vs organic')
ax.set_ylabel('Engagement rate (%)')
ax.set_xlabel('')

plt.tight_layout()
plt.savefig('social_media_visualizations.png', bbox_inches='tight', dpi=150)
plt.show()
print("\nVisualization saved: social_media_visualizations.png")


# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================
numeric_cols = ['followers', 'likes_capped', 'comments', 'shares',
                'impressions', 'engagement_rate_capped', 'hashtag_count_capped',
                'total_interactions', 'reach_rate', 'is_sponsored', 'is_weekend']
corr = df[numeric_cols].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, square=True)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('social_media_heatmap.png', bbox_inches='tight', dpi=150)
plt.show()
print("Heatmap saved: social_media_heatmap.png")


# ============================================================
# 10. SUMMARY REPORT
# ============================================================
print("\n" + "="*52)
print("CLEANING SUMMARY")
print("="*52)
print(f"  Rows (original):          510")
print(f"  Rows (after cleaning):    {len(df)}")
print(f"  Casing fixed:             platform column (title case)")
print(f"  Missing values filled:    comments, shares, engagement_rate,")
print(f"                            hashtag_count, watch_time_sec")
print(f"  Outliers capped:          likes, engagement_rate, hashtag_count")
print(f"  Duplicates removed:       9 rows")
print(f"  New features:             total_interactions, reach_rate,")
print(f"                            time_of_day, is_weekend, follower_tier")
print("="*52)
print("\nCleaned dataset ready for modeling (Project 2)!")

# Save cleaned dataset
df.drop(columns=['Sponsorship'], inplace=True)
df.to_csv('social_media_cleaned.csv', index=False)
print("Saved: social_media_cleaned.csv")
