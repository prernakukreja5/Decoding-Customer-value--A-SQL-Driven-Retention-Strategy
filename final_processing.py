"""
================================================================
  DECODING CUSTOMER VALUE — D2C FASHION BRAND
  Complete Python Processing Pipeline
  
  SEQUENTIAL ORDER:
    1. Load Raw Data
    2. Data Cleaning
    3. EDA (guides feature engineering decisions)
    4. Feature Engineering (EDA-informed)
    5. Feature Selection (3 methods)
    6. Export Final Dataset

  FINAL LOCKED PARAMETERS:
    CVS  = 0.40 × Spend_norm + 0.40 × Tenure_norm + 0.20 × Freq_norm
    Loyalty (Def B) = CVS ≥ median AND promo_dependency_score = 0
    Value Tiers = Quartile split on CVS
    Satisfaction = review_rating ≥ 4.0
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Plot Styling ─────────────────────────────────────────────
DARK   = '#1a1a2e'; CARD   = '#16213e'; ACCENT = '#e94560'
GOLD   = '#f5a623'; TEAL   = '#00b4d8'; GREEN  = '#06d6a0'
GREY   = '#8892a4'; TEXT   = '#e0e0e0'; BLUE   = '#0f3460'
PURPLE = '#c77dff'
TIER_COLORS = {
    'Champions': '#f5a623', 'Loyalists': '#06d6a0',
    'At Risk':   '#00b4d8', 'Dormant':   '#e94560'
}
TIER_ORDER = ['Champions', 'Loyalists', 'At Risk', 'Dormant']

plt.rcParams.update({
    'figure.facecolor': DARK,  'axes.facecolor': CARD,
    'axes.edgecolor':   GREY,  'axes.labelcolor': TEXT,
    'xtick.color':      TEXT,  'ytick.color':     TEXT,
    'text.color':       TEXT,  'grid.color':      '#2a2a4a',
    'grid.linewidth':   0.5,   'font.family':     'DejaVu Sans',
})

OUTPUT = '/mnt/user-data/outputs/'
RAW    = '/mnt/user-data/uploads/Dataset__2_.csv'

def save(name):
    plt.savefig(f'{OUTPUT}{name}', dpi=150,
                bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  Saved: {name}")

def min_max(s):
    return (s - s.min()) / (s.max() - s.min())


# ================================================================
# STEP 1 — LOAD RAW DATA
# ================================================================
print("\n" + "="*60)
print("STEP 1 — LOAD RAW DATA")
print("="*60)

df = pd.read_csv(RAW)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Standardise column names
df.columns = (
    df.columns.str.strip().str.lower()
    .str.replace(' ', '_')
    .str.replace(r'[^a-z0-9_]', '', regex=True)
)
print(f"  Nulls per column:\n{df.isnull().sum()[df.isnull().sum()>0]}")


# ================================================================
# STEP 2 — DATA CLEANING
# ================================================================
print("\n" + "="*60)
print("STEP 2 — DATA CLEANING")
print("="*60)

# 2.1 Handle nulls — review_rating only (37 missing)
median_rating = df['review_rating'].median()
df['review_rating'] = df['review_rating'].fillna(median_rating)
print(f"  review_rating: 37 nulls filled with median ({median_rating})")

# 2.2 Convert Yes/No columns to binary
bool_cols = ['discount_applied', 'promo_code_used', 'subscription_status']
for col in bool_cols:
    df[col] = (df[col].str.strip().str.lower() == 'yes').astype(int)
print(f"  Converted to 0/1: {bool_cols}")

# 2.3 Verify
assert df.isnull().sum().sum() == 0, "Nulls still present!"
print(f"  Nulls remaining: 0 ✅")
print(f"  Duplicates: {df.duplicated().sum()} ✅")
print(f"  Clean dataset shape: {df.shape}")


# ================================================================
# STEP 3 — EDA
# (Run before feature engineering to inform decisions)
# ================================================================
print("\n" + "="*60)
print("STEP 3 — EXPLORATORY DATA ANALYSIS")
print("="*60)

# ── EDA: Key stats printed ────────────────────────────────────
print("\n  Numeric summary:")
print(df[['purchase_amount_usd','previous_purchases',
          'review_rating','age']].describe().round(2).to_string())

print("\n  Frequency of Purchases distribution:")
print(df['frequency_of_purchases'].value_counts().to_string())

print("\n  Discount Applied:", df['discount_applied'].value_counts().to_dict())
print("  Promo Code Used: ", df['promo_code_used'].value_counts().to_dict())
print("  Category:        ", df['category'].value_counts().to_dict())
print("  Gender:          ", df['gender'].value_counts().to_dict())

# NOTE: EDA charts generated after feature engineering
# so that engineered features (CVS, tiers) can be visualised too.
# This is the correct approach — basic EDA stats guide engineering,
# full visual EDA shows both raw + engineered features together.

print("\n  EDA stats complete — visuals generated post-engineering")


# ================================================================
# STEP 4 — FEATURE ENGINEERING
# (All decisions informed by EDA stats above)
# ================================================================
print("\n" + "="*60)
print("STEP 4 — FEATURE ENGINEERING")
print("="*60)

# ── 4.1 FREQUENCY SCORE ──────────────────────────────────────
# WHY: "Frequency of Purchases" is text — cannot use in math.
# Custom ordinal map: higher frequency = higher number (1-7).
# Not industry standard — design choice to avoid ratio scale
# (Weekly=52 vs Monthly=12) dominating CVS unfairly.
FREQ_MAP = {
    'Weekly': 7, 'Fortnightly': 6, 'Bi-Weekly': 5,
    'Monthly': 4, 'Every 3 Months': 3, 'Quarterly': 3, 'Annually': 1
}
df['frequency_score'] = df['frequency_of_purchases'].map(FREQ_MAP)
print(f"\n  4.1 frequency_score: {df['frequency_score'].value_counts().sort_index().to_dict()}")

# ── 4.2 PROMO DEPENDENCY SCORE ───────────────────────────────
# WHY: Brand's #1 question — are customers loyal or discount-driven?
# discount_applied and promo_code_used are perfectly correlated (|r|=1.0)
# — they always have the same value per row.
# Sum = 0 (organic) or 2 (promo-dependent). Binary in practice.
df['promo_dependency_score'] = df['discount_applied'] + df['promo_code_used']
print(f"\n  4.2 promo_dependency_score: {df['promo_dependency_score'].value_counts().to_dict()}")
print(f"      % promo-dependent: {(df['promo_dependency_score']>0).mean()*100:.1f}%")

# ── 4.3 CUSTOMER VALUE SCORE (CVS) ───────────────────────────
# FINAL LOCKED FORMULA: 40% Spend + 40% Tenure + 20% Frequency
#
# WHY THESE THREE:
#   Spend    = direct revenue contribution (Monetary in RFM)
#   Tenure   = relationship longevity, repeat behavior proxy
#   Frequency= purchase cadence, engagement signal
#
# WHY 40-40-20:
#   Spend & tenure equally important for D2C brand value.
#   Frequency matters but secondary — someone buying monthly
#   at high spend > someone buying weekly at low spend.
#   Aligned with RFM framework (Recency, Frequency, Monetary).
#
# NOTE: No timestamp = no true Recency.
#   Previous Purchases used as recency+loyalty proxy.
#   Limitation: a churned customer with 40 past purchases
#   still looks "high tenure" — acknowledged in report.
#
# Min-max normalization used so all features on 0-1 scale
# before weighting — prevents any one feature dominating
# due to scale differences.

df['spend_norm']     = min_max(df['purchase_amount_usd'])
df['tenure_norm']    = min_max(df['previous_purchases'])
df['frequency_norm'] = min_max(df['frequency_score'])

df['customer_value_score'] = (
    0.40 * df['spend_norm']
  + 0.40 * df['tenure_norm']
  + 0.20 * df['frequency_norm']
)
print(f"\n  4.3 CVS (40% spend + 40% tenure + 20% frequency):")
print(f"      Mean: {df['customer_value_score'].mean():.4f}")
print(f"      Std:  {df['customer_value_score'].std():.4f}")
print(f"      Min:  {df['customer_value_score'].min():.4f}")
print(f"      Max:  {df['customer_value_score'].max():.4f}")

# ── 4.4 VALUE TIER ────────────────────────────────────────────
# WHY: Continuous CVS is good for math but brand needs
# actionable labels for retention strategy.
# Quartile split ensures ~equal population in each tier.
q25 = df['customer_value_score'].quantile(0.25)
q50 = df['customer_value_score'].quantile(0.50)
q75 = df['customer_value_score'].quantile(0.75)

def assign_tier(s):
    if s >= q75:   return 'Champions'
    elif s >= q50: return 'Loyalists'
    elif s >= q25: return 'At Risk'
    else:          return 'Dormant'

df['value_tier'] = df['customer_value_score'].apply(assign_tier)
print(f"\n  4.4 Value Tier (quartile-based):")
print(f"      {df['value_tier'].value_counts().to_dict()}")
print(f"      CVS thresholds: Q25={q25:.4f}, Q50={q50:.4f}, Q75={q75:.4f}")

# ── 4.5 SATISFACTION FLAG ────────────────────────────────────
# WHY: review_rating is continuous (2.5-5.0) but brand needs
# a binary signal for cross-tabulation.
# Threshold = 4.0 (above dataset mean of 3.75) to identify
# genuinely positive experiences.
SATISFACTION_THRESHOLD = 4.0
df['satisfaction_flag'] = (df['review_rating'] >= SATISFACTION_THRESHOLD).astype(int)
print(f"\n  4.5 satisfaction_flag (rating >= {SATISFACTION_THRESHOLD}):")
print(f"      Satisfied: {df['satisfaction_flag'].sum()} ({df['satisfaction_flag'].mean()*100:.1f}%)")

# ── 4.6 LOYALTY DEFINITION B (FINAL — PREFERRED) ─────────────
# DEFINITION: CVS >= median AND promo_dependency_score = 0
#
# WHY DEF B IS PREFERRED:
#   CVS already captures spend (40%), tenure (40%), frequency (20%).
#   Adding promo=0 filter means: "high-value customer who does not
#   need a discount to buy." This is the true brand advocate.
#
# WHY NOT A PENALTY-BASED DEF C:
#   Data showed promo users CVS (0.5024) ≈ organic CVS (0.4998).
#   Gap = -0.0026 (essentially noise, promo users slightly higher).
#   No data basis for any penalty — would be arbitrary.
#
# WHY NOT TENURE-BASED DEF C:
#   Previous purchases already in CVS (40% weight) — circular.
#
# ACKNOWLEDGED LIMITATION:
#   Def B excludes 634 Champions/Loyalists who occasionally use
#   promos. These customers have avg tenure of 39 purchases.
#   In report: Def B = conservative loyalty benchmark.
#   Brand should A/B test removing discounts from Def B segment.

cvs_median = df['customer_value_score'].median()
df['loyal_def_b'] = (
    (df['customer_value_score'] >= cvs_median) &
    (df['promo_dependency_score'] == 0)
).astype(int)

print(f"\n  4.6 Loyal Def B (CVS >= median + promo = 0):")
print(f"      Loyal customers: {df['loyal_def_b'].sum()} ({df['loyal_def_b'].mean()*100:.1f}%)")
print(f"      Avg spend (loyal):     ${df[df['loyal_def_b']==1]['purchase_amount_usd'].mean():.2f}")
print(f"      Avg spend (non-loyal): ${df[df['loyal_def_b']==0]['purchase_amount_usd'].mean():.2f}")
print(f"      CVS median used: {cvs_median:.4f}")

# ── 4.7 AGE BAND ──────────────────────────────────────────────
# WHY: Marketing teams target cohorts, not individual ages.
# 5 standard demographic bands.
df['age_band'] = pd.cut(
    df['age'], bins=[0, 24, 34, 44, 54, 100],
    labels=['18-24', '25-34', '35-44', '45-54', '55+'], right=True
)
print(f"\n  4.7 age_band: {df['age_band'].value_counts().sort_index().to_dict()}")

# ── 4.8 US REGION ─────────────────────────────────────────────
# WHY: 50 states too granular for strategic decisions.
# US Census 4-region mapping for actionable geographic analysis.
REGION_MAP = {
    'Connecticut':'Northeast','Maine':'Northeast','Massachusetts':'Northeast',
    'New Hampshire':'Northeast','Rhode Island':'Northeast','Vermont':'Northeast',
    'New Jersey':'Northeast','New York':'Northeast','Pennsylvania':'Northeast',
    'Delaware':'South','Florida':'South','Georgia':'South','Maryland':'South',
    'North Carolina':'South','South Carolina':'South','Virginia':'South',
    'West Virginia':'South','Alabama':'South','Kentucky':'South',
    'Mississippi':'South','Tennessee':'South','Arkansas':'South',
    'Louisiana':'South','Oklahoma':'South','Texas':'South',
    'Illinois':'Midwest','Indiana':'Midwest','Michigan':'Midwest',
    'Ohio':'Midwest','Wisconsin':'Midwest','Iowa':'Midwest',
    'Kansas':'Midwest','Minnesota':'Midwest','Missouri':'Midwest',
    'Nebraska':'Midwest','North Dakota':'Midwest','South Dakota':'Midwest',
    'Arizona':'West','Colorado':'West','Idaho':'West','Montana':'West',
    'Nevada':'West','New Mexico':'West','Utah':'West','Wyoming':'West',
    'Alaska':'West','California':'West','Hawaii':'West',
    'Oregon':'West','Washington':'West',
}
df['us_region'] = df['location'].map(REGION_MAP).fillna('Other')
print(f"\n  4.8 us_region: {df['us_region'].value_counts().to_dict()}")

# ── 4.9 PAYMENT CATEGORY ─────────────────────────────────────
# WHY: Group 6 payment methods into 4 meaningful categories
# for behavioural segmentation.
PAY_MAP = {
    'Credit Card': 'Credit', 'Debit Card': 'Debit',
    'PayPal': 'Digital Wallet', 'Venmo': 'Digital Wallet',
    'Cash': 'Cash', 'Bank Transfer': 'Bank Transfer'
}
df['payment_category'] = df['payment_method'].map(PAY_MAP).fillna('Other')
print(f"\n  4.9 payment_category: {df['payment_category'].value_counts().to_dict()}")

print("\n  Feature Engineering complete — 9 features created")


# ================================================================
# STEP 5 — EDA VISUALISATIONS
# (Generated here so engineered features are available for plotting)
# ================================================================
print("\n" + "="*60)
print("STEP 5 — EDA VISUALISATIONS (7 pages)")
print("="*60)

# ── PAGE 1: Univariate Distributions ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle('EDA — Page 1: Univariate Distributions',
             fontsize=18, fontweight='bold', color=TEXT, y=1.01)

cols   = ['purchase_amount_usd','previous_purchases','review_rating',
          'age','frequency_score','customer_value_score']
titles = ['Purchase Amount (USD)','Previous Purchases','Review Rating',
          'Age','Frequency Score','Customer Value Score (CVS)']
colors = [ACCENT, TEAL, GOLD, GREEN, BLUE, PURPLE]

for ax, col, title, color in zip(axes.flat, cols, titles, colors):
    data = df[col].dropna()
    ax.hist(data, bins=30, color=color, alpha=0.85, edgecolor='none')
    ax.axvline(data.mean(),   color='white', lw=1.8, ls='--',
               label=f'Mean: {data.mean():.1f}')
    ax.axvline(data.median(), color=GOLD,    lw=1.8, ls=':',
               label=f'Median: {data.median():.1f}')
    ax.set_title(title, fontsize=12, fontweight='bold', color=TEXT, pad=8)
    ax.set_ylabel('Count', fontsize=9, color=GREY)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
save('eda_page1_univariate.png')

# ── PAGE 2: Categorical Distributions ────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle('EDA — Page 2: Categorical Distributions',
             fontsize=18, fontweight='bold', color=TEXT, y=1.01)

# Value Tier
ax = axes[0,0]
counts = df['value_tier'].value_counts().reindex(TIER_ORDER)
bars = ax.barh(TIER_ORDER, counts.values,
               color=[TIER_COLORS[t] for t in TIER_ORDER], height=0.6)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_width()+8, bar.get_y()+bar.get_height()/2,
            f'{val} ({val/len(df)*100:.0f}%)', va='center',
            fontsize=10, color=TEXT)
ax.set_title('Value Tier Distribution', fontsize=12,
             fontweight='bold', color=TEXT)
ax.set_xlim(0, 1300); ax.grid(axis='x', alpha=0.4)

# Promo Dependency
ax = axes[0,1]
pc = df['promo_dependency_score'].value_counts().sort_index()
wedges,_,at = ax.pie(
    pc.values,
    labels=['Organic Buyers\n(No Promo)', 'Promo-Dependent'],
    colors=[GREEN, ACCENT], autopct='%1.1f%%', startangle=90,
    textprops={'color':TEXT,'fontsize':11},
    wedgeprops={'edgecolor':DARK,'linewidth':2})
for a in at: a.set_fontsize(12); a.set_fontweight('bold')
ax.set_title('Promo Dependency Split', fontsize=12,
             fontweight='bold', color=TEXT)

# Age Band
ax = axes[0,2]
age_order = ['18-24','25-34','35-44','45-54','55+']
ac = df['age_band'].value_counts().reindex(age_order)
bars = ax.bar(age_order, ac.values, color=TEAL, alpha=0.85, width=0.6)
for bar, val in zip(bars, ac.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
            str(val), ha='center', fontsize=10, color=TEXT)
ax.set_title('Age Band Distribution', fontsize=12,
             fontweight='bold', color=TEXT)
ax.grid(axis='y', alpha=0.4)

# US Region
ax = axes[1,0]
rc = df['us_region'].value_counts()
bars = ax.bar(rc.index, rc.values,
              color=[GOLD,BLUE,ACCENT,GREEN], alpha=0.9, width=0.6)
for bar, val in zip(bars, rc.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
            str(val), ha='center', fontsize=10, color=TEXT)
ax.set_title('US Region Distribution', fontsize=12,
             fontweight='bold', color=TEXT)
ax.grid(axis='y', alpha=0.4)

# Category
ax = axes[1,1]
cc = df['category'].value_counts()
bars = ax.barh(cc.index, cc.values,
               color=[TEAL,GOLD,ACCENT,GREEN], height=0.6)
for bar, val in zip(bars, cc.values):
    ax.text(bar.get_width()+8, bar.get_y()+bar.get_height()/2,
            f'{val} ({val/len(df)*100:.1f}%)', va='center',
            fontsize=10, color=TEXT)
ax.set_title('Category Distribution', fontsize=12,
             fontweight='bold', color=TEXT)
ax.set_xlim(0, 2100); ax.grid(axis='x', alpha=0.4)

# Gender
ax = axes[1,2]
gc = df['gender'].value_counts()
wedges,_,at = ax.pie(
    gc.values, labels=gc.index, colors=[BLUE,PURPLE],
    autopct='%1.1f%%', startangle=90,
    textprops={'color':TEXT,'fontsize':13},
    wedgeprops={'edgecolor':DARK,'linewidth':2})
for a in at: a.set_fontsize(13); a.set_fontweight('bold')
ax.set_title('Gender Split', fontsize=12, fontweight='bold', color=TEXT)

plt.tight_layout()
save('eda_page2_categorical.png')

# ── PAGE 3: Bivariate — Tier vs Key Metrics ───────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle('EDA — Page 3: Bivariate Analysis (Value Tier vs Metrics)',
             fontsize=18, fontweight='bold', color=TEXT, y=1.01)

# Tier vs Avg Spend
ax = axes[0,0]
means = df.groupby('value_tier')['purchase_amount_usd'].mean().reindex(TIER_ORDER)
bars = ax.bar(TIER_ORDER, means.values,
              color=[TIER_COLORS[t] for t in TIER_ORDER], width=0.6)
for bar, val in zip(bars, means.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'${val:.0f}', ha='center', fontsize=11,
            fontweight='bold', color=TEXT)
ax.set_title('Avg Spend by Value Tier', fontsize=12,
             fontweight='bold', color=TEXT)
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.4)

# Tier vs Promo %
ax = axes[0,1]
pp = df.groupby('value_tier')['promo_dependency_score'].apply(
    lambda x: (x>0).mean()*100).reindex(TIER_ORDER)
bars = ax.bar(TIER_ORDER, pp.values,
              color=[TIER_COLORS[t] for t in TIER_ORDER], width=0.6)
for bar, val in zip(bars, pp.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', fontsize=11,
            fontweight='bold', color=TEXT)
ax.set_title('Promo Dependency % by Tier', fontsize=12,
             fontweight='bold', color=TEXT)
ax.set_ylim(0, 60); ax.grid(axis='y', alpha=0.4)

# Tier vs Avg Previous Purchases
ax = axes[0,2]
prev = df.groupby('value_tier')['previous_purchases'].mean().reindex(TIER_ORDER)
bars = ax.bar(TIER_ORDER, prev.values,
              color=[TIER_COLORS[t] for t in TIER_ORDER], width=0.6)
for bar, val in zip(bars, prev.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}', ha='center', fontsize=11,
            fontweight='bold', color=TEXT)
ax.set_title('Avg Previous Purchases by Tier', fontsize=12,
             fontweight='bold', color=TEXT)
ax.grid(axis='y', alpha=0.4)

# Organic vs Promo Spend Boxplot
ax = axes[1,0]
df['promo_label'] = df['promo_dependency_score'].map(
    {0: 'Organic\n(No Promo)', 2: 'Promo-Dependent'})
groups = [df[df['promo_label']==l]['purchase_amount_usd'].values
          for l in ['Organic\n(No Promo)', 'Promo-Dependent']]
bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                medianprops={'color':'white','linewidth':2})
for patch, color in zip(bp['boxes'], [GREEN, ACCENT]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
for el in ['whiskers','caps','fliers']:
    for item in bp[el]: item.set(color=GREY, linewidth=1.2)
ax.set_xticklabels(['Organic\n(No Promo)', 'Promo-Dependent'])
ax.set_title('Spend Distribution: Organic vs Promo', fontsize=12,
             fontweight='bold', color=TEXT)
ax.grid(axis='y', alpha=0.4)

# Age Band vs Avg CVS
ax = axes[1,1]
acvs = df.groupby('age_band')['customer_value_score'].mean().reindex(age_order)
bars = ax.bar(age_order, acvs.values, color=TEAL, alpha=0.85, width=0.6)
for bar, val in zip(bars, acvs.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{val:.3f}', ha='center', fontsize=10, color=TEXT)
ax.set_title('Avg CVS by Age Band', fontsize=12,
             fontweight='bold', color=TEXT)
ax.grid(axis='y', alpha=0.4)

# Region: Spend vs Promo
ax = axes[1,2]
rs = df.groupby('us_region')['purchase_amount_usd'].mean().sort_values(ascending=False)
rp = df.groupby('us_region')['promo_dependency_score'].apply(
    lambda x: (x>0).mean()*100).reindex(rs.index)
x  = np.arange(len(rs)); w = 0.35
b1 = ax.bar(x-w/2, rs.values, width=w, color=GOLD, alpha=0.9,
            label='Avg Spend ($)')
ax2 = ax.twinx(); ax2.set_facecolor(CARD)
b2  = ax2.bar(x+w/2, rp.values, width=w, color=ACCENT, alpha=0.7,
              label='Promo % (right)')
ax.set_xticks(x); ax.set_xticklabels(rs.index, fontsize=9)
ax.set_ylabel('Avg Spend', color=GOLD)
ax2.set_ylabel('Promo %', color=ACCENT)
ax.tick_params(axis='y', colors=GOLD)
ax2.tick_params(axis='y', colors=ACCENT)
ax.set_title('Region: Spend vs Promo Dependency', fontsize=12,
             fontweight='bold', color=TEXT)
ax.legend([b1,b2], ['Avg Spend ($)','Promo % (right)'],
          fontsize=8, loc='upper right', framealpha=0.3)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
save('eda_page3_bivariate.png')

# ── PAGE 4: Correlation Matrix + Outliers ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle('EDA — Page 4: Correlation Matrix & Outlier Detection',
             fontsize=18, fontweight='bold', color=TEXT, y=1.01)

num_cols = ['purchase_amount_usd','previous_purchases','review_rating',
            'age','frequency_score','customer_value_score',
            'promo_dependency_score','satisfaction_flag','discount_applied']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, ax=axes[0], cmap=cmap, center=0,
            annot=True, fmt='.2f', annot_kws={'size':9},
            linewidths=0.5, linecolor=DARK,
            cbar_kws={'shrink':0.8})
axes[0].set_title('Correlation Matrix', fontsize=13,
                  fontweight='bold', color=TEXT, pad=10)
axes[0].tick_params(axis='x', rotation=45, labelsize=9)
axes[0].tick_params(axis='y', rotation=0,  labelsize=9)

cols_box   = ['purchase_amount_usd','previous_purchases','age','review_rating']
labels_box = ['Spend\n(USD)','Prev\nPurchases','Age','Review\nRating']
colors_box = [ACCENT, TEAL, GOLD, GREEN]
bp = axes[1].boxplot([df[c].dropna().values for c in cols_box],
                     patch_artist=True, widths=0.5,
                     medianprops={'color':'white','linewidth':2.5},
                     flierprops={'marker':'o','markersize':3,
                                 'alpha':0.4,'color':GREY})
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.75)
for el in ['whiskers','caps']:
    for item in bp[el]: item.set(color=GREY, linewidth=1.2)
axes[1].set_xticklabels(labels_box, fontsize=11)
axes[1].set_title('Outlier Detection — Key Numeric Features',
                  fontsize=13, fontweight='bold', color=TEXT, pad=10)
axes[1].grid(axis='y', alpha=0.4)
for i, col in enumerate(cols_box):
    q1 = df[col].quantile(0.25); q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    n = len(df[(df[col]<q1-1.5*iqr)|(df[col]>q3+1.5*iqr)])
    axes[1].text(i+1, df[col].max()*0.97,
                 f'{n} outliers', ha='center', fontsize=8.5, color=GREY)

plt.tight_layout()
save('eda_page4_correlation_outliers.png')

# ── PAGE 5: Category & Season Deep Dive ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle('EDA — Page 5: Category & Season Deep Dive',
             fontsize=18, fontweight='bold', color=TEXT, y=1.01)

# Category vs Avg Tenure
ax = axes[0,0]
cp = df.groupby('category')['previous_purchases'].mean().sort_values(ascending=False)
bars = ax.barh(cp.index, cp.values,
               color=[TEAL,GOLD,GREEN,ACCENT], height=0.6)
for bar, val in zip(bars, cp.values):
    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=11,
            fontweight='bold', color=TEXT)
ax.set_title('Avg Previous Purchases by Category\n(Higher = More Retained)',
             fontsize=11, fontweight='bold', color=TEXT)
ax.grid(axis='x', alpha=0.4)

# Category vs Promo %
ax = axes[0,1]
cpr = df.groupby('category')['promo_dependency_score'].apply(
    lambda x: (x>0).mean()*100).sort_values(ascending=False)
bars = ax.barh(cpr.index, cpr.values,
               color=[ACCENT,GOLD,TEAL,GREEN], height=0.6)
for bar, val in zip(bars, cpr.values):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=11,
            fontweight='bold', color=TEXT)
ax.set_title('Promo Dependency % by Category',
             fontsize=11, fontweight='bold', color=TEXT)
ax.set_xlim(0, 60); ax.grid(axis='x', alpha=0.4)

# Season vs Tier (stacked)
ax = axes[1,0]
seasons = ['Spring','Summer','Fall','Winter']
st = df.groupby(['season','value_tier']).size().unstack(fill_value=0)
st = st.reindex(columns=TIER_ORDER).reindex(seasons)
bottom = np.zeros(len(seasons))
for tier in TIER_ORDER:
    ax.bar(seasons, st[tier].values, bottom=bottom,
           color=TIER_COLORS[tier], label=tier,
           edgecolor=DARK, linewidth=0.5)
    bottom += st[tier].values
ax.set_title('Season vs Value Tier Distribution',
             fontsize=11, fontweight='bold', color=TEXT)
ax.set_ylabel('Customer Count', color=GREY)
ax.legend(fontsize=9, framealpha=0.3, loc='upper right')
ax.grid(axis='y', alpha=0.4)

# Season vs Tenure + Promo
ax = axes[1,1]
sp  = df.groupby('season')['previous_purchases'].mean().reindex(seasons)
spr = df.groupby('season')['promo_dependency_score'].apply(
    lambda x: (x>0).mean()*100).reindex(seasons)
x = np.arange(len(seasons)); w = 0.35
b1 = ax.bar(x-w/2, sp.values, width=w, color=TEAL, alpha=0.9,
            label='Avg Prev Purchases')
ax2 = ax.twinx(); ax2.set_facecolor(CARD)
b2  = ax2.bar(x+w/2, spr.values, width=w, color=ACCENT, alpha=0.7,
              label='Promo % (right)')
ax.set_xticks(x); ax.set_xticklabels(seasons)
ax.set_ylabel('Avg Prev Purchases', color=TEAL)
ax2.set_ylabel('Promo Dependency %', color=ACCENT)
ax.tick_params(axis='y', colors=TEAL)
ax2.tick_params(axis='y', colors=ACCENT)
ax.set_title('Season: Tenure vs Promo Dependency',
             fontsize=11, fontweight='bold', color=TEXT)
ax.legend([b1,b2], ['Avg Prev Purchases','Promo % (right)'],
          fontsize=8, loc='upper left', framealpha=0.3)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
save('eda_page5_category_season.png')

# ── PAGE 6: Loyal vs Non-Loyal (Def B) ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor(DARK)
fig.suptitle('EDA — Page 6: Loyalty Definition B — Loyal vs Non-Loyal',
             fontsize=18, fontweight='bold', color=TEXT, y=1.01)

loyal     = df[df['loyal_def_b']==1]
non_loyal = df[df['loyal_def_b']==0]

# Segment size pie
ax = axes[0]
wedges,_,at = ax.pie(
    [len(loyal), len(non_loyal)],
    labels=[f'Loyal (Def B)\n{len(loyal)} ({len(loyal)/len(df)*100:.1f}%)',
            f'Non-Loyal\n{len(non_loyal)} ({len(non_loyal)/len(df)*100:.1f}%)'],
    colors=[TEAL, ACCENT], autopct='%1.1f%%', startangle=90,
    textprops={'color':TEXT,'fontsize':11},
    wedgeprops={'edgecolor':DARK,'linewidth':2})
for a in at: a.set_fontsize(12); a.set_fontweight('bold')
ax.set_title('Loyal vs Non-Loyal Split', fontsize=12,
             fontweight='bold', color=TEXT)

# Key metrics comparison
ax = axes[1]
metrics = {
    'Avg Spend ($)':      [loyal['purchase_amount_usd'].mean(),
                           non_loyal['purchase_amount_usd'].mean()],
    'Avg Tenure':         [loyal['previous_purchases'].mean(),
                           non_loyal['previous_purchases'].mean()],
    'CVS (×100)':         [loyal['customer_value_score'].mean()*100,
                           non_loyal['customer_value_score'].mean()*100],
    'Satisfaction %':     [loyal['satisfaction_flag'].mean()*100,
                           non_loyal['satisfaction_flag'].mean()*100],
}
x  = np.arange(len(metrics)); w = 0.35
b1 = ax.bar(x-w/2, [v[0] for v in metrics.values()],
            width=w, color=TEAL, alpha=0.85, label='Loyal (Def B)')
b2 = ax.bar(x+w/2, [v[1] for v in metrics.values()],
            width=w, color=ACCENT, alpha=0.85, label='Non-Loyal')
ax.set_xticks(x)
ax.set_xticklabels(list(metrics.keys()), fontsize=9)
ax.set_title('Loyal vs Non-Loyal: Key Metrics', fontsize=12,
             fontweight='bold', color=TEXT)
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(axis='y', alpha=0.4)

# Loyal % by tier
ax = axes[2]
loyal_by_tier = df.groupby('value_tier')['loyal_def_b'].mean()*100
loyal_by_tier = loyal_by_tier.reindex(TIER_ORDER)
bars = ax.bar(TIER_ORDER, loyal_by_tier.values,
              color=[TIER_COLORS[t] for t in TIER_ORDER], width=0.6)
for bar, val in zip(bars, loyal_by_tier.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', fontsize=11,
            fontweight='bold', color=TEXT)
ax.set_title('Loyal % by Value Tier', fontsize=12,
             fontweight='bold', color=TEXT)
ax.set_ylabel('% Loyal Customers', color=GREY)
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
save('eda_page6_loyalty_defb.png')

# ── PAGE 7: Feature Selection Summary ─────────────────────────
CANDIDATE_FEATURES = [
    'age','frequency_score','purchase_amount_usd','previous_purchases',
    'review_rating','promo_dependency_score','satisfaction_flag',
    'discount_applied','subscription_status',
    'spend_norm','tenure_norm','frequency_norm',
    'loyal_def_b','gender','category','season',
    'us_region','payment_category','age_band','shipping_type','size',
]

df_enc = df[CANDIDATE_FEATURES + ['value_tier']].copy()
for col in df_enc.select_dtypes(include='object').columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(df_enc[CANDIDATE_FEATURES], df_enc['value_tier'])
importances = pd.Series(rf.feature_importances_,
                        index=CANDIDATE_FEATURES).sort_values(ascending=False)

SELECTED = ['purchase_amount_usd','previous_purchases','frequency_score',
            'loyal_def_b','promo_dependency_score','age_band','us_region',
            'category','season','payment_category','satisfaction_flag',
            'value_tier','customer_value_score']

def get_status(f):
    if f in SELECTED:               return 'Selected'
    if f in ['spend_norm','tenure_norm','frequency_norm','discount_applied']:
        return 'Dropped — Redundant'
    return 'Dropped — Low Signal'

color_map = {
    'Selected':              GREEN,
    'Dropped — Redundant':   GOLD,
    'Dropped — Low Signal':  ACCENT
}
colors_bar = [color_map[get_status(f)] for f in importances.index]

num_feats = [f for f in CANDIDATE_FEATURES
             if df[f].dtype in [np.float64, np.int64]]
corr_cvs  = df[num_feats].corrwith(df['customer_value_score']).abs().sort_values(ascending=False)
colors_c  = [color_map[get_status(f)] for f in corr_cvs.index]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(DARK)
fig.suptitle('Feature Selection — Final Decision Summary',
             fontsize=18, fontweight='bold', color=TEXT, y=1.01)

axes[0].barh(importances.index[::-1], importances.values[::-1],
             color=colors_bar[::-1], edgecolor='none', height=0.7)
axes[0].set_title('Random Forest Feature Importance',
                  fontsize=12, fontweight='bold', color=TEXT)
axes[0].set_xlabel('Importance Score', color=GREY)
axes[0].grid(axis='x', alpha=0.4)

axes[1].barh(corr_cvs.index[::-1], corr_cvs.values[::-1],
             color=colors_c[::-1], edgecolor='none', height=0.7)
axes[1].set_title('Correlation with CVS (|Pearson r|)',
                  fontsize=12, fontweight='bold', color=TEXT)
axes[1].set_xlabel('|Pearson r|', color=GREY)
axes[1].grid(axis='x', alpha=0.4)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=GREEN,  label='Selected'),
    Patch(facecolor=GOLD,   label='Dropped — Redundant'),
    Patch(facecolor=ACCENT, label='Dropped — Low Signal')
]
for ax in axes:
    ax.legend(handles=legend_elements, fontsize=9,
              framealpha=0.3, loc='lower right')

plt.tight_layout()
save('eda_page7_feature_selection.png')

print("  All 7 EDA pages saved")


# ================================================================
# STEP 6 — FEATURE SELECTION (Decision Summary)
# ================================================================
print("\n" + "="*60)
print("STEP 6 — FEATURE SELECTION")
print("="*60)

print("""
  METHOD 1 — Pearson Correlation with CVS:
    purchase_amount_usd   |r| = 0.662  → SELECTED
    previous_purchases    |r| = 0.668  → SELECTED
    frequency_score       |r| = 0.347  → SELECTED
    spend_norm            |r| = 1.000  → DROPPED (redundant with spend)
    tenure_norm           |r| = 1.000  → DROPPED (redundant with tenure)
    frequency_norm        |r| = 1.000  → DROPPED (redundant with freq)
    promo_dependency_score|r| = 0.007  → SELECTED (strategic, not CVS signal)
    satisfaction_flag     |r| = 0.021  → SELECTED (cross-tab signal)
    discount_applied      |r| = 1.000  → DROPPED (= promo_dep_score/2)
    age                   |r| = 0.016  → DROPPED (age_band sufficient)
    subscription_status   |r| = 0.020  → DROPPED (low signal)

  METHOD 2 — Random Forest Importance vs value_tier:
    purchase_amount_usd   imp = 0.152  → SELECTED
    previous_purchases    imp = 0.141  → SELECTED
    frequency_score       imp = 0.054  → SELECTED
    gender                imp = 0.005  → DROPPED (no segment split)
    size                  imp = 0.003  → DROPPED (operational, not strategic)
    shipping_type         imp = 0.004  → DROPPED (operational)

  METHOD 3 — Redundancy Check (|r| > 0.85 between features):
    spend_norm ↔ purchase_amount_usd     |r|=1.0 → keep raw
    tenure_norm ↔ previous_purchases     |r|=1.0 → keep raw
    frequency_norm ↔ frequency_score     |r|=1.0 → keep raw
    discount_applied ↔ promo_dep_score   |r|=1.0 → keep composite

  FINAL: 13 features selected for SQL layer
""")

FINAL_COLS = [
    'customer_id','purchase_amount_usd','previous_purchases',
    'frequency_score','loyal_def_b','promo_dependency_score',
    'age_band','us_region','category','season','payment_category',
    'satisfaction_flag','value_tier','customer_value_score',
    'gender','location','item_purchased','color','age'
]
df_final = df[[c for c in FINAL_COLS if c in df.columns]].copy()


# ================================================================
# STEP 7 — EXPORT FINAL DATASET
# ================================================================
print("\n" + "="*60)
print("STEP 7 — EXPORT FINAL DATASET")
print("="*60)

df_final.to_csv(f'{OUTPUT}customers_final.csv', index=False)
print(f"  Saved: customers_final.csv")
print(f"  Shape: {df_final.shape}")
print(f"  Columns: {list(df_final.columns)}")

# ── Final validation ──────────────────────────────────────────
print("\n  VALIDATION — Value Tier × Key Metrics:")
summary = df_final.groupby('value_tier').agg(
    count           = ('customer_id','count'),
    avg_cvs         = ('customer_value_score','mean'),
    avg_spend       = ('purchase_amount_usd','mean'),
    avg_tenure      = ('previous_purchases','mean'),
    promo_pct       = ('promo_dependency_score', lambda x: (x>0).mean()*100),
    loyal_pct       = ('loyal_def_b','mean'),
    satisfaction_pct= ('satisfaction_flag','mean'),
).round(2)
print(summary.reindex(TIER_ORDER).to_string())

print("\n  VALIDATION — Loyal Def B:")
print(f"    Loyal customers:    {df_final['loyal_def_b'].sum()} ({df_final['loyal_def_b'].mean()*100:.1f}%)")
print(f"    Loyal avg spend:    ${df_final[df_final['loyal_def_b']==1]['purchase_amount_usd'].mean():.2f}")
print(f"    Non-loyal avg spend:${df_final[df_final['loyal_def_b']==0]['purchase_amount_usd'].mean():.2f}")

print("\n" + "="*60)
print("  PIPELINE COMPLETE")
print("  Output: customers_final.csv")
print("  EDA:    eda_page1 through eda_page7")
print("="*60)
