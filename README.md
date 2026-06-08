# Decoding-Customer-value--A-SQL-Driven-Retention-Strategy
SQL-driven customer segmentation and retention strategy for a D2C fashion brand — RFM-based value scoring, promo dependency analysis, and loyalty classification across 3,900 customers.

# Decoding Customer Value: A SQL-Driven Retention Strategy

> End-to-end customer intelligence project for a D2C fashion brand — RFM-based value scoring, promo dependency analysis, loyalty classification, and a data-backed retention playbook built entirely from transactional data with no pre-existing loyalty scores or churn labels.

---

## The business problem

A direct-to-consumer fashion brand with ~3,900 customers runs a promotional discount programme but has never measured whether it builds loyalty or simply attracts one-time bargain hunters. The brand cannot answer:

- Who are its most valuable customers, and what do they look like?
- Is discount spend generating incremental revenue, or just giving away margin?
- Which geographies and demographics show organic demand that has not been deliberately targeted?
- What does a data-backed retention strategy look like, with named segments, timelines, and measurable outcomes?

This project answers all four questions using only the transactional and behavioural data available.

---

## Key findings

| Finding | Number |
|---|---|
| Promo-dependent customers | 43% of base |
| Organic revenue share | 57.3% ($133,670) |
| Avg spend difference: organic vs promo | Less than $1 |
| Champion promo dependency | 43.9% — even top customers are discount-habituated |
| High Spend + High Tenure = Champion | 97.6% classification rate |
| High Spend alone (low tenure) = Champion | Only 5.7% |

**Central conclusion:** The brand is not building a loyal customer base. It is reliant on continuous promotional activity. Discounts generate no incremental revenue — organic and promo buyers spend almost identically ($60.13 vs $59.28). This is a margin problem disguised as a loyalty programme.

---

## Repository structure

```
├── data/
│   └── ecommerce_customer_data_large.csv       # Raw dataset (UCI)
│
├── python/
│   └── feature_engineering.py                  # Data cleaning + all engineered features
│
└── sql/
    └── customer_segmentation_queries.sql        # All 5 segmentation queries
```

> **Reports, EDA visuals, and Power BI dashboard are on Google Drive:**
> [https://drive.google.com/drive/u/0/folders/12qDPqV8lWgdL0x6N49uzZqnUE4n2fpUo](https://drive.google.com/drive/u/0/folders/12qDPqV8lWgdL0x6N49uzZqnUE4n2fpUo)
>
> Drive contains: `executive_summary.docx` · `retention_playbook.docx` · `sql_results_report.docx` · `founder_dashboard.pbix` · EDA pages 1–6 (PNG)

---

## Python: `feature_engineering.py`

### What it does

Takes the raw dataset and produces a cleaned, analysis-ready CSV with seven engineered features. No features are borrowed from existing loyalty frameworks without justification — every metric is constructed from available variables and defended on business logic.

### Engineered features

| Feature | Logic | What it captures |
|---|---|---|
| `customer_value_score` (CVS) | Normalised composite of spend (50%) and previous purchases (50%) | Proxy for RFM value — customers worth retaining |
| `value_tier` | CVS quartile split: Champions / Loyalists / At Risk / Dormant | Actionable segmentation without a pre-existing loyalty label |
| `promo_dependency_score` | Binary flag: 1 if discount applied, else 0 | Whether a customer requires a discount incentive to buy |
| `frequency_score` | Binned purchase frequency: 1 (once) to 7 (very frequent) | Engagement cadence independent of spend |
| `satisfaction_flag` | Binary: 1 if review rating >= 4.0, else 0 | Proxy for experience quality — correlated 0.85 with review_rating |
| `loyalty_def_b` (Strict) | No promo use AND CVS above median | Conservative: genuinely organic, tenure-qualified customers |
| `loyalty_def_c` (Preferred) | CVS above median regardless of occasional promo use | Realistic: recovers 634 high-value customers misclassified by Def B |

### Why Definition C is preferred over Definition B

Definition B excludes any customer who has ever used a promo code. This misclassifies 634 Champions and Loyalists who meet every tenure and spend threshold but happen to have used a discount at some point. Definition C treats occasional promo use as noise rather than disqualification. The 42.6% of loyal customers who sometimes use promos still spend $68.63 on average versus $54.29 for non-loyal customers — they are not bargain hunters.

### How to run

```bash
pip install pandas numpy scikit-learn
python python/feature_engineering.py
```

Input: `data/ecommerce_customer_data_large.csv`
Output: `data/customers_engineered.csv`

---

## SQL: `customer_segmentation_queries.sql`

### What it does

Five structured queries that answer the brand's five core business questions. Each query is self-contained and annotated. All run on the engineered CSV loaded into any standard SQL environment (SQLite, PostgreSQL, MySQL, DuckDB).

### Query index

**Q1 — Loyal vs Non-Loyal (Definition C)**
Compares avg spend, tenure, and satisfaction between loyal and non-loyal customers. Calculates loyal % within each value tier. Splits total revenue between organic and promo-dependent buyers.

Key output: Loyal customers spend $14 more on average and have 2.3x higher tenure. Discounts add less than $1 to basket size.

**Q2 — Behavioural patterns**
Tests the High Spend + High Tenure combination hypothesis. Calculates Champion classification rate for four spend-tenure combinations. Computes avg tenure and promo dependency by category and by season.

Key output: High Spend + High Tenure produces 97.6% Champions. High Spend alone produces 5.7%. Category tenure differences are negligible (~25 purchases across all four).

**Q3 — Geographic analysis**
Calculates organic buyer %, avg spend, and Champion concentration by US region and by state. Cross-tabs region with category to find the highest-organic combinations.

Key output: Northeast is the most organic region (59.5%). Pennsylvania has the highest Champion concentration of any state (45.9%). Northeast x Accessories is the most underlevered combination (64.1% organic).

**Q4 — Promotional strategy**
Calculates promo lift (organic avg spend minus promo avg spend) by value tier and by category. Ranks the top promo-dependent Champion sub-segments by sunset priority score.

Key output: Discounts reduce spend in 3 of 4 categories. Outerwear shows -$3.34 lift. Only Footwear shows a positive lift (+$2.72). Champions with promo spend $1.06 less than Champions without.

**Q5 — Ideal Customer Profile**
Aggregates all key dimensions (age, region, category, season, payment, spend, tenure, frequency) across the Champions segment to produce a targetable ICP.

Key output: 55+, male, South/Midwest, Clothing entry, Digital Wallet, $80.43 avg spend, 37.9 avg purchases, monthly cadence.

### How to run

Load `data/customers_engineered.csv` into your SQL environment as a table named `customers`, then run:

```sql
-- SQLite example
.mode csv
.import data/customers_engineered.csv customers
.read sql/customer_segmentation_queries.sql
```

---

## Dataset

Source: [UCI E-Commerce Customer Behaviour dataset](https://archive.ics.uci.edu/dataset/ecommerce)
Size: ~3,900 rows, one row per customer
Key fields: `purchase_amount_usd`, `previous_purchases`, `age`, `us_region`, `category`, `season`, `payment_method`, `shipping_type`, `discount_applied`, `review_rating`

**Important constraint:** The dataset has no timestamps, no transaction history, and no churn labels. Every analytical concept (loyalty, retention, value, tenure) is constructed from available variables. `previous_purchases` is used as a tenure proxy throughout.

---

## Deliverables

| Deliverable | Location | Purpose |
|---|---|---|
| Cleaned dataset + features | `data/customers_engineered.csv` | Input to all SQL queries and dashboard |
| Feature engineering code | `python/feature_engineering.py` | Reproducible pipeline |
| Segmentation queries | `sql/customer_segmentation_queries.sql` | Answers to all 5 business questions |
| Founder dashboard | [Google Drive](https://drive.google.com/drive/u/0/folders/12qDPqV8lWgdL0x6N49uzZqnUE4n2fpUo) | Non-technical decision-making tool |
| Executive summary | [Google Drive](https://drive.google.com/drive/u/0/folders/12qDPqV8lWgdL0x6N49uzZqnUE4n2fpUo) | One-page submission summary |
| Retention playbook | [Google Drive](https://drive.google.com/drive/u/0/folders/12qDPqV8lWgdL0x6N49uzZqnUE4n2fpUo) | Phased promo sunset + ICP with trade-offs |
| SQL results report | [Google Drive](https://drive.google.com/drive/u/0/folders/12qDPqV8lWgdL0x6N49uzZqnUE4n2fpUo) | Annotated query outputs |
| EDA visuals (6 pages) | [Google Drive](https://drive.google.com/drive/u/0/folders/12qDPqV8lWgdL0x6N49uzZqnUE4n2fpUo) | Univariate, bivariate, correlation, category, loyalty analysis |

---

## Skills demonstrated

`Python` `Pandas` `Feature Engineering` `SQL` `Customer Segmentation` `RFM Analysis` `Power BI` `Business Analytics` `Retention Strategy` `A/B Test Design`

---

*Project completed as part of a consulting analytics programme — 2025*
