
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Health Insurance Analytics",
    page_icon="🏥",
    layout="wide"
)

sns.set_theme(style="darkgrid")

# ==========================================
# 2. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR / "new_insurance_data.csv")

# Standardize column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
    .str.strip("_")
    .str.lower()
)

# Convert numerical columns
numeric_columns = [
    "age", "bmi", "children", "claim_amount",
    "past_consultations", "num_of_steps",
    "hospital_expenditure",
    "number_of_past_hospitalizations",
    "anual_salary", "charges"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "charges" not in df.columns:
    st.error("The dataset must contain a charges column.")
    st.stop()

# ==========================================
# 3. HEADER
# ==========================================

st.title("🏥 HEALTH INSURANCE ANALYTICS")
st.caption(
    "Machine Learning | Exploratory Data Analysis | "
    "Predictive Analytics"
)

st.divider()

# ==========================================
# 4. SIDEBAR FILTERS
# ==========================================

st.sidebar.title("Dashboard Filters")

filtered_df = df.copy()

if "region" in df.columns:
    regions = ["All"] + sorted(
        df["region"].dropna().unique().tolist()
    )

    selected_region = st.sidebar.selectbox(
        "Select Region", regions
    )

    if selected_region != "All":
        filtered_df = filtered_df[
            filtered_df["region"] == selected_region
        ]

if "smoker" in df.columns:
    smokers = ["All"] + sorted(
        df["smoker"].dropna().unique().tolist()
    )

    selected_smoker = st.sidebar.selectbox(
        "Smoking Status", smokers
    )

    if selected_smoker != "All":
        filtered_df = filtered_df[
            filtered_df["smoker"] == selected_smoker
        ]

if "age" in df.columns and df["age"].notna().any():
    min_age = int(df["age"].min())
    max_age = int(df["age"].max())

    age_range = st.sidebar.slider(
        "Select Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    filtered_df = filtered_df[
        filtered_df["age"].between(
            age_range[0], age_range[1]
        )
    ]

st.sidebar.divider()
st.sidebar.caption("Health Insurance Analytics Dashboard")

# ==========================================
# 5. KPI CARDS
# ==========================================

st.subheader("📊 Dashboard Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Total Customers",
        f"{len(filtered_df):,}"
    )

with c2:
    st.metric(
        "Average Insurance Charges",
        f"{filtered_df['charges'].mean():,.2f}"
    )

with c3:
    if "age" in filtered_df.columns:
        st.metric(
            "Average Age",
            f"{filtered_df['age'].mean():.1f} years"
        )

with c4:
    if "bmi" in filtered_df.columns:
        st.metric(
            "Average BMI",
            f"{filtered_df['bmi'].mean():.2f}"
        )

st.divider()

# ==========================================
# 6. SIX VISUALIZATION OBJECTIVES
# ==========================================

st.header("📈 Exploratory Data Analysis")

st.caption(
    "Six analytical views of customer demographics, "
    "health factors and insurance charges."
)

# ---------- OBJECTIVE 1: AGE DISTRIBUTION ----------

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Age Distribution")

    if "age" in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            filtered_df["age"].dropna(),
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customer Age Distribution")

        st.pyplot(fig)
        plt.close(fig)

# ---------- OBJECTIVE 2: BMI VS CHARGES ----------

with col2:
    st.subheader("2. BMI vs Insurance Charges")

    if "bmi" in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))

        sns.scatterplot(
            data=filtered_df,
            x="bmi",
            y="charges",
            hue="smoker" if "smoker" in filtered_df.columns else None,
            ax=ax
        )

        ax.set_title("BMI vs Insurance Charges")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Insurance Charges")

        st.pyplot(fig)
        plt.close(fig)

# ---------- OBJECTIVE 3: SMOKING ANALYSIS ----------

col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Smoking Status Analysis")

    if "smoker" in filtered_df.columns:
        smoker_data = (
            filtered_df.groupby("smoker")["charges"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        smoker_data.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Average Charges by Smoking Status")
        ax.set_xlabel("Smoking Status")
        ax.set_ylabel("Average Charges")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

# ---------- OBJECTIVE 4: REGION ANALYSIS ----------

with col4:
    st.subheader("4. Region-wise Insurance Charges")

    if "region" in filtered_df.columns:
        region_data = (
            filtered_df.groupby("region")["charges"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        region_data.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Average Charges by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Average Charges")
        ax.tick_params(axis="x", rotation=20)

        st.pyplot(fig)
        plt.close(fig)

# ---------- OBJECTIVE 5: CHILDREN ANALYSIS ----------

col5, col6 = st.columns(2)

with col5:
    st.subheader("5. Children vs Insurance Charges")

    if "children" in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))

        sns.boxplot(
            data=filtered_df,
            x="children",
            y="charges",
            ax=ax
        )

        ax.set_title("Children vs Insurance Charges")
        ax.set_xlabel("Number of Children")
        ax.set_ylabel("Insurance Charges")

        st.pyplot(fig)
        plt.close(fig)

# ---------- OBJECTIVE 6: CORRELATION HEATMAP ----------

with col6:
    st.subheader("6. Correlation Heatmap")

    numeric_df = filtered_df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        ax.set_title("Feature Correlation Analysis")

        st.pyplot(fig)
        plt.close(fig)

st.divider()

# ==========================================
# 7. ADDITIONAL DATASET INFORMATION
# ==========================================

st.header("📋 Dataset Explorer")

with st.expander("View Complete Dataset"):
    st.dataframe(
        filtered_df,
        use_container_width=True
    )

with st.expander("View Statistical Summary"):
    st.dataframe(
        filtered_df.describe(),
        use_container_width=True
    )

with st.expander("Missing Values Analysis"):
    missing = filtered_df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

# Download option
csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Download Filtered Dataset",
    data=csv_data,
    file_name="insurance_analysis.csv",
    mime="text/csv"
)

# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Health Insurance Analytics & Prediction | "
    "Developed using Python, Pandas, Seaborn and Streamlit"
)
