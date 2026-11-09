"""Application entry point for the customer churn intelligence project."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_churn_model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ---------------------------------
# LOAD DATA AND MODEL
# ---------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    return df


try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.info(
        "Check that the model exists in models/ and the "
        "dataset exists in data/raw/."
    )
    st.stop()


# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("📊 Churn Intelligence")
st.sidebar.caption("Customer Retention Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Churn Prediction",
        "Customer Analytics",
        "Retention Recommendations"
    ]
)

st.sidebar.divider()
st.sidebar.success("Machine Learning Model Loaded")


# ---------------------------------
# DASHBOARD OVERVIEW
# ---------------------------------

if page == "Dashboard Overview":

    st.title("Customer Churn Intelligence")
    st.caption("Understand customer behaviour and identify churn risk.")

    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = (df["Churn"] == "No").sum()
    churn_rate = churned / total_customers * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Churn Distribution")

        churn_data = df["Churn"].value_counts().reset_index()
        churn_data.columns = ["Status", "Customers"]

        fig = px.pie(
            churn_data,
            names="Status",
            values="Customers",
            hole=0.5,
            title="Churn vs Retained Customers"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Churn by Contract Type")

        contract_data = (
            df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Contract-wise Churn"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Charges Distribution")

    fig = px.histogram(
        df,
        x="MonthlyCharges",
        color="Churn",
        nbins=30,
        title="Monthly Charges by Churn Status"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


# ---------------------------------
# CHURN PREDICTION
# ---------------------------------

elif page == "Churn Prediction":

    st.title("🔮 Customer Churn Prediction")
    st.write(
        "Enter customer details to estimate their churn probability."
    )

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender", ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen", [0, 1]
            )

            partner = st.selectbox(
                "Has Partner", ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Has Dependents", ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (Months)", 0, 72, 12
            )

            phone = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )

        with col2:
            multiple = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

        with col3:
            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=float(monthly * tenure)
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        try:
            probability = model.predict_proba(customer)[0][1]
            prediction = model.predict(customer)[0]

            st.divider()
            st.subheader("Prediction Result")

            c1, c2 = st.columns(2)

            c1.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

            c2.metric(
                "Predicted Class",
                "Likely to Churn" if prediction == 1
                else "Likely to Stay"
            )

            st.progress(float(probability))

            if probability >= 0.70:
                st.error("High Churn Risk")
            elif probability >= 0.40:
                st.warning("Medium Churn Risk")
            else:
                st.success("Low Churn Risk")

            st.caption(
                "Risk bands are dashboard thresholds. "
                "They are not independently calibrated risk categories."
            )

            st.subheader("Suggested Retention Actions")

            if probability >= 0.40:

                if contract == "Month-to-month":
                    st.write(
                        "• Offer a suitable long-term contract."
                    )

                if payment == "Electronic check":
                    st.write(
                        "• Offer assistance with automatic payment options."
                    )

                if tenure < 12:
                    st.write(
                        "• Consider an onboarding or loyalty offer."
                    )

                if tech == "No":
                    st.write(
                        "• Offer technical support assistance."
                    )

                st.write(
                    "• Contact the customer and understand their concerns."
                )

            else:
                st.write(
                    "• Maintain service quality and regular engagement."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------------------------
# CUSTOMER ANALYTICS
# ---------------------------------

elif page == "Customer Analytics":

    st.title("📈 Customer Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Internet Service")

        internet_data = (
            df.groupby(["InternetService", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            internet_data,
            x="InternetService",
            y="Customers",
            color="Churn",
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Payment Method")

        payment_data = (
            df.groupby(["PaymentMethod", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            payment_data,
            x="PaymentMethod",
            y="Customers",
            color="Churn",
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tenure vs Monthly Charges")

    fig = px.scatter(
        df,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        opacity=0.6
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------
# RETENTION RECOMMENDATIONS
# ---------------------------------

elif page == "Retention Recommendations":

    st.title("💡 Retention Intelligence")

    churned_df = df[df["Churn"] == "Yes"]

    st.metric(
        "Historical Churned Customers",
        len(churned_df)
    )

    st.subheader("Common Churn Patterns")

    recommendations = [
        (
            "Month-to-month contracts",
            "Consider offering annual contract discounts."
        ),
        (
            "Electronic check payments",
            "Encourage convenient automatic payment methods."
        ),
        (
            "New customers",
            "Improve onboarding and early customer engagement."
        ),
        (
            "Customers without tech support",
            "Promote technical support services."
        ),
        (
            "High monthly charges",
            "Review pricing and offer suitable service packages."
        )
    ]

    for title, description in recommendations:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    st.info(
        "These are general business recommendations based on "
        "common churn patterns. They are not individual SHAP explanations."
    )

    st.subheader("Historical Churned Customer Records")

    st.dataframe(
        churned_df.head(20),
        use_container_width=True
    )

st.sidebar.caption("Customer Churn Intelligence | M.Sc Data Science Project")