import streamlit as st
st.set_page_config(page_title="Sales Analysis & Prediction Dashboard", layout="wide")
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# For Prophet and XGBoost (simulated implementations)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ------------------------
# Configuration and Constants
# ------------------------
class Config:
    FORECAST_DAYS = [30, 60, 90, 180, 365]
    SEASONALITY_PERIODS = {
        'daily': 7,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4
    }
    BUSINESS_METRICS = {
        'revenue_growth_threshold': 0.1,
        'customer_retention_threshold': 0.8,
        'seasonal_variance_threshold': 0.15
    }

# ------------------------
# Data Loading and Preparation
# ------------------------
@st.cache_data
def load_and_prepare_sales_data():
    """Load your actual sales data"""
    try:
        # Load the actual CSV file
        df = pd.read_csv("sales_data1.csv")
        
        # Convert invoicedate to datetime
        df['invoicedate'] = pd.to_datetime(df['invoicedate'])

        df = df.dropna(subset=['customerid'])  # Remove NaN customer IDs
        df = df[df['customerid'] != 'nan']     # Remove string 'nan' values
        #df = df[df['customerid'].str.strip() != ''] 
        
        # Ensure proper data types based on your dataset
        df['invoiceno'] = df['invoiceno'].astype(str)
        df['stockcode'] = df['stockcode'].astype(str)
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df['customerid'] = df['customerid'].astype(str)
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        
        # Remove any rows with missing critical data
        df = df.dropna(subset=['invoicedate', 'revenue', 'quantity'])
        
        # Filter out negative quantities and zero prices (returns/cancellations)
        df = df[(df['quantity'] > 0) & (df['unitprice'] > 0)]
        
        # Add derived columns if they don't exist
        if 'year' not in df.columns:
            df['year'] = df['invoicedate'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['invoicedate'].dt.month
        if 'day' not in df.columns:
            df['day'] = df['invoicedate'].dt.day
        if 'weekday' not in df.columns:
            df['weekday'] = df['invoicedate'].dt.day_name()
        
        return df
        
    except FileNotFoundError:
        st.error("sales_data.csv file not found. Please ensure the file is in the same directory as this script.")
        st.info("Creating sample data for demonstration...")
        return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if the actual file is not available"""
    np.random.seed(42)
    
    # Create sample data matching your structure
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    countries = ['United Kingdom', 'Germany', 'France', 'Netherlands', 'EIRE']
    
    sample_data = []
    for i, date in enumerate(date_range):
        # Generate multiple transactions per day
        daily_transactions = np.random.poisson(10) + 5
        
        for j in range(daily_transactions):
            invoice_id = f"{540000 + i*100 + j}"
            stock_code = f"{np.random.randint(10000, 99999)}"
            description = f"Product {np.random.randint(1, 1000)}"
            quantity = np.random.poisson(5) + 1
            unit_price = np.random.uniform(0.5, 50.0)
            customer_id = f"{15000 + np.random.randint(0, 5000)}"
            country = np.random.choice(countries)
            revenue = quantity * unit_price
            
            sample_data.append({
                'invoiceno': invoice_id,
                'stockcode': stock_code,
                'description': description,
                'quantity': quantity,
                'invoicedate': date,
                'unitprice': unit_price,
                'customerid': customer_id,
                'country': country,
                'revenue': revenue,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'weekday': date.strftime('%A')
            })
    
    df = pd.DataFrame(sample_data)
    return df

def prepare_time_series_data(df, aggregation='daily'):
    """Prepare data for time series analysis"""
    df_copy = df.copy()
    df_copy['date'] = df_copy['invoicedate'].dt.date
    
    if aggregation == 'daily':
        time_series = df_copy.groupby('date').agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'invoiceno': 'nunique',
            'customerid': 'nunique'
        }).reset_index()
    elif aggregation == 'weekly':
        df_copy['week'] = df_copy['invoicedate'].dt.to_period('W')
        time_series = df_copy.groupby('week').agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'invoiceno': 'nunique',
            'customerid': 'nunique'
        }).reset_index()
        time_series['date'] = time_series['week'].dt.start_time.dt.date
        time_series = time_series[['date', 'revenue', 'quantity', 'invoiceno', 'customerid']]
    elif aggregation == 'monthly':
        df_copy['month_period'] = df_copy['invoicedate'].dt.to_period('M')
        time_series = df_copy.groupby('month_period').agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'invoiceno': 'nunique',
            'customerid': 'nunique'
        }).reset_index()
        time_series['date'] = time_series['month_period'].dt.start_time.dt.date
        time_series = time_series[['date', 'revenue', 'quantity', 'invoiceno', 'customerid']]
    
    time_series.columns = ['date', 'revenue', 'quantity', 'orders', 'customers']
    return time_series.sort_values('date')

# ------------------------
# Prophet + XGBoost Hybrid Model
# ------------------------
class HybridSalesPredictor:
    def __init__(self):
        self.prophet_model = None
        self.xgboost_model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
    def create_features(self, df):
        """Create features for XGBoost model"""
        df_features = df.copy()
        df_features['date'] = pd.to_datetime(df_features['date'])
        
        # Time-based features
        df_features['day_of_week'] = df_features['date'].dt.dayofweek
        df_features['day_of_month'] = df_features['date'].dt.day
        df_features['day_of_year'] = df_features['date'].dt.dayofyear
        df_features['week_of_year'] = df_features['date'].dt.isocalendar().week
        df_features['month'] = df_features['date'].dt.month
        df_features['quarter'] = df_features['date'].dt.quarter
        df_features['year'] = df_features['date'].dt.year
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            df_features[f'revenue_lag_{lag}'] = df_features['revenue'].shift(lag)
            
        # Rolling features
        for window in [7, 14, 30]:
            df_features[f'revenue_rolling_mean_{window}'] = df_features['revenue'].rolling(window=window).mean()
            df_features[f'revenue_rolling_std_{window}'] = df_features['revenue'].rolling(window=window).std()
            
        # Fill NaN values
        df_features = df_features.fillna(method='bfill').fillna(0)
        
        return df_features
    
    def fit(self, time_series_data):
        """Fit both Prophet and XGBoost models"""
        try:
            # Prepare Prophet data
            prophet_data = time_series_data[['date', 'revenue']].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            
            # Simulate Prophet fitting (replace with actual Prophet when available)
            if PROPHET_AVAILABLE:
                self.prophet_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                self.prophet_model.fit(prophet_data)
            
            # Prepare XGBoost features
            features_df = self.create_features(time_series_data)
            
            # Select feature columns (excluding target and date)
            feature_cols = [col for col in features_df.columns if col not in ['date', 'revenue', 'orders', 'customers', 'quantity']]
            X = features_df[feature_cols]
            y = features_df['revenue']
            
            # Remove any remaining NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Fit XGBoost (simulate if not available)
            if XGBOOST_AVAILABLE:
                self.xgboost_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                self.xgboost_model.fit(X_scaled, y)
            else:
                # Use RandomForest as fallback
                from sklearn.ensemble import RandomForestRegressor
                self.xgboost_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.xgboost_model.fit(X_scaled, y)
            
            self.feature_columns = feature_cols
            self.is_fitted = True
            
            return True
            
        except Exception as e:
            st.error(f"Error fitting models: {str(e)}")
            return False
    
    def predict(self, time_series_data, forecast_days=30):
        """Make hybrid predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Generate future dates
            last_date = pd.to_datetime(time_series_data['date'].max())
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Prophet predictions (simulated)
            prophet_predictions = []
            if PROPHET_AVAILABLE and self.prophet_model:
                future_df = pd.DataFrame({'ds': future_dates})
                prophet_forecast = self.prophet_model.predict(future_df)
                prophet_predictions = prophet_forecast['yhat'].tolist()
            else:
                # Simulate Prophet-like trend + seasonality
                base_value = time_series_data['revenue'].mean()
                trend = np.linspace(0, 0.1 * base_value, forecast_days)
                seasonal = base_value * 0.1 * np.sin(2 * np.pi * np.arange(forecast_days) / 7)
                noise = np.random.normal(0, base_value * 0.05, forecast_days)
                prophet_predictions = base_value + trend + seasonal + noise
            
            # XGBoost predictions
            # Create future features based on recent data patterns
            recent_data = time_series_data.tail(30).copy()
            future_predictions = []
            
            for i, future_date in enumerate(future_dates):
                # Create a row for the future date
                future_row = pd.DataFrame({
                    'date': [future_date],
                    'revenue': [recent_data['revenue'].mean()],  # Placeholder
                    'quantity': [recent_data['quantity'].mean()],
                    'orders': [recent_data['orders'].mean()],
                    'customers': [recent_data['customers'].mean()]
                })
                
                # Create features
                extended_data = pd.concat([recent_data, future_row], ignore_index=True)
                features_df = self.create_features(extended_data)
                
                # Get the last row (our future prediction)
                future_features = features_df[self.feature_columns].iloc[-1:].values
                future_features_scaled = self.feature_scaler.transform(future_features)
                
                # Predict
                xgb_pred = self.xgboost_model.predict(future_features_scaled)[0]
                future_predictions.append(xgb_pred)
                
                # Update recent_data for next iteration
                recent_data = recent_data.iloc[1:]  # Remove oldest
                recent_data = pd.concat([recent_data, pd.DataFrame({
                    'date': [future_date],
                    'revenue': [xgb_pred],
                    'quantity': [recent_data['quantity'].mean()],
                    'orders': [recent_data['orders'].mean()],
                    'customers': [recent_data['customers'].mean()]
                })], ignore_index=True)
            
            # Combine predictions (weighted average)
            if prophet_predictions:
                combined_predictions = [
                    0.6 * p + 0.4 * x for p, x in zip(prophet_predictions, future_predictions)
                ]
            else:
                combined_predictions = future_predictions
            
            # Create prediction dataframe
            predictions_df = pd.DataFrame({
                'date': future_dates,
                'prophet_prediction': prophet_predictions if prophet_predictions else [0] * forecast_days,
                'xgboost_prediction': future_predictions,
                'hybrid_prediction': combined_predictions,
                'confidence_lower': [p * 0.85 for p in combined_predictions],
                'confidence_upper': [p * 1.15 for p in combined_predictions]
            })
            
            return predictions_df
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None

# ------------------------
# Business Intelligence Functions
# ------------------------
def calculate_business_metrics(df):
    """Calculate key business metrics"""
    metrics = {}
    
    # Revenue metrics
    metrics['total_revenue'] = df['revenue'].sum()
    metrics['avg_daily_revenue'] = df.groupby(df['invoicedate'].dt.date)['revenue'].sum().mean()
    metrics['revenue_growth'] = calculate_growth_rate(df, 'revenue')
    
    # Customer metrics
    metrics['total_customers'] = df['customerid'].nunique()
    metrics['avg_order_value'] = df['revenue'].sum() / df['invoiceno'].nunique()
    metrics['customer_lifetime_value'] = df.groupby('customerid')['revenue'].sum().mean()
    
    # Product metrics
    metrics['total_products'] = df['stockcode'].nunique()
    metrics['avg_quantity_per_order'] = df['quantity'].mean()
    
    # Geographic metrics
    metrics['countries_served'] = df['country'].nunique()
    metrics['top_country'] = df.groupby('country')['revenue'].sum().idxmax()
    
    return metrics

def calculate_growth_rate(df, column, periods=30):
    """Calculate growth rate over specified periods"""
    df_temp = df.copy()
    df_temp['date'] = df_temp['invoicedate'].dt.date
    daily_values = df_temp.groupby('date')[column].sum().sort_index()
    
    if len(daily_values) < periods * 2:
        return 0
    
    recent_avg = daily_values.tail(periods).mean()
    previous_avg = daily_values.iloc[-periods*2:-periods].mean()
    
    if previous_avg == 0:
        return 0
    
    growth_rate = (recent_avg - previous_avg) / previous_avg
    return growth_rate

def calculate_rfm_analysis(df):
    """Calculate RFM (Recency, Frequency, Monetary) analysis"""
    current_date = df['invoicedate'].max()
    
    rfm = df.groupby('customerid').agg({
        'invoicedate': lambda x: (current_date - x.max()).days,  # Recency
        'invoiceno': 'nunique',  # Frequency
        'revenue': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customerid', 'recency', 'frequency', 'monetary']
    
    # Calculate RFM scores (1-5 scale)
    rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
    
    rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    
    return rfm

def segment_customers(rfm_score):
    """Customer segmentation based on RFM scores"""
    if rfm_score in ['555', '554', '544', '545', '454', '455', '445']:
        return 'Champions'
    elif rfm_score in ['543', '444', '435', '355', '354', '345', '344', '335']:
        return 'Loyal Customers'
    elif rfm_score in ['512', '511', '422', '421', '412', '411', '311']:
        return 'Potential Loyalists'
    elif rfm_score in ['155', '154', '144', '214', '215', '115', '114']:
        return 'New Customers'
    elif rfm_score in ['331', '321', '231', '241', '251']:
        return 'Need Attention'
    elif rfm_score in ['133', '134', '143', '244', '334', '343', '344']:
        return 'At Risk'
    elif rfm_score in ['111', '112', '121', '131', '141', '151']:
        return 'Cannot Lose Them'
    else:
        return 'Lost'

def calculate_churn_risk(df):
    """Calculate customer churn risk"""
    df_temp = df.copy()
    df_temp['date'] = df_temp['invoicedate'].dt.date
    
    # Customer last purchase analysis
    customer_last_purchase = df_temp.groupby('customerid')['date'].max().reset_index()
    customer_last_purchase['days_since_last_purchase'] = (df_temp['date'].max() - customer_last_purchase['date']).apply(lambda x: x.days)
    
    # Define churn risk categories
    def classify_churn_risk(days):
        if days <= 30:
            return 'Low Risk'
        elif days <= 90:
            return 'Medium Risk'
        elif days <= 180:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    customer_last_purchase['churn_risk'] = customer_last_purchase['days_since_last_purchase'].apply(classify_churn_risk)
    
    return customer_last_purchase

def generate_business_insights(df, predictions_df=None):
    """Generate actionable business insights"""
    insights = []
    
    # Seasonal analysis
    monthly_revenue = df.groupby(df['invoicedate'].dt.month)['revenue'].sum()
    peak_month = monthly_revenue.idxmax()
    low_month = monthly_revenue.idxmin()
    
    insights.append({
        'category': 'Seasonality',
        'insight': f'Peak sales occur in month {peak_month}, while month {low_month} shows lowest sales',
        'action': f'Plan inventory and marketing campaigns around month {peak_month} peak',
        'priority': 'Medium'
    })
    
    # Customer behavior
    customer_stats = df.groupby('customerid').agg({
        'revenue': 'sum',
        'invoiceno': 'nunique',
        'invoicedate': ['min', 'max']
    })
    customer_stats.columns = ['total_revenue', 'total_orders', 'first_order', 'last_order']
    high_value_customers = len(customer_stats[customer_stats['total_revenue'] > customer_stats['total_revenue'].quantile(0.8)])
    
    insights.append({
        'category': 'Customer Segmentation',
        'insight': f'{high_value_customers} customers represent top 20% of revenue',
        'action': 'Implement VIP customer program and personalized retention strategies',
        'priority': 'High'
    })
    
    # Product performance
    product_performance = df.groupby('description')['revenue'].sum().sort_values(ascending=False)
    top_products = len(product_performance[product_performance > product_performance.quantile(0.9)])
    
    insights.append({
        'category': 'Product Strategy',
        'insight': f'Top {top_products} products drive majority of sales',
        'action': 'Focus marketing and inventory on high-performing products',
        'priority': 'Medium'
    })
    
    # Churn analysis
    churn_data = calculate_churn_risk(df)
    high_risk_customers = len(churn_data[churn_data['churn_risk'].isin(['High Risk', 'Very High Risk'])])
    
    insights.append({
        'category': 'Customer Retention',
        'insight': f'{high_risk_customers} customers at high risk of churning',
        'action': 'Launch targeted retention campaign for at-risk customers',
        'priority': 'High'
    })
    
    # Forecast insights
    if predictions_df is not None:
        current_avg = df.groupby(df['invoicedate'].dt.date)['revenue'].sum().tail(30).mean()
        predicted_avg = predictions_df['hybrid_prediction'].mean()
        predicted_growth = (predicted_avg / current_avg - 1) * 100 if current_avg > 0 else 0
        
        insights.append({
            'category': 'Future Outlook',
            'insight': f'Projected {predicted_growth:.1f}% change in daily revenue',
            'action': 'Adjust resource allocation based on predicted demand',
            'priority': 'Medium' if abs(predicted_growth) < 10 else 'High'
        })
    
    return insights

# ------------------------
# Streamlit App
# ------------------------
def main():
    st.title("🎯 Sales Analysis & Prediction Dashboard")
    st.markdown("Advanced analytics and forecasting for retail sales data")
    
    # Load data
    with st.spinner("Loading sales data..."):
        df = load_and_prepare_sales_data()
    
    if df is not None and not df.empty:
        st.success(f"✅ Data loaded successfully! {len(df):,} records from {df['invoicedate'].min().strftime('%Y-%m-%d')} to {df['invoicedate'].max().strftime('%Y-%m-%d')}")
        
        # Sidebar filters
        st.sidebar.header("📊 Filters & Settings")
        
        # Date range filter
        min_date = df['invoicedate'].min().date()
        max_date = df['invoicedate'].max().date()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Country filter
        countries = ['All'] + sorted(df['country'].unique().tolist())
        selected_country = st.sidebar.selectbox("Select Country", countries)
        
        # Apply filters
        if len(date_range) == 2:
            df_filtered = df[
                (df['invoicedate'].dt.date >= date_range[0]) & 
                (df['invoicedate'].dt.date <= date_range[1])
            ]
        else:
            df_filtered = df
            
        if selected_country != 'All':
            df_filtered = df_filtered[df_filtered['country'] == selected_country]
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", "🔮 Predictions", "👥 Customer Analysis", 
            "📦 Product Analysis", "💡 Business Insights"
        ])
        
        with tab1:
            st.header("Business Overview")
            
            # Key metrics
            metrics = calculate_business_metrics(df_filtered)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Revenue", f"${metrics['total_revenue']:,.2f}")
            with col2:
                st.metric("Total Customers", f"{metrics['total_customers']:,}")
            with col3:
                st.metric("Average Order Value", f"${metrics['avg_order_value']:.2f}")
            with col4:
                growth_pct = metrics['revenue_growth'] * 100
                st.metric("Revenue Growth", f"{growth_pct:.1f}%")
            
            # Time series visualization
            col1, col2 = st.columns([3, 1])
            with col2:
                aggregation = st.selectbox("Time Aggregation", ['daily', 'weekly', 'monthly'])
            
            time_series = prepare_time_series_data(df_filtered, aggregation)
            
            fig = px.line(time_series, x='date', y='revenue', 
                         title=f"Revenue Trend ({aggregation.title()})")
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Revenue"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Revenue by country
            country_revenue = df_filtered.groupby('country')['revenue'].sum().sort_values(ascending=False)
            top_5_countries = country_revenue.head(5)  # Only take top 5
            fig_country = px.bar(x=top_5_countries.index, y=top_5_countries.values,
                            title="Revenue by Top 5 Countries")
            fig_country.update_layout(
                height=400,
                xaxis_title="Country",
                yaxis_title="Revenue"
            )
            st.plotly_chart(fig_country, use_container_width=True)
        
        with tab2:
            st.header("Sales Predictions")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                forecast_days = st.selectbox("Forecast Period", Config.FORECAST_DAYS, index=0)
                # Removed the aggregation selectbox - now hardcoded to weekly
            
            # Prepare time series for prediction (hardcoded to weekly)
            aggregation_pred = 'weekly'  # Hardcoded value
            time_series_pred = prepare_time_series_data(df_filtered, aggregation_pred)
            
            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Training models and generating predictions..."):
                    # Initialize and train model
                    predictor = HybridSalesPredictor()
                    
                    if predictor.fit(time_series_pred):
                        predictions = predictor.predict(time_series_pred, forecast_days)
                        
                        if predictions is not None:
                            # Combine historical and predicted data for visualization
                            historical = time_series_pred[['date', 'revenue']].copy()
                            historical['type'] = 'Historical'
                            historical['lower'] = historical['revenue']
                            historical['upper'] = historical['revenue']
                            
                            predicted = predictions[['date', 'hybrid_prediction', 'confidence_lower', 'confidence_upper']].copy()
                            predicted.columns = ['date', 'revenue', 'lower', 'upper']
                            predicted['type'] = 'Predicted'
                            
                            # Plot results
                            fig = go.Figure()
                            
                            # Historical data
                            fig.add_trace(go.Scatter(
                                x=historical['date'], y=historical['revenue'],
                                mode='lines', name='Historical',
                                line=dict(color='blue')
                            ))
                            
                            # Predictions
                            fig.add_trace(go.Scatter(
                                x=predicted['date'], y=predicted['revenue'],
                                mode='lines', name='Predicted',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            # Confidence interval
                            fig.add_trace(go.Scatter(
                                x=predicted['date'], y=predicted['upper'],
                                mode='lines', line=dict(width=0),
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=predicted['date'], y=predicted['lower'],
                                mode='lines', line=dict(width=0),
                                fill='tonexty', fillcolor='rgba(255,0,0,0.2)',
                                name='Confidence Interval'
                            ))
                            
                            fig.update_layout(
                                title=f"Weekly Sales Forecast - Next {forecast_days} Days",
                                xaxis_title="Date",
                                yaxis_title="Revenue",
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Prediction summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                avg_predicted = predictions['hybrid_prediction'].mean()
                                st.metric("Avg Weekly Revenue (Predicted)", f"${avg_predicted:,.2f}")
                            with col2:
                                total_predicted = predictions['hybrid_prediction'].sum()
                                st.metric("Total Predicted Revenue", f"${total_predicted:,.2f}")
                            with col3:
                                current_avg = time_series_pred['revenue'].tail(4).mean()  # Last 4 weeks instead of 30 days
                                growth = (avg_predicted / current_avg - 1) * 100 if current_avg > 0 else 0
                                st.metric("Expected Growth", f"{growth:.1f}%")
                            
                            # Detailed predictions table
                            st.subheader("Detailed Weekly Predictions")
                            predictions_display = predictions.copy()
                            predictions_display['date'] = pd.to_datetime(predictions_display['date']).dt.strftime('%Y-%m-%d')
                            predictions_display = predictions_display.round(2)
                            st.dataframe(predictions_display, use_container_width=True)
                            
                            # Store predictions for insights
                            st.session_state['predictions'] = predictions
                    else:
                        st.error("Failed to train prediction models. Please check your data.")
        
        with tab3:
            st.header("Customer Analysis")
            
            # RFM Analysis
            st.subheader("RFM Analysis")
            rfm_data = calculate_rfm_analysis(df_filtered)
            rfm_data['segment'] = rfm_data['rfm_score'].apply(segment_customers)
            
            # Customer segments distribution
            segment_counts = rfm_data['segment'].value_counts()
            fig_segments = px.pie(values=segment_counts.values, names=segment_counts.index,
                                 title="Customer Segments Distribution")
            st.plotly_chart(fig_segments, use_container_width=True)
            
            # RFM scatter plot
            fig_rfm = px.scatter(rfm_data, x='frequency', y='monetary', 
                               color='segment', size='recency',
                               title="RFM Analysis - Frequency vs Monetary Value",
                               hover_data=['customerid', 'recency'])
            st.plotly_chart(fig_rfm, use_container_width=True)
            
            # Top customers
            st.subheader("Top 10 Customers by Revenue")
            top_customers = df_filtered.groupby('customerid').agg({
                'revenue': 'sum',
                'invoiceno': 'nunique',
                'quantity': 'sum',
                'invoicedate': ['min', 'max']
            }).round(2)
            top_customers.columns = ['Total Revenue', 'Orders', 'Items Purchased', 'First Order', 'Last Order']
            top_customers = top_customers.sort_values('Total Revenue', ascending=False).head(10)
            st.dataframe(top_customers, use_container_width=True)
            
            # Customer churn analysis
            st.subheader("Customer Churn Risk Analysis")
            churn_data = calculate_churn_risk(df_filtered)
            churn_summary = churn_data['churn_risk'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_churn = px.bar(x=churn_summary.index, y=churn_summary.values,
                                  title="Customer Churn Risk Distribution")
                fig_churn.update_layout(
                height=400,
                xaxis_title="Risk",
                yaxis_title="Customer"
            )
                st.plotly_chart(fig_churn, use_container_width=True)
            
            with col2:
                st.write("**Churn Risk Definitions:**")
                st.write("- **Low Risk**: Last purchase ≤ 30 days")
                st.write("- **Medium Risk**: 31-90 days")
                st.write("- **High Risk**: 91-180 days")
                st.write("- **Very High Risk**: > 180 days")
                
                high_risk_count = len(churn_data[churn_data['churn_risk'].isin(['High Risk', 'Very High Risk'])])
                st.warning(f"⚠️ {high_risk_count} customers at high churn risk!")
        
        with tab4:
            st.header("Product Analysis")
            
            # Remove product outliers and invalid descriptions
            df_products = df_filtered.copy()
            # Remove products with suspicious names or very low sales
            outlier_keywords = ['PAPER CRAFT ,LITTLE BIRDIE', 'test', 'sample', 'damaged', 'lost']
            for keyword in outlier_keywords:
                df_products = df_products[~df_products['description'].str.contains(keyword, na=False)]

            # Remove products with extremely low quantities or prices
            df_products = df_products[
                (df_products['quantity'] >= 70000) & 
                (df_products['unitprice'] >= 0.01) &
                (df_products['revenue'] >= 0.01)
            ]

            # Then use df_products instead of df_filtered for product analysis:
            product_stats = df_products.groupby('description').agg({
                'revenue': 'sum',
                'quantity': 'sum', 
                'unitprice': 'mean',
                'invoiceno': 'nunique'
            }).round(2)
            # Product performance
            product_stats = df_filtered.groupby('description').agg({
                'revenue': 'sum',
                'quantity': 'sum',
                'unitprice': 'mean',
                'invoiceno': 'nunique'
            }).round(2)
            product_stats.columns = ['Total Revenue', 'Units Sold', 'Avg Price', 'Orders']
            product_stats = product_stats.sort_values('Total Revenue', ascending=False)
            
            # Top products
            st.subheader("Products by Revenue")
            top_products = product_stats.head(15)
            
            fig_products = px.bar(x=top_products.index, y=top_products['Total Revenue'],
                                 title="Top 15 Products by Revenue")
            fig_products.update_xaxes(tickangle=45)
            fig_products.update_layout(
                height=500,
                xaxis_title="Products",
                yaxis_title="Revenue"
            )
            st.plotly_chart(fig_products, use_container_width=True)
            
            # Product performance matrix
            st.subheader("Product Performance Analysis")
            
            # Calculate product metrics
            product_stats['Revenue per Order'] = product_stats['Total Revenue'] / product_stats['Orders']
            product_stats['Revenue Rank'] = product_stats['Total Revenue'].rank(ascending=False)
            product_stats['Volume Rank'] = product_stats['Units Sold'].rank(ascending=False)
            
            # Scatter plot: Revenue vs Volume
            fig_matrix = px.scatter(product_stats.head(60), 
                                   x='Units Sold', y='Total Revenue',
                                   size='Avg Price', hover_name=product_stats.head(60).index,
                                   title="Product Performance Matrix (Top 50 Products)")
            fig_matrix.update_layout(height=400,width = 10)
            st.plotly_chart(fig_matrix, use_container_width=True)
            
            # Monthly product trends
            st.subheader("Monthly Product Category Trends")
            monthly_products = df_filtered.groupby([df_filtered['invoicedate'].dt.to_period('M'), 'description'])['revenue'].sum().unstack(fill_value=0)
            
            # Show trends for top 5 products
            top_5_products = product_stats.head(5).index
            monthly_top5 = monthly_products[top_5_products]
            monthly_top5.index = monthly_top5.index.astype(str)
            
            fig_trends = px.line(monthly_top5.reset_index(), x='invoicedate', 
                               y=monthly_top5.columns.tolist(),
                               title="Monthly Revenue Trends - Top 5 Products")
            fig_trends.update_layout(height=600)
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Product summary table
            st.subheader("Product Performance Summary")
            product_display = product_stats.head(25).copy()
            st.dataframe(product_display, use_container_width=True)
        
        with tab5:
            st.header("Business Insights & Recommendations")
            
            # Generate insights
            predictions_for_insights = st.session_state.get('predictions', None)
            insights = generate_business_insights(df_filtered, predictions_for_insights)
            
            # Priority insights
            high_priority = [i for i in insights if i['priority'] == 'High']
            medium_priority = [i for i in insights if i['priority'] == 'Medium']
            
            if high_priority:
                st.subheader("🚨 High Priority Insights")
                for insight in high_priority:
                    with st.expander(f"{insight['category']}: {insight['insight'][:50]}..."):
                        st.write(f"**Insight:** {insight['insight']}")
                        st.write(f"**Recommended Action:** {insight['action']}")
                        st.write(f"**Priority:** {insight['priority']}")
            
            if medium_priority:
                st.subheader("⚠️ Medium Priority Insights")
                for insight in medium_priority:
                    with st.expander(f"{insight['category']}: {insight['insight'][:50]}..."):
                        st.write(f"**Insight:** {insight['insight']}")
                        st.write(f"**Recommended Action:** {insight['action']}")
                        st.write(f"**Priority:** {insight['priority']}")
            
            # Business performance summary
            st.subheader("📊 Performance Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Revenue Analysis:**")
                total_revenue = df_filtered['revenue'].sum()
                daily_avg = df_filtered.groupby(df_filtered['invoicedate'].dt.date)['revenue'].sum().mean()
                monthly_growth = calculate_growth_rate(df_filtered, 'revenue', 30) * 100
                
                st.write(f"- Total Revenue: ${total_revenue:,.2f}")
                st.write(f"- Daily Average: ${daily_avg:,.2f}")
                st.write(f"- 30-day Growth: {monthly_growth:.1f}%")
                
                # Revenue trend indicator
                if monthly_growth > 5:
                    st.success("📈 Strong growth trend")
                elif monthly_growth > 0:
                    st.info("📊 Positive growth trend")
                else:
                    st.warning("📉 Declining trend - needs attention")
            
            with col2:
                st.write("**Customer Analysis:**")
                total_customers = df_filtered['customerid'].nunique()
                avg_customer_value = df_filtered.groupby('customerid')['revenue'].sum().mean()
                repeat_customers = len(df_filtered.groupby('customerid').filter(lambda x: len(x) > 1)['customerid'].unique())
                repeat_rate = (repeat_customers / total_customers) * 100 if total_customers > 0 else 0
                
                st.write(f"- Total Customers: {total_customers:,}")
                st.write(f"- Avg Customer Value: ${avg_customer_value:.2f}")
                st.write(f"- Repeat Customer Rate: {repeat_rate:.1f}%")
                
                # Customer health indicator
                if repeat_rate > 60:
                    st.success("👥 Strong customer loyalty")
                elif repeat_rate > 40:
                    st.info("👥 Good customer retention")
                else:
                    st.warning("👥 Low retention - focus needed")
            
            # Action items
            st.subheader("🎯 Recommended Action Items")
            
            action_items = [
                "Monitor high-risk customers and implement retention campaigns",
                "Focus marketing efforts on peak seasonal periods",
                "Optimize inventory for top-performing products",
                "Develop loyalty programs for high-value customer segments",
                "Analyze and replicate success factors from top-performing regions"
            ]
            
            for i, action in enumerate(action_items, 1):
                st.write(f"{i}. {action}")
            
            # Export functionality
            st.subheader("📁 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Business Metrics"):
                    metrics_df = pd.DataFrame([metrics])
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="Download Metrics CSV",
                        data=csv,
                        file_name=f"business_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("👥 Export Customer Analysis"):
                    rfm_export = calculate_rfm_analysis(df_filtered)
                    rfm_export['segment'] = rfm_export['rfm_score'].apply(segment_customers)
                    csv = rfm_export.to_csv(index=False)
                    st.download_button(
                        label="Download Customer Analysis CSV",
                        data=csv,
                        file_name=f"customer_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("🔮 Export Predictions"):
                    if 'predictions' in st.session_state:
                        csv = st.session_state['predictions'].to_csv(index=False)
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv,
                            file_name=f"sales_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Generate predictions first to export")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Sales Analysis & Prediction Dashboard | Built with Streamlit & Advanced ML</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Failed to load data. Please check your data file and try again.")
        st.info("Make sure 'sales_data.csv' is in the same directory as this script.")

# ------------------------
# Run the app
# ------------------------
if __name__ == "__main__":
    main()