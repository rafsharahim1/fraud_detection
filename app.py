import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# For ML clustering and anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}
.high-risk {
    background-color: #ffe6e6;
    border-left-color: #ff4757;
}
.medium-risk {
    background-color: #fff3cd;
    border-left-color: #ffa502;
}
.low-risk {
    background-color: #d4edda;
    border-left-color: #2ed573;
}
</style>
""", unsafe_allow_html=True)

class FraudDetectorDashboard:
    def __init__(self):
        self.df = None
        self.pan_grouped_df = None
        
    def load_data(self, uploaded_file):
        """Load and prepare data from uploaded file"""
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                self.df = pd.read_csv(uploaded_file)
            else:
                st.error("Please upload a CSV file")
                return False
            
            # Prepare data
            self.prepare_data()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def prepare_data(self):
        """Clean and prepare the data for analysis"""
        # Convert DateTime to proper datetime format
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'], format='%d/%m/%Y %H:%M', errors='coerce')
        
        # Extract hour for time-based analysis
        self.df['Hour'] = self.df['DateTime'].dt.hour
        self.df['Date'] = self.df['DateTime'].dt.date
        
        # Clean email addresses and extract domains
        self.df['Email_Domain'] = self.df['Customer Email'].apply(self.extract_email_domain)
        
        # Convert amounts to numeric
        self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        if 'PURCHASE_AMOUNT' in self.df.columns:
            self.df['PURCHASE_AMOUNT'] = pd.to_numeric(self.df['PURCHASE_AMOUNT'], errors='coerce')
    
    def extract_email_domain(self, email):
        """Extract domain from email address"""
        if pd.isna(email) or email == '':
            return 'unknown'
        try:
            return email.split('@')[1].lower()
        except:
            return 'invalid'
    
    def calculate_entropy(self, value_counts):
        """Calculate entropy for diversity measurement"""
        if len(value_counts) <= 1:
            return 0
        probabilities = value_counts / value_counts.sum()
        return -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    def is_suspicious_email(self, email):
        """Check if email appears suspicious"""
        if pd.isna(email):
            return False
        
        email = str(email).lower()
        
        suspicious_patterns = [
            r'\d{5,}',  # Contains 5+ consecutive digits
            r'[a-z]{10,}',  # Very long random strings
            r'temp',    # Contains 'temp'
            r'test',    # Contains 'test'
            r'fake',    # Contains 'fake'
            r'[0-9]+[a-z]+[0-9]+',  # Mixed numbers and letters pattern
        ]
        
        disposable_domains = [
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'throwaway.email', 'temp-mail.org'
        ]
        
        domain = self.extract_email_domain(email)
        if domain in disposable_domains:
            return True
            
        for pattern in suspicious_patterns:
            if re.search(pattern, email):
                return True
        
        return False
    
    def calculate_fraud_score(self, unique_merchants, unique_emails, unique_domains, 
                            total_transactions, time_span_hours, domain_entropy, suspicious_emails):
        """Calculate composite fraud score"""
        score = 0
        
        if unique_merchants > 3 and total_transactions < 20:
            score += 3
        elif unique_merchants > 5:
            score += 5
        
        if unique_emails > unique_merchants:
            score += 2
        
        if domain_entropy > 2:
            score += 2
        
        if total_transactions > 5 and time_span_hours < 24:
            score += 3
        elif total_transactions > 3 and time_span_hours < 2:
            score += 4
        
        suspicious_ratio = suspicious_emails / total_transactions if total_transactions > 0 else 0
        if suspicious_ratio > 0.5:
            score += 4
        elif suspicious_ratio > 0.2:
            score += 2
        
        return min(score, 10)
    
    def create_pan_grouped_analysis(self):
        """Create detailed analysis grouped by PAN"""
        pan_groups = []
        
        for pan in self.df['PAN'].unique():
            if pd.isna(pan):
                continue
                
            pan_data = self.df[self.df['PAN'] == pan].copy()
            
            # Basic statistics
            unique_merchants = pan_data['MerchantName'].nunique()
            unique_emails = pan_data['Customer Email'].nunique()
            unique_domains = pan_data['Email_Domain'].nunique()
            unique_mobile = pan_data['Customer Mobile'].nunique()
            total_transactions = len(pan_data)
            total_amount = pan_data['Amount'].sum()
            avg_amount = pan_data['Amount'].mean()
            
            # Time analysis
            first_transaction = pan_data['DateTime'].min()
            last_transaction = pan_data['DateTime'].max()
            time_span_hours = 0
            if len(pan_data) > 1:
                time_span_hours = (last_transaction - first_transaction).total_seconds() / 3600
            
            # Transaction patterns
            successful_txns = pan_data[pan_data['Status'] == 'Success'].shape[0]
            pending_txns = pan_data[pan_data['Status'] == 'Pending'].shape[0]
            failed_txns = pan_data[pan_data['Status'].isin(['Failed', 'Declined', 'Error'])].shape[0]
            
            # Geographic and other diversity metrics
            unique_cities = pan_data['MerchantCity'].nunique()
            unique_mccs = pan_data['MCC'].nunique()
            unique_banks = pan_data['BankName'].nunique()
            
            # Email analysis
            email_domains = pan_data['Email_Domain'].value_counts()
            domain_entropy = self.calculate_entropy(email_domains)
            suspicious_emails = pan_data['Customer Email'].apply(self.is_suspicious_email).sum()
            
            # Calculate fraud score
            fraud_score = self.calculate_fraud_score(
                unique_merchants, unique_emails, unique_domains, 
                total_transactions, time_span_hours, domain_entropy, suspicious_emails
            )
            
            # Red flags
            red_flags = []
            if unique_merchants > 5:
                red_flags.append(f"{unique_merchants}Merchants")
            if unique_emails > unique_merchants:
                red_flags.append(f"{unique_emails}Emails")
            if domain_entropy > 2:
                red_flags.append(f"HighDomainEntropy")
            if total_transactions > 5 and time_span_hours < 24:
                red_flags.append(f"RapidTransactions")
            if suspicious_emails / total_transactions > 0.2:
                red_flags.append("SuspiciousEmails")
            
            # Risk level
            if fraud_score >= 8:
                risk_level = "CRITICAL"
            elif fraud_score >= 6:
                risk_level = "HIGH"
            elif fraud_score >= 4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Compile lists
            merchants_list = '; '.join(pan_data['MerchantName'].dropna().unique())
            emails_list = '; '.join(pan_data['Customer Email'].dropna().unique())
            mobile_list = '; '.join(pan_data['Customer Mobile'].dropna().astype(str).unique())
            transaction_ids_list = '; '.join(pan_data['TransactionID'].dropna().astype(str).unique())
            
            pan_groups.append({
                'PAN': pan,
                'Risk_Level': risk_level,
                'Fraud_Score': fraud_score,
                'Total_Transactions': total_transactions,
                'Unique_Merchants': unique_merchants,
                'Unique_Emails': unique_emails,
                'Unique_Mobile_Numbers': unique_mobile,
                'Unique_Cities': unique_cities,
                'Unique_MCCs': unique_mccs,
                'Unique_Banks': unique_banks,
                'Total_Amount': total_amount,
                'Average_Amount': avg_amount,
                'Successful_Transactions': successful_txns,
                'Pending_Transactions': pending_txns,
                'Failed_Transactions': failed_txns,
                'Success_Rate': (successful_txns / total_transactions * 100) if total_transactions > 0 else 0,
                'First_Transaction_Date': first_transaction,
                'Last_Transaction_Date': last_transaction,
                'Time_Span_Hours': time_span_hours,
                'Domain_Entropy': domain_entropy,
                'Suspicious_Emails': suspicious_emails,
                'Suspicious_Email_Ratio': (suspicious_emails / total_transactions * 100) if total_transactions > 0 else 0,
                'All_Merchants_Used': merchants_list,
                'All_Emails_Used': emails_list,
                'All_Mobile_Numbers_Used': mobile_list,
                'All_Transaction_IDs': transaction_ids_list,
                'Red_Flags': '; '.join(red_flags) if red_flags else 'None'
            })
        
        self.pan_grouped_df = pd.DataFrame(pan_groups)
        return self.pan_grouped_df
    
    def ml_anomaly_detection(self):
        """Use ML algorithms to detect anomalous patterns"""
        if not hasattr(self, 'pan_grouped_df') or self.pan_grouped_df is None:
            self.create_pan_grouped_analysis()
        
        # Prepare features for ML
        features = ['Unique_Merchants', 'Unique_Emails', 'Unique_Cities', 
                   'Total_Transactions', 'Time_Span_Hours', 'Domain_Entropy', 
                   'Suspicious_Emails', 'Total_Amount']
        
        # Fill NaN values and scale features
        X = self.pan_grouped_df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X_scaled)
        
        # Add results to analysis
        self.pan_grouped_df['ML_Anomaly'] = anomalies == -1
        
        return self.pan_grouped_df
    
    def export_to_excel(self):
        """Export analysis to Excel and return as bytes"""
        # Ensure we have analysis
        if not hasattr(self, 'pan_grouped_df') or self.pan_grouped_df is None:
            self.create_pan_grouped_analysis()
            self.ml_anomaly_detection()
        
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sort by fraud score
            pan_summary = self.pan_grouped_df.sort_values('Fraud_Score', ascending=False)
            
            # Different sheets
            pan_summary.to_excel(writer, sheet_name='PAN_Summary', index=False)
            
            high_risk = pan_summary[pan_summary['Fraud_Score'] >= 6]
            high_risk.to_excel(writer, sheet_name='High_Risk_PANs', index=False)
            
            ml_anomalies = pan_summary[pan_summary['ML_Anomaly'] == True]
            ml_anomalies.to_excel(writer, sheet_name='ML_Anomalies', index=False)
            
            multi_merchant = pan_summary[pan_summary['Unique_Merchants'] >= 3]
            multi_merchant.to_excel(writer, sheet_name='Multi_Merchant_PANs', index=False)
            
            # Statistics
            stats_data = [
                ['Total PANs Analyzed', len(pan_summary)],
                ['High Risk PANs', len(high_risk)],
                ['ML Anomalies', len(ml_anomalies)],
                ['Multi-Merchant PANs', len(multi_merchant)],
                ['Average Fraud Score', pan_summary['Fraud_Score'].mean()],
                ['Total Transaction Amount', pan_summary['Total_Amount'].sum()]
            ]
            stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        output.seek(0)
        return output

# Initialize the dashboard with auto Excel generation
@st.cache_data
def load_fraud_detector_with_excel(uploaded_file):
    detector = FraudDetectorDashboard()
    if detector.load_data(uploaded_file):
        # Automatically run all analysis and generate Excel
        detector.create_pan_grouped_analysis()
        detector.ml_anomaly_detection()
        excel_buffer = detector.export_to_excel()
        return detector, excel_buffer
    return None, None

def main():
    st.title("🚨 Fraud Detection Dashboard")
    st.markdown("Upload your transaction data to detect potential fraudulent patterns")
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your transaction data CSV file"
    )
    
    if uploaded_file is not None:
        # Auto-generate everything including Excel report
        with st.spinner("🔄 Analyzing data and generating Excel report..."):
            detector, excel_buffer = load_fraud_detector_with_excel(uploaded_file)
        
        if detector is not None and excel_buffer is not None:
            # Success message with download
            st.success("✅ Analysis completed! Excel report ready for download.")
            
            # Immediate download button at the top
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_buffer,
                    file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
            
            st.divider()
            
            # Display dashboard results
            display_dashboard(detector)
        else:
            st.error("❌ Failed to process the uploaded file. Please check your data format.")
    else:
        st.info("👆 Please upload a CSV file to begin fraud analysis")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        sample_data = {
            'PAN': ['1234****5678', '9876****4321'],
            'TransactionID': ['TXN001', 'TXN002'],
            'DateTime': ['01/01/2024 10:30', '01/01/2024 11:45'],
            'Amount': [100.50, 250.75],
            'MerchantName': ['Shop A', 'Store B'],
            'Customer Email': ['user@email.com', 'customer@domain.com'],
            'Customer Mobile': ['1234567890', '9876543210'],
            'Status': ['Success', 'Success'],
            'MerchantCity': ['City A', 'City B'],
            'MCC': ['5411', '5812'],
            'BankName': ['Bank A', 'Bank B']
        }
        st.dataframe(pd.DataFrame(sample_data))

def display_dashboard(detector):
    """Display the main dashboard with KPIs and visualizations"""
    
    # Key Metrics Row
    st.header("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_pans = len(detector.pan_grouped_df)
    high_risk_pans = len(detector.pan_grouped_df[detector.pan_grouped_df['Fraud_Score'] >= 6])
    ml_anomalies = len(detector.pan_grouped_df[detector.pan_grouped_df['ML_Anomaly'] == True])
    avg_fraud_score = detector.pan_grouped_df['Fraud_Score'].mean()
    total_amount = detector.pan_grouped_df['Total_Amount'].sum()
    
    with col1:
        st.metric(
            label="Total PANs",
            value=f"{total_pans:,}",
            help="Total number of unique PANs analyzed"
        )
    
    with col2:
        st.metric(
            label="High Risk PANs",
            value=f"{high_risk_pans:,}",
            delta=f"{(high_risk_pans/total_pans*100):.1f}%",
            delta_color="inverse",
            help="PANs with fraud score >= 6"
        )
    
    with col3:
        st.metric(
            label="ML Anomalies",
            value=f"{ml_anomalies:,}",
            delta=f"{(ml_anomalies/total_pans*100):.1f}%",
            delta_color="inverse",
            help="PANs detected as anomalous by ML algorithms"
        )
    
    with col4:
        st.metric(
            label="Avg Fraud Score",
            value=f"{avg_fraud_score:.2f}",
            help="Average fraud score across all PANs (0-10 scale)"
        )
    
    with col5:
        st.metric(
            label="Total Amount",
            value=f"${total_amount:,.2f}",
            help="Total transaction amount across all PANs"
        )
    
    # Charts Row
    st.header("📈 Fraud Analysis Charts")
    
    # Row 1: Simple Risk Overview
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Level Bar Chart
        risk_counts = detector.pan_grouped_df['Risk_Level'].value_counts()
        
        fig_risk = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="🚨 PANs by Risk Level",
            color=risk_counts.index,
            color_discrete_map={
                'CRITICAL': '#ff4757',
                'HIGH': '#ffa502',
                'MEDIUM': '#f1c40f',
                'LOW': '#2ed573'
            }
        )
        fig_risk.update_layout(
            xaxis_title="Risk Level",
            yaxis_title="Number of PANs",
            showlegend=False
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Simple Fraud Score Ranges
        detector.pan_grouped_df['Score_Range'] = pd.cut(
            detector.pan_grouped_df['Fraud_Score'], 
            bins=[0, 2, 4, 6, 8, 10], 
            labels=['0-2', '3-4', '5-6', '7-8', '9-10']
        )
        score_counts = detector.pan_grouped_df['Score_Range'].value_counts().sort_index()
        
        fig_score = px.bar(
            x=score_counts.index,
            y=score_counts.values,
            title="📊 PANs by Fraud Score Range",
            color_discrete_sequence=['#3742fa']
        )
        fig_score.update_layout(
            xaxis_title="Fraud Score Range",
            yaxis_title="Number of PANs"
        )
        st.plotly_chart(fig_score, use_container_width=True)
    
    # Row 2: Merchant and Email Analysis  
    col1, col2 = st.columns(2)
    
    with col1:
        # Multi-Merchant PANs
        merchant_groups = detector.pan_grouped_df['Unique_Merchants'].value_counts().sort_index()
        
        fig_merchants = px.bar(
            x=[f"{idx} Merchants" for idx in merchant_groups.index],
            y=merchant_groups.values,
            title="🏪 PANs by Number of Merchants Used",
            color_discrete_sequence=['#2ed573']
        )
        fig_merchants.update_layout(
            xaxis_title="Number of Merchants",
            yaxis_title="Number of PANs"
        )
        st.plotly_chart(fig_merchants, use_container_width=True)
    
    with col2:
        # Suspicious Email Analysis
        detector.pan_grouped_df['Email_Category'] = detector.pan_grouped_df['Suspicious_Email_Ratio'].apply(
            lambda x: 'High (>50%)' if x > 50 else 'Medium (20-50%)' if x > 20 else 'Low (<20%)'
        )
        email_counts = detector.pan_grouped_df['Email_Category'].value_counts()
        
        fig_email = px.bar(
            x=email_counts.index,
            y=email_counts.values,
            title="📧 PANs by Suspicious Email Level",
            color=email_counts.index,
            color_discrete_map={
                'High (>50%)': '#ff4757',
                'Medium (20-50%)': '#ffa502',
                'Low (<20%)': '#2ed573'
            }
        )
        fig_email.update_layout(
            xaxis_title="Suspicious Email Level",
            yaxis_title="Number of PANs",
            showlegend=False
        )
        st.plotly_chart(fig_email, use_container_width=True)
    
    # Row 3: Transaction Patterns
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction Volume Groups
        detector.pan_grouped_df['Volume_Category'] = pd.cut(
            detector.pan_grouped_df['Total_Transactions'],
            bins=[0, 1, 5, 10, 20, 1000],
            labels=['1', '2-5', '6-10', '11-20', '20+']
        )
        volume_counts = detector.pan_grouped_df['Volume_Category'].value_counts().sort_index()
        
        fig_volume = px.bar(
            x=[f"{idx} Transactions" for idx in volume_counts.index],
            y=volume_counts.values,
            title="💳 PANs by Transaction Volume",
            color_discrete_sequence=['#5f27cd']
        )
        fig_volume.update_layout(
            xaxis_title="Number of Transactions",
            yaxis_title="Number of PANs"
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        # Success Rate by Risk Level
        success_by_risk = detector.pan_grouped_df.groupby('Risk_Level')['Success_Rate'].mean().reset_index()
        success_by_risk = success_by_risk.sort_values('Success_Rate', ascending=True)
        
        fig_success = px.bar(
            success_by_risk,
            x='Risk_Level',
            y='Success_Rate',
            title="✅ Average Success Rate by Risk Level",
            color='Risk_Level',
            color_discrete_map={
                'CRITICAL': '#ff4757',
                'HIGH': '#ffa502',
                'MEDIUM': '#f1c40f',
                'LOW': '#2ed573'
            }
        )
        fig_success.update_layout(
            xaxis_title="Risk Level",
            yaxis_title="Success Rate (%)",
            showlegend=False
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    # Row 4: Time Analysis (Simplified)
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction Hours Analysis
        if len(detector.df) > 0:
            hour_counts = detector.df['Hour'].value_counts().sort_index()
            
            fig_hours = px.bar(
                x=hour_counts.index,
                y=hour_counts.values,
                title="⏰ Transactions by Hour of Day",
                color_discrete_sequence=['#00d2d3']
            )
            fig_hours.update_layout(
                xaxis_title="Hour of Day (0-23)",
                yaxis_title="Number of Transactions"
            )
            st.plotly_chart(fig_hours, use_container_width=True)
    
    with col2:
        # Rapid Transaction PANs
        rapid_pans = len(detector.pan_grouped_df[
            (detector.pan_grouped_df['Total_Transactions'] >= 5) & 
            (detector.pan_grouped_df['Time_Span_Hours'] <= 24)
        ])
        normal_pans = len(detector.pan_grouped_df) - rapid_pans
        
        fig_rapid = px.bar(
            x=['Normal PANs', 'Rapid Transaction PANs'],
            y=[normal_pans, rapid_pans],
            title="⚡ Rapid vs Normal Transaction Patterns",
            color=['Normal PANs', 'Rapid Transaction PANs'],
            color_discrete_map={
                'Normal PANs': '#2ed573',
                'Rapid Transaction PANs': '#ff4757'
            }
        )
        fig_rapid.update_layout(
            xaxis_title="Transaction Pattern",
            yaxis_title="Number of PANs",
            showlegend=False
        )
        st.plotly_chart(fig_rapid, use_container_width=True)
    
    # High Risk PANs Table
    st.header("🚨 High Risk PANs (Fraud Score >= 6)")
    
    high_risk_pans = detector.pan_grouped_df[
        detector.pan_grouped_df['Fraud_Score'] >= 6
    ].sort_values('Fraud_Score', ascending=False)
    
    if len(high_risk_pans) > 0:
        # Display key columns
        display_columns = [
            'PAN', 'Risk_Level', 'Fraud_Score', 'Total_Transactions',
            'Unique_Merchants', 'Unique_Emails', 'Total_Amount', 'Red_Flags'
        ]
        
        st.dataframe(
            high_risk_pans[display_columns],
            use_container_width=True,
            height=400
        )
        
        # Expandable section for detailed view
        with st.expander("View Detailed Information"):
            selected_pan = st.selectbox(
                "Select PAN for detailed view:",
                high_risk_pans['PAN'].tolist()
            )
            
            if selected_pan:
                pan_details = high_risk_pans[high_risk_pans['PAN'] == selected_pan].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Transaction Details:**")
                    st.write(f"- Total Transactions: {pan_details['Total_Transactions']}")
                    st.write(f"- Total Amount: ${pan_details['Total_Amount']:,.2f}")
                    st.write(f"- Success Rate: {pan_details['Success_Rate']:.1f}%")
                    st.write(f"- Time Span: {pan_details['Time_Span_Hours']:.1f} hours")
                
                with col2:
                    st.write("**Diversity Metrics:**")
                    st.write(f"- Unique Merchants: {pan_details['Unique_Merchants']}")
                    st.write(f"- Unique Emails: {pan_details['Unique_Emails']}")
                    st.write(f"- Unique Cities: {pan_details['Unique_Cities']}")
                    st.write(f"- Suspicious Emails: {pan_details['Suspicious_Emails']}")
                
                st.write("**Merchants Used:**")
                st.write(pan_details['All_Merchants_Used'])
                
                st.write("**Emails Used:**")
                st.write(pan_details['All_Emails_Used'])
    else:
        st.success("✅ No high-risk PANs detected!")
    
    # Download Section - Simplified since Excel is auto-generated
    st.header("📥 Additional Downloads")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Option to regenerate/re-download Excel
        if st.button("🔄 Regenerate Excel Report"):
            with st.spinner("Regenerating Excel report..."):
                new_excel_buffer = detector.export_to_excel()
                st.download_button(
                    label="📊 Download Fresh Excel Report",
                    data=new_excel_buffer,
                    file_name=f"fraud_analysis_fresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with col2:
        st.info("""
        **Excel Report Contains:**
        - PAN Summary with all metrics
        - High Risk PANs (Score >= 6)
        - ML Detected Anomalies
        - Multi-Merchant PANs
        - Detailed Statistics
        
        *Excel report was automatically generated when you uploaded your data!*
        """)
    
    st.divider()
    
    # ML Anomalies Section
    st.header("🤖 Machine Learning Detected Anomalies")
    
    ml_anomalies = detector.pan_grouped_df[detector.pan_grouped_df['ML_Anomaly'] == True]
    
    if len(ml_anomalies) > 0:
        st.warning(f"⚠️ {len(ml_anomalies)} PANs detected as anomalous by ML algorithms")
        
        display_columns = [
            'PAN', 'Risk_Level', 'Fraud_Score', 'Total_Transactions',
            'Unique_Merchants', 'Unique_Emails', 'Total_Amount'
        ]
        
        st.dataframe(
            ml_anomalies[display_columns].sort_values('Fraud_Score', ascending=False),
            use_container_width=True
        )
    else:
        st.success("✅ No ML anomalies detected!")

if __name__ == "__main__":
    main()
