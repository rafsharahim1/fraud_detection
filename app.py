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

# Custom CSS for better styling + Keep Alive functionality
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

<script>
// Keep the app alive by preventing sleep
function keepAlive() {
    // Send a small ping every 5 minutes to keep session active
    fetch(window.location.href, {
        method: 'HEAD',
        cache: 'no-cache'
    }).catch(console.log);
}

// Start keep alive functionality
setInterval(keepAlive, 300000); // 5 minutes

// Prevent page from sleeping on mobile/tablet
function preventSleep() {
    document.addEventListener('visibilitychange', function() {
        if (document.visibilityState === 'visible') {
            keepAlive();
        }
    });
    
    // Keep screen awake if supported
    if ('wakeLock' in navigator) {
        let wakeLock = null;
        async function requestWakeLock() {
            try {
                wakeLock = await navigator.wakeLock.request('screen');
            } catch (err) {
                console.log('Wake Lock not supported');
            }
        }
        requestWakeLock();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', preventSleep);

// Auto-refresh functionality for long-running sessions
let sessionTime = 0;
setInterval(function() {
    sessionTime += 60000; // 1 minute
    // Auto-refresh after 4 hours of inactivity
    if (sessionTime >= 14400000) {
        console.log('Refreshing to maintain session');
        location.reload();
    }
}, 60000);

// Reset session timer on user activity
document.addEventListener('click', () => sessionTime = 0);
document.addEventListener('keypress', () => sessionTime = 0);
document.addEventListener('scroll', () => sessionTime = 0);

console.log('🚀 Fraud Detection Dashboard - Keep Alive Enabled');
</script>
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
        # Debug: Check original DateTime values first
        print(f"Original DateTime sample values:")
        print(f"First 3 raw DateTime values: {self.df['DateTime'].head(3).tolist()}")
        print(f"DateTime column type: {type(self.df['DateTime'].iloc[0]) if len(self.df) > 0 else 'Empty'}")
        
        # Clean the DateTime column first - remove any extra spaces or characters
        self.df['DateTime'] = self.df['DateTime'].astype(str).str.strip()
        
        # Try multiple datetime parsing approaches based on detected format
        datetime_converted = False
        
        # Approach 1: Direct format matching YYYY-MM-DD HH:MM:SS (ISO format)
        try:
            self.df['DateTime'] = pd.to_datetime(self.df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            successful_conversions = self.df['DateTime'].notna().sum()
            print(f"Approach 1 (YYYY-MM-DD HH:MM:SS): {successful_conversions} successful conversions")
            if successful_conversions > 0:
                datetime_converted = True
        except Exception as e:
            print(f"Approach 1 failed: {e}")
        
        # Approach 2: Try ISO format parsing if approach 1 didn't work
        if not datetime_converted or self.df['DateTime'].isna().sum() > len(self.df) * 0.1:
            try:
                self.df['DateTime'] = pd.to_datetime(self.df['DateTime'], errors='coerce')
                successful_conversions = self.df['DateTime'].notna().sum()
                print(f"Approach 2 (ISO auto-detect): {successful_conversions} successful conversions")
                if successful_conversions > 0:
                    datetime_converted = True
            except Exception as e:
                print(f"Approach 2 failed: {e}")
        
        # Approach 3: Try automatic inference as last resort
        if not datetime_converted or self.df['DateTime'].isna().sum() > len(self.df) * 0.1:
            try:
                self.df['DateTime'] = pd.to_datetime(self.df['DateTime'], infer_datetime_format=True, errors='coerce')
                successful_conversions = self.df['DateTime'].notna().sum()
                print(f"Approach 3 (infer_datetime_format): {successful_conversions} successful conversions")
            except Exception as e:
                print(f"Approach 3 failed: {e}")
        
        # Final check and fallback
        null_count = self.df['DateTime'].isnull().sum()
        total_count = len(self.df)
        print(f"Final result: {total_count - null_count}/{total_count} dates successfully parsed")
        
        if null_count == total_count:
            print("WARNING: All datetime parsing failed. Using current timestamp as fallback.")
            self.df['DateTime'] = pd.Timestamp.now()
        elif null_count > 0:
            print(f"WARNING: {null_count} dates failed to parse and will be treated as NaT")
        
        # Extract hour and date for time-based analysis
        self.df['Hour'] = self.df['DateTime'].dt.hour
        self.df['Date'] = self.df['DateTime'].dt.date
        
        # Clean email addresses and extract domains
        self.df['Email_Domain'] = self.df['Customer Email'].apply(self.extract_email_domain)
        
        # Convert amounts to numeric
        self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        if 'PURCHASE_AMOUNT' in self.df.columns:
            self.df['PURCHASE_AMOUNT'] = pd.to_numeric(self.df['PURCHASE_AMOUNT'], errors='coerce')
        
        # Show final sample of converted datetimes
        print(f"Sample of successfully converted DateTime values:")
        valid_datetimes = self.df[self.df['DateTime'].notna()]['DateTime'].head(3)
        print(f"First 3 valid DateTime values: {valid_datetimes.tolist()}")
        
        # Show extracted hour and date samples
        print(f"Sample extracted hours: {self.df['Hour'].head(3).tolist()}")
        print(f"Sample extracted dates: {self.df['Date'].head(3).tolist()}")
    
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
                            total_transactions, time_span_hours, domain_entropy, suspicious_emails,
                            unique_channels, unique_ecis, unique_issuers):
        """Calculate enhanced composite fraud score including Channel, ECI, and Issuer diversity"""
        score = 0
        
        # Original scoring logic
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
        
        # Enhanced scoring with Channel, ECI, and Issuer diversity
        if unique_channels > 2:
            score += 2
        
        if unique_ecis > 3:
            score += 2
        
        if unique_issuers > 2:
            score += 1
        
        # Additional risk if high diversity across multiple dimensions
        diversity_score = unique_merchants + unique_channels + unique_ecis + unique_issuers
        if diversity_score > 10 and total_transactions < 30:
            score += 2
        
        return min(score, 10)
    
    def create_pan_grouped_analysis(self):
        """Create detailed analysis grouped by PAN with enhanced Channel, ECI, and Issuer metrics"""
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
            
            # Enhanced metrics - Channel, ECI, and Issuer diversity
            unique_channels = pan_data['Channel'].nunique() if 'Channel' in pan_data.columns else 0
            unique_ecis = pan_data['ECI Value'].nunique() if 'ECI Value' in pan_data.columns else 0
            unique_issuers = pan_data['Issuer'].nunique() if 'Issuer' in pan_data.columns else 0
            
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
            
            # Calculate enhanced fraud score
            fraud_score = self.calculate_fraud_score(
                unique_merchants, unique_emails, unique_domains, 
                total_transactions, time_span_hours, domain_entropy, suspicious_emails,
                unique_channels, unique_ecis, unique_issuers
            )
            
            # Enhanced red flags
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
            if unique_channels > 2:
                red_flags.append(f"{unique_channels}Channels")
            if unique_ecis > 3:
                red_flags.append(f"{unique_ecis}ECIs")
            if unique_issuers > 2:
                red_flags.append(f"{unique_issuers}Issuers")
            
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
            
            # Enhanced lists for Channel, ECI, and Issuer
            channels_list = '; '.join(pan_data['Channel'].dropna().astype(str).unique()) if 'Channel' in pan_data.columns else 'N/A'
            ecis_list = '; '.join(pan_data['ECI Value'].dropna().astype(str).unique()) if 'ECI Value' in pan_data.columns else 'N/A'
            issuers_list = '; '.join(pan_data['Issuer'].dropna().astype(str).unique()) if 'Issuer' in pan_data.columns else 'N/A'
            
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
                'Unique_Channels': unique_channels,
                'Unique_ECIs': unique_ecis,
                'Unique_Issuers': unique_issuers,
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
                'All_Channels_Used': channels_list,
                'All_ECIs_Used': ecis_list,
                'All_Issuers_Used': issuers_list,
                'All_Transaction_IDs': transaction_ids_list,
                'Red_Flags': '; '.join(red_flags) if red_flags else 'None'
            })
        
        self.pan_grouped_df = pd.DataFrame(pan_groups)
        return self.pan_grouped_df
    
    def ml_anomaly_detection(self):
        """Use ML algorithms to detect anomalous patterns with enhanced features"""
        if not hasattr(self, 'pan_grouped_df') or self.pan_grouped_df is None:
            self.create_pan_grouped_analysis()
        
        # Enhanced features for ML including Channel, ECI, and Issuer diversity
        features = ['Unique_Merchants', 'Unique_Emails', 'Unique_Cities', 
                   'Total_Transactions', 'Time_Span_Hours', 'Domain_Entropy', 
                   'Suspicious_Emails', 'Total_Amount', 'Unique_Channels', 
                   'Unique_ECIs', 'Unique_Issuers']
        
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
        """Export enhanced analysis to Excel and return as bytes"""
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
            
            # Enhanced analysis sheets
            multi_channel = pan_summary[pan_summary['Unique_Channels'] >= 2]
            multi_channel.to_excel(writer, sheet_name='Multi_Channel_PANs', index=False)
            
            multi_eci = pan_summary[pan_summary['Unique_ECIs'] >= 3]
            multi_eci.to_excel(writer, sheet_name='Multi_ECI_PANs', index=False)
            
            multi_issuer = pan_summary[pan_summary['Unique_Issuers'] >= 2]
            multi_issuer.to_excel(writer, sheet_name='Multi_Issuer_PANs', index=False)
            
            # Enhanced statistics
            stats_data = [
                ['Total PANs Analyzed', len(pan_summary)],
                ['High Risk PANs', len(high_risk)],
                ['ML Anomalies', len(ml_anomalies)],
                ['Multi-Merchant PANs', len(multi_merchant)],
                ['Multi-Channel PANs', len(multi_channel)],
                ['Multi-ECI PANs', len(multi_eci)],
                ['Multi-Issuer PANs', len(multi_issuer)],
                ['Average Fraud Score', pan_summary['Fraud_Score'].mean()],
                ['Total Transaction Amount', pan_summary['Total_Amount'].sum()],
                ['Average Unique Channels per PAN', pan_summary['Unique_Channels'].mean()],
                ['Average Unique ECIs per PAN', pan_summary['Unique_ECIs'].mean()],
                ['Average Unique Issuers per PAN', pan_summary['Unique_Issuers'].mean()]
            ]
            stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Enhanced_Statistics', index=False)
        
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
    st.title("Fraud Detection Dashboard")
    st.markdown("Upload your transaction data to detect potential fraudulent patterns with Channel, ECI, and Issuer analysis")
    
    # Add keep-alive status indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**Status:** 🟢 Active & Protected from Sleep")
    with col2:
        if st.button("🔄 Refresh Session"):
            st.rerun()
    with col3:
        st.markdown(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    st.divider()
    
    # Sidebar for file upload
    st.sidebar.header("📁 Data Upload")
    st.sidebar.success("🛡️ App Sleep Protection: ON")
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
            st.success("✅ Analysis completed! Enhanced Excel report ready for download.")
            
            # Immediate download button at the top
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📊 Download Enhanced Excel Report",
                    data=excel_buffer,
                    file_name=f"enhanced_fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
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
        st.subheader("📋 Expected Data Format (Enhanced)")
        sample_data = {
            'PAN': ['1234****5678', '9876****4321'],
            'TransactionID': ['TXN001', 'TXN002'],
            'DateTime': ['2024-01-01 10:30:00', '2024-01-01 11:45:00'],
            'Amount': [100.50, 250.75],
            'MerchantName': ['Shop A', 'Store B'],
            'Customer Email': ['user@email.com', 'customer@domain.com'],
            'Customer Mobile': ['1234567890', '9876543210'],
            'Status': ['Success', 'Success'],
            'MerchantCity': ['City A', 'City B'],
            'MCC': ['5411', '5812'],
            'BankName': ['Bank A', 'Bank B'],
            'Channel': ['Online', 'Mobile'],
            'ECI Value': ['05', '07'],
            'Issuer': ['Issuer A', 'Issuer B']
        }
        st.dataframe(pd.DataFrame(sample_data))

def display_dashboard(detector):
    """Display the enhanced dashboard with KPIs and visualizations including Channel, ECI, and Issuer analysis"""
    
    # Enhanced Key Metrics Row
    st.header("📊 Enhanced Key Performance Indicators")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_pans = len(detector.pan_grouped_df)
    high_risk_pans = len(detector.pan_grouped_df[detector.pan_grouped_df['Fraud_Score'] >= 6])
    ml_anomalies = len(detector.pan_grouped_df[detector.pan_grouped_df['ML_Anomaly'] == True])
    avg_fraud_score = detector.pan_grouped_df['Fraud_Score'].mean()
    total_amount = detector.pan_grouped_df['Total_Amount'].sum()
    
    # Enhanced metrics
    multi_channel_pans = len(detector.pan_grouped_df[detector.pan_grouped_df['Unique_Channels'] >= 2])
    
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
            label="Multi-Channel",
            value=f"{multi_channel_pans:,}",
            delta=f"{(multi_channel_pans/total_pans*100):.1f}%",
            delta_color="inverse",
            help="PANs using multiple channels"
        )
    
    with col5:
        st.metric(
            label="Avg Fraud Score",
            value=f"{avg_fraud_score:.2f}",
            help="Average fraud score across all PANs (0-10 scale)"
        )
    
    with col6:
        st.metric(
            label="Total Amount",
            value=f"${total_amount:,.2f}",
            help="Total transaction amount across all PANs"
        )
    
    # Enhanced Charts Row
    st.header("📈 Enhanced Fraud Analysis Charts")
    
    # Row 1: Risk Overview + Channel Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Level Bar Chart
        risk_counts = detector.pan_grouped_df['Risk_Level'].value_counts()
        
        fig_risk = px.bar(
            x=risk_counts.index.tolist(),
            y=risk_counts.values.tolist(),
            title="🚨 PANs by Risk Level",
            color=risk_counts.index.tolist(),
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
            x=score_counts.index.tolist(),
            y=score_counts.values.tolist(),
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
            x=[f"{idx} Merchants" for idx in merchant_groups.index.tolist()],
            y=merchant_groups.values.tolist(),
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
            x=email_counts.index.tolist(),
            y=email_counts.values.tolist(),
            title="📧 PANs by Suspicious Email Level",
            color=email_counts.index.tolist(),
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
            x=[f"{idx} Transactions" for idx in volume_counts.index.tolist()],
            y=volume_counts.values.tolist(),
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
                pd.DataFrame({
                    'Hour': hour_counts.index.tolist(),
                    'Transactions': hour_counts.values.tolist()
                }),
                x='Hour',
                y='Transactions',
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
