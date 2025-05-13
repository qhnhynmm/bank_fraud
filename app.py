import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime
import plotly.graph_objects as go
import time
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report

# Set page configuration - phải là lệnh Streamlit đầu tiên
st.set_page_config(
    page_title="Hệ Thống Phát Hiện Gian Lận",
    page_icon="🔍",
    layout="wide"
)

# Đọc config file
def load_config():
    try:
        with open('src/config/config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Lỗi khi đọc cấu hình: {str(e)}")
        st.info("Vui lòng kiểm tra file cấu hình tại src/config/config.yaml")
        st.stop()

# Khởi tạo inference engine
def get_inference_engine(config, model_type):
    from src.task.inference import ModelInference
    
    # Cập nhật config với model đã chọn
    config['model']['type'] = model_type
    
    try:
        return ModelInference(config)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy model {model_type}")
        st.info("Vui lòng đảm bảo bạn đã đào tạo mô hình:")
        st.code(f"python main.py --mode train --model {model_type}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        st.stop()

# Kiểm tra xem model có tồn tại không
def check_model_exists(model_type):
    if model_type == 'mlp':
        model_path = os.path.join('checkpoints', f'best_model_{model_type}.pt')
    else:
        model_path = os.path.join('checkpoints', f'best_model_{model_type}.joblib')
    
    return os.path.exists(model_path)

def main():
    # Tiêu đề
    st.title("🔍 Hệ Thống Phát Hiện Gian Lận Ngân Hàng")
    st.markdown("---")
    
    # Thông tin thời gian hiện tại
    st.info(f"Thời gian hiện tại: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Đọc config
    config = load_config()
    
    # Sidebar cho cấu hình
    st.sidebar.header("Cấu Hình Hệ Thống")
    
    # Danh sách models
    model_types = ["mlp", "random_forest", "xgboost", "lightgbm"]
    model_names = ["MLP (Neural Network)", "Random Forest", "XGBoost", "LightGBM"]
    model_map = {m_type: m_name for m_type, m_name in zip(model_types, model_names)}
    
    # Hiển thị các model sẵn có
    available_models = []
    for m_type in model_types:
        if check_model_exists(m_type):
            available_models.append(f"{model_map[m_type]} ✓")
        else:
            available_models.append(f"{model_map[m_type]}")
    
    # Chọn model
    selected_idx = st.sidebar.selectbox(
        "Chọn Mô Hình",
        range(len(model_types)),
        format_func=lambda i: available_models[i]
    )
    model_type = model_types[selected_idx]
    
    # Ngưỡng phát hiện
    threshold = st.sidebar.slider(
        "Ngưỡng Phát Hiện Gian Lận",
        min_value=0.0,
        max_value=1.0, 
        value=float(config['inference']['threshold']),
        step=0.05
    )
    
    # Cập nhật threshold trong config
    config['inference']['threshold'] = threshold
    
    # Tabs chính
    tab1, tab2 = st.tabs(["Phân Tích Dữ Liệu", "Kiểm Tra Giao Dịch"])
    
    # Tab 1: Phân tích dữ liệu
    with tab1:
        st.header("Phân Tích Dữ Liệu Giao Dịch")
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Tải lên file CSV chứa dữ liệu giao dịch",
            type=['csv']
        )
        
        # Nút load dữ liệu mẫu
        col1, col2 = st.columns([3, 1])
        with col2:
            sample_button = st.button("Dùng Dữ Liệu Mẫu", use_container_width=True)
            
            if sample_button:
                sample_path = config['inference'].get('test_file', 'data/test.csv')
                if os.path.exists(sample_path):
                    uploaded_file = open(sample_path, 'rb')
                    st.success("✅ Đã tải dữ liệu mẫu!")
                else:
                    st.error("❌ Không tìm thấy dữ liệu mẫu.")
        
        if uploaded_file is not None:
            try:
                # Đọc dữ liệu
                data = pd.read_csv(uploaded_file)
                
                # Hiển thị thông tin cơ bản
                st.subheader("Thông Tin Dữ Liệu")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Số Giao Dịch", f"{len(data):,}")
                with col2:
                    st.metric("Số Thuộc Tính", data.shape[1])
                with col3:
                    st.metric("Giá Trị Bị Thiếu", f"{data.isnull().sum().sum():,}")
                
                # Hiển thị dữ liệu
                st.subheader("Xem Trước Dữ Liệu")
                st.dataframe(data.head())
                
                # Phân tích
                if st.button("Phân Tích Gian Lận"):
                    with st.spinner("Đang xử lý dữ liệu..."):
                        # Hiển thị tiến trình
                        progress = st.progress(0)
                        status = st.empty()
                        
                        status.text("Đang tiền xử lý dữ liệu...")
                        progress.progress(25)
                        time.sleep(0.5)
                        
                        # Khởi tạo inference engine
                        inference = get_inference_engine(config, model_type)
                        
                        status.text("Đang thực hiện dự đoán...")
                        progress.progress(50)
                        time.sleep(0.5)
                        
                        # Thực hiện dự đoán
                        try:
                            probs = inference.predict(data, return_proba=True)
                            predictions = (probs >= threshold).astype(int)
                            
                            # Thêm kết quả vào dataframe
                            results = data.copy()
                            results['Xác Suất Gian Lận'] = probs
                            results['Cảnh Báo Gian Lận'] = predictions
                            
                            progress.progress(75)
                            status.text("Đang phân tích kết quả...")
                            time.sleep(0.5)
                            
                            # Hoàn thành
                            progress.progress(100)
                            status.text("Hoàn thành!")
                            time.sleep(0.5)
                            
                            # Xóa thanh tiến trình
                            progress.empty()
                            status.empty()
                            
                            # Hiển thị kết quả
                            st.subheader("Kết Quả Phân Tích")
                            
                            # Thống kê
                            fraud_count = int(predictions.sum())
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Giao Dịch Gian Lận", f"{fraud_count:,}")
                                st.metric("Tỷ Lệ Gian Lận", f"{(fraud_count / len(predictions)) * 100:.2f}%")
                            
                            with col2:
                                st.metric("Xác Suất Trung Bình", f"{np.mean(probs):.4f}")
                                st.metric("Xác Suất Cao Nhất", f"{np.max(probs):.4f}")
                            
                            # Hiển thị ma trận điểm (Confusion Matrix)
                            st.subheader("Ma Trận Điểm (Confusion Matrix)")
                            
                            # Kiểm tra xem có cột Fraud trong dữ liệu ban đầu hay không
                            if 'Fraud' in data.columns:
                                # Tính confusion matrix
                                true_labels = data['Fraud'].values
                                cm = confusion_matrix(true_labels, predictions)
                                
                                # Tạo nhãn
                                categories = ['Bình thường', 'Gian lận']
                                
                                # Tạo annotation text
                                annotations = []
                                for i in range(len(categories)):
                                    for j in range(len(categories)):
                                        annotations.append({
                                            'x': categories[j],
                                            'y': categories[i],
                                            'text': str(cm[i, j]),
                                            'font': {'color': 'white' if cm[i, j] > cm.max()/2 else 'black'},
                                            'showarrow': False
                                        })
                                
                                # Tạo heatmap
                                fig_cm = ff.create_annotated_heatmap(
                                    z=cm,
                                    x=categories,
                                    y=categories,
                                    annotation_text=[[str(y) for y in x] for x in cm],
                                    colorscale='Blues'
                                )
                                
                                # Cập nhật layout
                                fig_cm.update_layout(
                                    title='Ma Trận Điểm',
                                    xaxis=dict(title='Dự đoán'),
                                    yaxis=dict(title='Thực tế'),
                                    height=400
                                )
                                
                                # Hiển thị
                                st.plotly_chart(fig_cm, use_container_width=True)
                                
                                # Hiển thị các chỉ số đánh giá chính
                                st.subheader("Các Chỉ Số Đánh Giá")
                                
                                # Tính TP, FP, TN, FN
                                tn, fp, fn, tp = cm.ravel()
                                
                                # Tính các chỉ số
                                accuracy = (tp + tn) / (tp + tn + fp + fn)
                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                                
                                # Hiển thị các chỉ số dưới dạng metrics
                                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                                
                                with metrics_col1:
                                    st.metric("Độ chính xác (Accuracy)", f"{accuracy:.4f}")
                                
                                with metrics_col2:
                                    st.metric("Độ nhạy (Recall)", f"{recall:.4f}")
                                
                                with metrics_col3:
                                    st.metric("Độ đặc hiệu (Precision)", f"{precision:.4f}")
                                
                                with metrics_col4:
                                    st.metric("Điểm F1 (F1-Score)", f"{f1:.4f}")
                                
                            else:
                                st.info("Không thể hiển thị ma trận điểm vì không có cột 'Fraud' trong dữ liệu ban đầu để so sánh với dự đoán.")
                                
                                # Tạo ma trận điểm mô phỏng
                                st.markdown("#### Ma Trận Điểm Mô Phỏng")
                                st.markdown("Dưới đây là mô phỏng ma trận điểm dựa trên ngưỡng phát hiện:")
                                
                                # Chia phân phối xác suất
                                low_risk = np.sum(probs < 0.3)
                                medium_risk = np.sum((probs >= 0.3) & (probs < threshold))
                                high_risk = np.sum(probs >= threshold)
                                
                                # Tạo ma trận mô phỏng
                                mock_cm = np.array([
                                    [low_risk, medium_risk],
                                    [0, high_risk]
                                ])
                                
                                # Tạo nhãn
                                categories = ['Dưới ngưỡng', 'Trên ngưỡng']
                                
                                # Tạo heatmap
                                fig_cm = ff.create_annotated_heatmap(
                                    z=mock_cm,
                                    x=categories,
                                    y=['Phân loại', 'Cảnh báo'],
                                    annotation_text=[[str(y) for y in x] for x in mock_cm],
                                    colorscale='Blues'
                                )
                                
                                # Cập nhật layout
                                fig_cm.update_layout(
                                    title=f'Phân Phối Điểm Theo Ngưỡng ({threshold})',
                                    height=300
                                )
                                
                                # Hiển thị
                                st.plotly_chart(fig_cm, use_container_width=True)
                            
                            # Hiển thị dữ liệu kết quả
                            st.subheader("Chi Tiết Giao Dịch")
                            
                            # Tùy chọn lọc
                            show_fraud_only = st.checkbox("Chỉ Hiển Thị Giao Dịch Gian Lận")
                            
                            # Lọc dữ liệu
                            if show_fraud_only:
                                display_data = results[results['Cảnh Báo Gian Lận'] == 1]
                            else:
                                display_data = results
                            
                            # Hiển thị bảng dữ liệu với độ rủi ro
                            st.dataframe(
                                display_data.style.background_gradient(
                                    subset=['Xác Suất Gian Lận'],
                                    cmap='RdYlGn_r'
                                ),
                                height=400
                            )
                            
                            # Tải về kết quả
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="Tải Xuống Kết Quả (CSV)",
                                data=csv,
                                file_name="ket_qua_phan_tich.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Lỗi khi phân tích: {str(e)}")
                            st.exception(e)
            
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {str(e)}")
                st.exception(e)
    
    # Tab 2: Kiểm tra giao dịch
    with tab2:
        st.header("Kiểm Tra Giao Dịch Đơn Lẻ")
        
        # Form nhập thông tin giao dịch
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Số Tiền Giao Dịch", min_value=0.0, value=1000.0)
            
            transaction_type = st.selectbox(
                "Loại Giao Dịch",
                options=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
            )
            
            day_of_week = st.selectbox(
                "Ngày Trong Tuần",
                options=["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
            )
        
        with col2:
            hour = st.slider("Giờ Giao Dịch", 0, 23, 12)
            
            merchant_category = st.selectbox(
                "Danh Mục Người Bán",
                options=["Bán Lẻ", "Điện Tử", "Du Lịch", "Giải Trí", "Thực Phẩm", "Khác"]
            )
            
            is_weekend = st.checkbox("Giao Dịch Cuối Tuần", value=False)
        
        # Nút phân tích
        if st.button("Phân Tích Giao Dịch"):
            with st.spinner("Đang phân tích..."):
                # Ánh xạ ngày trong tuần
                day_map = {
                    "Thứ Hai": 1, "Thứ Ba": 2, "Thứ Tư": 3,
                    "Thứ Năm": 4, "Thứ Sáu": 5, "Thứ Bảy": 6, "Chủ Nhật": 7
                }
                
                # Tạo đối tượng giao dịch
                transaction = {
                    'Transaction ID': f'#{int(time.time())}',
                    'Date': datetime.now().strftime('%d-%b-%y'),
                    'Amount': amount,
                    'Type of Transaction': transaction_type,
                    'Entry Mode': 'ONLINE',
                    'Day of Week': day_map.get(day_of_week, 1),
                    'Time': hour,
                    'Type of Card': 'Visa',
                    'Merchant Group': merchant_category,
                    'Country of Transaction': 'Vietnam',
                    'Country of Residence': 'Vietnam',
                    'Shipping Address': 'Vietnam',
                    'Gender': 'M',
                    'Age': 35,
                    'Bank': 'VCB',
                    'Is_Weekend': int(is_weekend)
                }
                
                try:
                    # Khởi tạo inference engine
                    inference = get_inference_engine(config, model_type)
                    
                    # Thực hiện dự đoán
                    probability = inference.predict_single(transaction, return_proba=True)
                    is_fraud = probability >= threshold
                    
                    # Hiển thị kết quả
                    st.subheader("Kết Quả Phân Tích")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Tạo biểu đồ gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=probability,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Xác Suất Gian Lận"},
                            gauge={
                                'axis': {'range': [0, 1]},
                                'bar': {'color': "red" if is_fraud else "green"},
                                'steps': [
                                    {'range': [0, 0.3], 'color': "lightgreen"},
                                    {'range': [0.3, 0.7], 'color': "orange"},
                                    {'range': [0.7, 1], 'color': "salmon"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': threshold
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Hiển thị quyết định
                        if is_fraud:
                            st.error("🚨 CẢNH BÁO GIAN LẬN")
                            st.warning("Khuyến nghị: Chặn giao dịch và kiểm tra")
                        else:
                            st.success("✅ GIAO DỊCH HỢP LỆ")
                            st.info("Khuyến nghị: Cho phép giao dịch")
                    
                    # Hiển thị các yếu tố rủi ro
                    st.subheader("Các Yếu Tố Rủi Ro")
                    
                    risk_factors = []
                    
                    # Xác định các yếu tố rủi ro
                    if amount > 5000:
                        risk_factors.append("Số tiền giao dịch lớn")
                    
                    if transaction_type in ["CASH_OUT", "TRANSFER"]:
                        risk_factors.append(f"Loại giao dịch có rủi ro cao ({transaction_type})")
                    
                    if is_weekend:
                        risk_factors.append("Giao dịch vào cuối tuần")
                    
                    if hour < 6 or hour > 22:
                        risk_factors.append(f"Thời gian giao dịch bất thường ({hour}:00)")
                    
                    # Hiển thị các yếu tố rủi ro
                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(factor)
                    else:
                        st.info("Không phát hiện yếu tố rủi ro đáng kể")
                
                except Exception as e:
                    st.error(f"Lỗi khi phân tích giao dịch: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()