import streamlit as st
import pandas as pd
import pickle
import os

# Đường dẫn tới các tệp mô hình
model_path1 = 'best_model.pkl'
model_path2 = 'best_model2.pkl'

# Kiểm tra và tải mô hình nếu tệp tồn tại
try:
    # Tải mô hình phân loại
    if os.path.exists(model_path1):
        with open(model_path1, 'rb') as f:
            best_model = pickle.load(f)
    else:
        st.error(f"Tệp mô hình '{model_path1}' không tồn tại hoặc không thể truy cập!")

    # Tải mô hình hồi quy
    if os.path.exists(model_path2):
        with open(model_path2, 'rb') as f:
            best_model2 = pickle.load(f)
    else:
        st.error(f"Tệp mô hình '{model_path2}' không tồn tại hoặc không thể truy cập!")

except Exception as e:
    st.error(f"Đã xảy ra lỗi khi tải mô hình: {str(e)}")

# Tiêu đề và mô tả
st.title("Dự báo Rủi ro Giao hàng Trễ và Doanh số Khách hàng")

# Nhập dữ liệu đầu vào cho cả hai dự đoán
st.header("Dự đoán Rủi ro Giao hàng Trễ")
days_for_shipment_scheduled = st.number_input("Days for shipment (scheduled)")
order_item_product_price = st.number_input("Order Item Product Price")
order_item_quantity = st.number_input("Order Item Quantity")

# Input data frame cho mô hình phân loại
input_data_classification = pd.DataFrame({
    'Days for shipment (scheduled)': [days_for_shipment_scheduled],
    'Order Item Product Price': [order_item_product_price],
    'Order Item Quantity': [order_item_quantity]
})

# Kiểm tra kiểu dữ liệu của các cột trong dữ liệu nhập
st.write("Kiểu dữ liệu của các cột trong dữ liệu nhập:", input_data_classification.dtypes)

# Kiểm tra xem dữ liệu nhập có chứa giá trị không hợp lệ hay không
if input_data_classification.select_dtypes(include=['number']).isnull().any().any():
    st.error("Dữ liệu nhập chứa giá trị thiếu (NaN). Vui lòng nhập đầy đủ các giá trị hợp lệ.")
else:
    # Dự đoán với mô hình phân loại
    if st.button("Dự đoán Rủi ro Giao hàng Trễ"):
        prediction_classification = best_model.predict(input_data_classification)[0]
        st.write("Dự đoán Rủi ro Giao hàng Trễ:", "Có" if prediction_classification == 1 else "Không")

# Nhập các thông tin khác để dự đoán doanh số khách hàng
st.header("Dự đoán Doanh số Khách hàng")
category_id = st.number_input("Category Id")
customer_city = st.text_input("Customer City")
customer_country = st.text_input("Customer Country")
customer_segment = st.text_input("Customer Segment")
customer_state = st.text_input("Customer State")
product_price = st.number_input("Product Price")
order_region = st.text_input("Order Region")
market = st.text_input("Market")

# Input data frame cho mô hình hồi quy
input_data_regression = pd.DataFrame({
    'Type': ['DEBIT'],
    'Days for shipment (scheduled)': [days_for_shipment_scheduled],
    'Delivery Status': ['Advance shipping'],
    'Category Id': [category_id],
    'Category Name': ['Cleats'],
    'Customer City': [customer_city],
    'Customer Country': [customer_country],
    'Customer Segment': [customer_segment],
    'Customer State': [customer_state],
    'Order City': ['Bikaner'],
    'Order Country': ['India'],
    'Order Item Product Price': [order_item_product_price],
    'Order Item Quantity': [order_item_quantity],
    'Order Status': ['COMPLETE'],
    'Product Card Id': [1360],
    'Product Price': [product_price],
    'Order Region': [order_region],
    'Market': [market]
})

# Kiểm tra kiểu dữ liệu của các cột trong dữ liệu hồi quy
st.write("Kiểu dữ liệu của các cột trong dữ liệu hồi quy:", input_data_regression.dtypes)

# Dự đoán với mô hình hồi quy
if st.button("Dự đoán Doanh số Khách hàng"):
    prediction_regression = best_model2.predict(input_data_regression.select_dtypes(include=['number']))[0]
    st.write("Dự đoán Doanh số Khách hàng:", prediction_regression)
