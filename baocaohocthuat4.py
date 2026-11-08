
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(
    page_title="Dự đoán chi phí bảo hiểm y tế",
    page_icon="📊",
    layout="wide"
)

st.title("ỨNG DỤNG DỰ ĐOÁN CHI PHÍ BẢO HIỂM Y TẾ")
st.subheader("Kết hợp Linear Regression và K-means")
st.write("Nhập thông tin cá nhân để dự đoán chi phí bảo hiểm và phân nhóm dữ liệu.")

# 2. TẢI DỮ LIỆU HUẤN LUYỆN
st.sidebar.header("Dữ liệu huấn luyện")

uploaded_file = st.sidebar.file_uploader(
    "Tải file dữ liệu Train CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Vui lòng tải lên file Dự đoán chi phí bảo hiểm y tế_train.csv")
    st.stop()

# Đọc dữ liệu
df = pd.read_csv(uploaded_file)

if "charges" not in df.columns:
    st.error("Dữ liệu phải có cột charges!")
    st.stop()

# Loại bỏ dữ liệu bị thiếu
df = df.dropna().copy()

# 3. TIỀN XỬ LÝ DỮ LIỆU
X_train_raw = df.drop("charges", axis=1)
y_train = df["charges"]

# Mã hóa các biến phân loại
X_train = pd.get_dummies(X_train_raw, drop_first=True)

# 4. HUẤN LUYỆN LINEAR REGRESSION
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# 5. HUẤN LUYỆN K-MEANS
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_train_scaled)

# 6. TẠO GIAO DIỆN NHẬP THÔNG TIN
st.header("Nhập thông tin cá nhân")

col1, col2 = st.columns(2)

# Lấy giá trị từ dữ liệu
with col1:
    age = st.number_input(
        "Tuổi",
        min_value=int(df["age"].min()),
        max_value=int(df["age"].max()),
        value=int(df["age"].median())
    )

    sex = st.selectbox(
        "Giới tính",
        sorted(df["sex"].unique())
    )

    bmi = st.number_input(
        "Chỉ số BMI",
        min_value=float(df["bmi"].min()),
        max_value=float(df["bmi"].max()),
        value=float(df["bmi"].median())
    )

with col2:
    children = st.number_input(
        "Số con",
        min_value=int(df["children"].min()),
        max_value=int(df["children"].max()),
        value=int(df["children"].median()),
        step=1
    )

    smoker = st.selectbox(
        "Tình trạng hút thuốc",
        sorted(df["smoker"].unique())
    )

    region = st.selectbox(
        "Khu vực sinh sống",
        sorted(df["region"].unique())
    )

# 7. NÚT DỰ ĐOÁN
if st.button("DỰ ĐOÁN", type="primary", use_container_width=True):

    # Tạo dữ liệu đầu vào
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # Mã hóa dữ liệu đầu vào
    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Căn chỉnh các cột giống dữ liệu huấn luyện
    input_encoded = input_encoded.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # 8. DỰ ĐOÁN BẰNG LINEAR REGRESSION
    prediction_lr = model_lr.predict(input_encoded)[0]

    # 9. PHÂN CỤM BẰNG K-MEANS
    input_scaled = scaler.transform(input_encoded)

    prediction_kmeans = kmeans.predict(input_scaled)[0]

    # 10. HIỂN THỊ KẾT QUẢ
    st.divider()
    st.header("KẾT QUẢ DỰ ĐOÁN")

    result1, result2 = st.columns(2)

    with result1:
        st.subheader("Linear Regression")
        st.metric(
            label="Chi phí bảo hiểm dự đoán",
            value=f"{prediction_lr:,.2f} USD"
        )

        st.success("Đã dự đoán chi phí bảo hiểm thành công!")

    with result2:
        st.subheader("K-means")
        st.metric(
            label="Nhóm dữ liệu",
            value=f"Cụm {prediction_kmeans + 1}"
        )

        st.success(
            f"Người dùng thuộc cụm số {prediction_kmeans + 1}."
        )

    # Hiển thị thông tin đã nhập
    st.divider()
    st.subheader("Thông tin đầu vào")

    st.dataframe(input_data, use_container_width=True)

    st.caption(
        "Lưu ý: K-means phân nhóm dữ liệu, không dự đoán chi phí bảo hiểm."
    )

# 11. THÔNG TIN CUỐI TRANG
st.divider()
st.caption("Ứng dụng thực hành thuật toán Linear Regression và K-means.")
