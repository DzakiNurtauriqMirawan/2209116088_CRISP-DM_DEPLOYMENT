import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np


# Membaca data dari file CSV
df = pd.read_csv('Data Cleaning.csv')


# # Gunakan One-Hot Encoding untuk mengubah kolom kategorikal menjadi bentuk numerik
# X = pd.get_dummies(df.drop('rate.dailyCategory', axis=1)) 
# y = df['rate.dailyCategory'] 

# # Bagi data menjadi data pelatihan dan data pengujian
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Inisialisasi model DecisionTreeClassifier
# model = DecisionTreeClassifier()

# # Latih model menggunakan data pelatihan
# model.fit(X_train, y_train)

# Judul halaman
st.image("https://1.bp.blogspot.com/-36_4mOSiZ5w/VnfelAx8mII/AAAAAAAAABc/xsmNetTD60s/s1600/car%2Brentals%2Bhire.jpg")
st.title("Car Rental Data")



# Judul sidebar
st.sidebar.title('Dashboard')

# Daftar navigasi
nav_selection = st.sidebar.multiselect("Go to", ["Home", "Distribusi", "Hubungan", "Perbandingan dan Komposisi", "Predict"])

# Jika pilihan di sidebar adalah "Home"
if "Home" in nav_selection:
    # st.image("https://1.bp.blogspot.com/-36_4mOSiZ5w/VnfelAx8mII/AAAAAAAAABc/xsmNetTD60s/s1600/car%2Brentals%2Bhire.jpg")
    st.image("https://i1.wp.com/bcomnotes.in/wp-content/uploads/2020/05/objectives-of-business-scaled.jpg?w=2400&ssl=1")
    st.subheader("Business Objective")
    st.write("Tujuan utama dari proyek ini adalah untuk mendorong keputusan bisnis yang terinformasi dan cerdas dalam industri persewaan mobil, termasuk dinamika pasar seperti popularitas kendaraan, harga sewa tipikal, potensi kesenjangan dan kejenuhan pasar.Kami fokus pada analisis data yang dikumpulkan secara independen untuk memahami  di berbagai kota besar di Amerika.")
    st.image("https://th.bing.com/th/id/OIP.hynVAaYDguoORn4RelYsugHaE8?rs=1&pid=ImgDetMain")
    st.subheader("Assess Situation")
    st.write("Situasi yang dinilai mencakup proses pengumpulan data yang dilakukan pada bulan Juli 2020 yang mencakup penambangan web dari berbagai sumber online di kota-kota besar di AS (Amerika Serikat). Metode ini menggunakan skrip scraping yang dikembangkan berdasarkan masukan dari komunitas StackOverflow dan memberikan pendekatan kreatif dan teknis untuk mengatasi tantangan pengumpulan data skala besar.")
    st.image("https://wisdomspringstraining.com/wp-content/uploads/2018/09/get-with-goals.jpg")
    st.subheader("Data Mining Goals")
    st.write("Tujuan utama dari penggalian data ini adalah untuk mengidentifikasi pola dan tren yang relevan dalam industri persewaan mobil, termasuk mengidentifikasi merek dan model mobil paling populer di setiap kota, menganalisis harga sewa pada umumnya, dan menilai potensi kejenuhan dan kesenjangan pasar itu bisa dieksploitasi. Selain itu, proyek ini juga akan menyelidiki hubungan antara peringkat yang diberikan oleh pengguna di situs persewaan mobil dan persepsi keandalan serta konsistensi nilai peringkat yang tinggi.")
    st.image("https://th.bing.com/th/id/OIP.y0zfban0QRxBy4Q17M2DrQHaGM?rs=1&pid=ImgDetMain")
    st.subheader("Project Plan")
    st.write("Rencana proyek mencakup langkah-langkah sistematis mulai dari pengumpulan data menggunakan skrip scraping yang dikembangkan hingga pembersihan dan pemrosesan data untuk memastikan kualitas dan konsistensi data yang dihasilkan. Analisis data kemudian dilakukan dengan menggunakan berbagai metode statistik dan teknik penambangan data untuk mengidentifikasi pola dan tren yang dapat memberikan wawasan berharga bagi bisnis . Hasil analisis akan diinterpretasikan secara cermat untuk mengambil keputusan strategis berdasarkan data, dengan harapan melalui pembagian kumpulan data yang dihasilkan kepada publik, akan memberikan kontribusi positif bagi industri persewaan mobil dan bermanfaat bagi masyarakat umum.")


# Jika pilihan di sidebar adalah "Distribusi"
if "Distribusi" in nav_selection:
    st.title("Distribusi")
    # Menyiapkan data
    data = df.groupby('vehicle.type')['reviewCount'].sum()

    # Membuat bar chart menggunakan Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Total Ulasan Berdasarkan Jenis Kendaraan')
    ax.set_xlabel('Jenis Kendaraan')
    ax.set_ylabel('Total Ulasan')
    ax.tick_params(axis='x', rotation=45)  # Mengatur rotasi label sumbu x
    ax.grid(axis='y')

    # Menampilkan bar chart menggunakan Streamlit
    st.pyplot(fig)

    # Menambahkan judul dan deskripsi
    st.write("Visualisasi di atas menampilkan total ulasan berdasarkan jenis kendaraan yang menggunakan bar plot. Dengan data Mobil: 71.000 ulasan, Minivan: 4.000 ulasan, SUV: 30.000 ulasan, Truk: 2.500 ulasan, dan Van: 0 ulasan. Semakin tinggi bar plotnya maka menandakan bahwa jenis kendaraan tersebut memiliki banyak ulasan")
    # st.write("Semakin tinggi bar plotnya maka menandakan bahwa jenis kendaraan tersebut memiliki banyak ulasan")
    # st.write("Data: Mobil: 71.000 ulasan")
    # st.write("Data Minivan: 4.000 ulasan")
    # st.write("Data SUV: 30.000 ulasan")
    # st.write("Data Truk: 2.500 ulasan")
    # st.write("Data Van: 0 ulasan")

# Jika pilihan di sidebar adalah "Hubungan"
if "Hubungan" in nav_selection:
    st.title("Hubungan")

    # Menghapus kolom non-numerik atau melakukan encoding jika diperlukan
    # Misal, melakukan encoding untuk kolom 'category'
    # df_file = pd.get_dummies(df_file, columns=['category'])

    # Menghitung korelasi antar kolom numerik
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    # Membuat heatmap korelasi
    st.write("Heatmap Korelasi Antar Kolom Numerik")
    st.write(corr)

    # Membuat figure dan axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Membuat heatmap menggunakan Seaborn
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Heatmap Korelasi Antar Kolom Numerik')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.write("tabel statistik diatas menunjukkan hubungan antara berbagai faktor yang memengaruhi Car Rental Data, seperti rating, RenterTripsTaken, reviewCount, location.latitude, owner.id, rate.daily, dan tahun vehicle.year.")
    # Menampilkan heatmap menggunakan Streamlit
    st.pyplot(fig)

    st.write("visualisasi diatas adalah sebuah heatmap yang menunjukkan korelasi antara kolom data numerik yang berbeda. Heatmap dibagi menjadi sembilan kotak, yang masing-masing mewakili korelasi antara dua kolom data. Warna setiap kotak menunjukkan kekuatan korelasi, dengan merah menunjukkan korelasi positif yang kuat, biru menunjukkan korelasi negatif yang kuat, dan putih menunjukkan tidak ada korelasi.")

# Panggil fungsi compositionAndComparison jika "Perbandingan dan Komposisi" ada dalam nav_selection
if "Perbandingan dan Komposisi" in nav_selection:
    st.title("Perbandingan dan Komposisi")

    def compositionAndComparison(df):
        # Hapus kolom non-numerik dari DataFrame
        df_numeric = df.select_dtypes(include=['float64', 'int64'])

        # Gabungkan kolom 'rate.dailyCategory' dengan DataFrame numerik
        if 'rate.dailyCategory' in df.columns:
            df_numeric['rate.dailyCategory'] = df['rate.dailyCategory']  # Tambahkan kolom 'rate.dailyCategory' ke DataFrame numerik
        else:
            st.write("Column 'rate.dailyCategory' not found in the DataFrame.")
            return

        # Hitung modus fitur untuk setiap kelas
        class_composition = df_numeric.groupby('rate.dailyCategory').agg(lambda x: x.mode().iloc[0])

        # Plot komposisi kelas
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(class_composition.T, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax)
        ax.set_title('Composition for each class')
        ax.set_xlabel('Label')
        ax.set_ylabel('Category')
        st.pyplot(plt.gcf())  # Menampilkan plot menggunakan Streamlit
        st.write("visualisasi tersebut menunjukkan perbandingan dan komposisi untuk setiap kategori. Perbandingan dan komposisi tersebut ditunjukkan dengan angka berwarna hijau, kuning, dan biru tua. Performa kategori diukur berdasarkan nilai rata-rata, jumlah perjalanan, jumlah ulasan, lintang lokasi, bujur lokasi, tarif harian, dan tahun kendaraan, di mana nilai hijau menunjukkan kinerja terbaik, kuning menunjukkan kinerja sedang, dan biru tua menunjukkan kinerja terburuk. Angka yang lebih tinggi menunjukkan nilai yang lebih baik, angka yang sama menunjukkan nilai yang setara, dan angka yang lebih rendah menunjukkan nilai yang lebih buruk")

    compositionAndComparison(df)


# Gunakan One-Hot Encoding untuk mengubah kolom kategorikal menjadi bentuk numerik
X = pd.get_dummies(df.drop(['YearCategory', 'rate.dailyCategory'], axis=1)) 
y = df[['YearCategory', 'rate.dailyCategory']]

# Bagi data menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model DecisionTreeClassifier
model = DecisionTreeClassifier()

# Latih model menggunakan data pelatihan
model.fit(X_train, y_train)

# Simpan model yang telah dilatih ke dalam file 'dtc.pkl'
with open('dtc.pkl', 'wb') as f:
    pickle.dump(model, f)

# Fungsi untuk memuat model yang telah dilatih
def load_model():
    with open('dtc.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Memuat model yang telah dilatih
model = load_model()

# def get_quality_text(prediction):
#     texts = []
#     for pred in prediction:
#         if (pred == 0).all():
#             texts.append("Buruk")
#         elif (pred == 1).all():
#             texts.append("Sedang")
#         elif (pred == 2).all():
#             texts.append("Baik")
#         else:
#             texts.append("Tidak Diketahui")
#     return texts


# def get_rating_text(prediction):
#     texts = []
#     for pred in prediction:
#         if (pred == 0).all():
#             texts.append("Buruk")
#         elif (pred == 1).all():
#             texts.append("Sedang")
#         elif (pred == 2).all():
#             texts.append("Baik")
#         else:
#             texts.append("Tidak Diketahui")
#     return texts


# Jika pilihan di sidebar adalah "Predict"
if "Predict" in nav_selection:
    st.title("Predict")

    # Tampilkan form input untuk pengguna
    st.subheader("Predict Car Rental Data")
    fuel_type = st.selectbox("Fuel Type", df['fuelType'].unique())
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
    renter_trips_taken = st.number_input("Renter Trips Taken", min_value=0, step=1)
    review_count = st.number_input("Review Count", min_value=0, step=1)
    city = st.text_input("City")
    country = st.text_input("Country")
    latitude = st.number_input("Latitude")
    longitude = st.number_input("Longitude")
    state = st.text_input("State")
    # owner_id = st.text_input("Owner ID")
    daily_rate = st.number_input("Daily Rate", min_value=0, step=1)
    make = st.text_input("Vehicle Make")
    model_input = st.text_input("Vehicle Model")
    vehicle_type = st.selectbox("Vehicle Type", df['vehicle.type'].unique())
    vehicle_year = st.number_input("Vehicle Year", min_value=1900, max_value=2025, step=1)

    # Jika tombol "Predict" ditekan
    if st.button("Predict"):
        # Ubah input pengguna menjadi DataFrame
        input_data = pd.DataFrame({
            "fuelType": [fuel_type],
            "rating": [rating],
            "renterTripsTaken": [renter_trips_taken],
            "reviewCount": [review_count],
            "location.city": [city],
            "location.country": [country],
            "location.latitude": [latitude],
            "location.longitude": [longitude],
            "location.state": [state],
            # "owner.id": [owner_id],
            "rate.daily": [daily_rate],
            "vehicle.make": [make],
            "vehicle.model": [model_input],
            "vehicle.type": [vehicle_type],
            "vehicle.year": [vehicle_year],
        })

        # Lakukan One-Hot Encoding untuk kolom kategorikal (jika diperlukan)
        input_data_encoded = pd.get_dummies(input_data)

        # Pastikan kolom-kolom yang dihasilkan dari one-hot encoding pada data pengguna
        # sesuai dengan kolom-kolom yang dihasilkan dari one-hot encoding pada data pelatihan
        missing_columns = set(X_train.columns) - set(input_data_encoded.columns)
        for col in missing_columns:
            input_data_encoded[col] = 0

        # Pastikan urutan kolom-kolom pada data pengguna sesuai dengan urutan kolom-kolom pada data pelatihan
        input_data_encoded = input_data_encoded.reindex(columns=X_train.columns, fill_value=0)

        # Prediksi dengan model yang telah dilatih
        prediction = model.predict(input_data_encoded)

        # Flatten the prediction array to ensure it's 1D
        prediction = prediction.flatten()

        # Tampilkan hasil prediksi sebagai teks
        prediction_text = f"Kualitas mobil rental: {prediction[0]}, Rating mobil rental: {prediction[1]}"
        st.write(prediction_text)


