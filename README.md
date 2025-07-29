# Deteksi Potensi Aterosklerosis dengan Analisis Tekstur Citra Ultrasound

Aplikasi berbasis **Streamlit** ini dikembangkan untuk membantu mendeteksi potensi **aterosklerosis** berdasarkan analisis tekstur dinding arteri dari citra ultrasound pasien. 

Aplikasi ini mendukung:
- **Pra-pemrosesan otomatis** (CLAHE, Median Filter, DsFlsmv)
- **Segmentasi area dinding arteri**
- **Ekstraksi fitur tekstur** (GLDS Contrast, SGLDM ASM, SF GSM, FDTA)
- **Prediksi klasifikasi potensi CVD** menggunakan model SVM

## 📦 Fitur Utama
- Unggah citra ultrasound (TIFF, PNG, JPG, dll)
- Input data klinis (usia, merokok, DM, dll)
- Visualisasi proses segmentasi & intensitas
- Prediksi potensi aterosklerosis


