# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import os
from scipy.ndimage import uniform_filter
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.exposure import rescale_intensity
from skimage.segmentation import active_contour
from skimage.draw import polygon2mask
import joblib  # untuk load model dan scaler


# =============================
# ---------- UTILS -----------
# =============================
def crop_image_array(img, asal="Dataset"):
    if asal == "Ultrasound Telemed":
        x1, y1, x2, y2 = 380, 50, 780, 400
    else:
        x1, y1, x2, y2 = 100, 0, 500, 350
    return img[y1:y2, x1:x2]

def preprocess_image(image):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    img_clahe = clahe.apply(image)
    img_median = cv2.medianBlur(img_clahe, 3)
    return img_median

def dsflsmv_local_filter(img, window_size=5, iterations=2):
    img = img.astype(np.float32)
    for _ in range(iterations):
        local_mean = uniform_filter(img, size=window_size)
        local_mean_sq = uniform_filter(img**2, size=window_size)
        local_var = local_mean_sq - local_mean**2
        global_var = np.mean(local_var)
        k = (local_var - global_var) / (local_var + global_var + 1e-8)
        k = np.clip(k, 0, 1)
        img = local_mean + k * (img - local_mean)
    return np.clip(img, 0, 255).astype(np.uint8)

# Fungsi ekstraksi SGLDM ASM
def sgldm_asm(img_gray, levels=8, radius=1):
    quantized = np.floor(img_gray / (256 / levels)).astype(np.uint8)
    counts = np.zeros(levels, dtype=np.float64)
    h, w = quantized.shape
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            center = quantized[y, x]
            neighbors = quantized[y - radius:y + radius + 1, x - radius:x + radius + 1].flatten()
            for val in neighbors:
                if val == center:
                    counts[center] += 1
    total = counts.sum()
    if total > 0:
        prob = counts / total
        asm = np.sum(prob ** 2)
    else:
        asm = np.nan
    return asm
def coarseness_ngtdm(img_gray):
    h, w = img_gray.shape
    diffs = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = img_gray[y, x]
            neighbors = [img_gray[y - 1, x], img_gray[y + 1, x], img_gray[y, x - 1], img_gray[y, x + 1]]
            diffs.append(np.mean([abs(int(center) - int(n)) for n in neighbors]))
    if len(diffs) == 0:
        return np.nan
    avg_diff = np.mean(diffs)
    return 1 / avg_diff if avg_diff != 0 else np.nan

def fractal_dimension(img, threshold):
    Z = (img < threshold * img.max()).astype(np.uint8)
    def boxcount(Z, k):
        S = (Z.shape[0] // k, Z.shape[1] // k)
        count = 0
        for i in range(S[0]):
            for j in range(S[1]):
                block = Z[i * k:(i + 1) * k, j * k:(j + 1) * k]
                if np.any(block):
                    count += 1
        return count
    sizes = 2 ** np.arange(int(np.log2(min(Z.shape))), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    logsizes = np.log(sizes)
    logcounts = np.log(counts)
    coeffs = np.polyfit(logsizes, logcounts, 1)
    return -coeffs[0]

@st.cache_resource(show_spinner=False)
def load_all_models():
    model_tekstur = joblib.load("svm_model_tekstur.joblib")
    scaler_tekstur = joblib.load("scaler_tekstur.joblib")

    model_gabungan = joblib.load("svm_model_gabungan.joblib")
    scaler_gabungan = joblib.load("scaler_gabungan.joblib")

    model_6fitur = joblib.load("svm_model_6fitur.joblib")
    scaler_6fitur = joblib.load("scaler_6fitur.joblib")

    return (model_tekstur, scaler_tekstur), (model_gabungan, scaler_gabungan), (model_6fitur, scaler_6fitur)


(model_tekstur, scaler_tekstur), (model_gabungan, scaler_gabungan), (model_6fitur, scaler_6fitur) = load_all_models()



# =============================
# ---------- PAGES -----------
# =============================

def page_home():
    st.title("Deteksi Potensi Aterosklerosis berdasarkan Tekstur Dinding Arteri Citra Ultrasound")
    st.markdown("""
    **Apa itu Aterosklerosis?**

    Aterosklerosis merupakan penumpukan plak pada dinding arteri yang mengakibatkan kekakuan arteri yang dapat berujung pada komplikasi serius seperti serangan jantung dan stroke. 32% kematian global disebabkan oleh Cardiovascular Disease (CVD), berdasarkan data WHO dengan Aterosklerosis sebagai bentuk yang paling umum.

   **Faktor Risiko:**  
    - **Diet tidak sehat**: tinggi lemak jenuh, garam, dan gula signifikan meningkatkan risiko aterosklerosis.  
    - **Gaya hidup sedentari**: kurangnya aktivitas fisik, meningkatkan risiko CVD.  
    - **Merokok**: faktor risiko utama, mempercepat proses aterosklerosis.  
    - **Diabetes Melitus (DM)**: kondisi hiperglikemia kronis, berkontribusi meningkatkan risiko aterosklerosis.

    **Kenapa Tekstur Dinding Arteri Penting?**  
    Melalui analisis tekstur, karakteristik spesifik yang mencerminkan perubahan komposisi seperti distribusi kolagen dan elastin dapat diungkap melalui pola tekstur pada citra USG.  
    Beberapa studi telah mengonfirmasi keunggulan analisis tekstur dalam mengidentifikasi risiko CVD:  
    - **GLDS Contrast**: Asosiasi signifikan dengan faktor risiko CVD.  
    - **SF GSM**: Digunakan untuk membedakan dampak faktor risiko (misalnya, merokok).  
    - **Kombinasi GLDS Contrast dan SGLDM ASM**: Meningkatkan prediktivitas risiko CVD.
    """)
    st.markdown("---")

def page_deteksi():
    st.title("🔍 Deteksi Potensi Aterosklerosis")

    asal_citra = st.radio("Asal Citra:", ["Dataset", "Ultrasound Telemed"], horizontal=True)
    metode = st.radio("Pilih metode input:", ["Tanpa data klinis", "Dengan data klinis"], horizontal=True)
    # Pilihan model hanya muncul jika dengan data klinis
    if metode == "Dengan data klinis":
        modelSVM = st.radio("Model yang digunakan:", ["Model gabungan (3 fitur)", "Model 6 fitur"], horizontal=True)
    else:
        modelSVM = "Model 3 fitur"
    uploaded_file = st.file_uploader("Unggah citra arteri", type=["tiff", "tif", "png", "jpg", "jpeg"])

    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

        original_img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)

        st.subheader("📂 Citra Original")
        st.image(original_img, caption="Citra Original", width=400)

        st.subheader("✂️ Proses Cropping")
        cropped_img = crop_image_array(original_img, asal=asal_citra)
        st.image(cropped_img, caption="Citra setelah Cropping (350x400)", width=400)

        st.subheader("✨ Proses Filtering")
        preprocessed = preprocess_image(cropped_img)
        filtered_img = dsflsmv_local_filter(preprocessed)
        st.image(filtered_img, caption="Filtered (CLAHE + Median + DsFlsmv)", width=400)

        img_float = filtered_img.astype(np.float32) / 255.0
        height, width = img_float.shape
        interval = 5
        columns_to_plot = list(range(0, width, interval))
        if columns_to_plot[-1] != width - 1:
            columns_to_plot.append(width - 1)

        st.subheader("📊 Profil Intensitas per Kolom (Interaktif)")
        selected_x = st.slider("Pilih kolom x", min_value=min(columns_to_plot), max_value=max(columns_to_plot), step=interval)
        fig_prof, ax_prof = plt.subplots(figsize=(6, 3))
        ax_prof.plot(img_float[:, selected_x])
        ax_prof.set_title(f"Profil Intensitas Kolom x = {selected_x}")
        ax_prof.set_xlabel("Y (baris)")
        ax_prof.set_ylabel("Intensitas")
        ax_prof.set_ylim(0, 1)
        ax_prof.grid(True)
        st.pyplot(fig_prof)

        st.subheader("🧾 Input Data Klinis")
        if metode == "Dengan data klinis":
            usia = st.number_input("Usia", min_value=0, max_value=120)
            perokok = st.selectbox("Apakah pasien merokok?", ["Tidak", "Ya"])
            bungkus_per_tahun = st.number_input("Bungkus per tahun", min_value=0.0)
            hipertensi = st.selectbox("Hipertensi?", ["Tidak", "Ya"])
            diabetes = st.selectbox("Diabetes?", ["Tidak", "Ya"])
        else:
            st.info("Tanpa input data klinis.")

        st.subheader("📍 Segmentasi Berdasarkan Threshold")
        threshold = st.number_input("Threshold intensitas (default 0.2)", value=0.2, format="%.3f")
        min_y_dark_start = st.number_input("Y minimal untuk area gelap (default 150)", value=150, step=1)
        max_y_dark_end = st.number_input("Y maksimal", value=400, min_value=0, max_value=1000, step=1)

        min_dark_length = 20
        pixels_after_dark = 20
        y_ranges = {}
        for x in columns_to_plot:
            profile = img_float[:, x]
            is_dark = profile < threshold
            start_idx = None
            longest_start = None
            longest_len = 0
            for i in range(len(is_dark)):
                if is_dark[i]:
                    if start_idx is None:
                        start_idx = i
                else:
                    if start_idx is not None:
                        dark_len = i - start_idx
                        if dark_len > longest_len and start_idx > min_y_dark_start and start_idx < max_y_dark_end :
                            longest_len = dark_len
                            longest_start = start_idx
                        start_idx = None
            if start_idx is not None:
                dark_len = len(is_dark) - start_idx
                if dark_len > longest_len and start_idx > min_y_dark_start and start_idx < max_y_dark_end:
                    longest_len = dark_len
                    longest_start = start_idx
            if longest_len >= min_dark_length:
                y_start = longest_start + longest_len - 5
                y_end = min(y_start + pixels_after_dark - 5, img_float.shape[0])
                y_ranges[x] = (y_start, y_end, longest_start, longest_len)

        if y_ranges:
            st.subheader("🎚️ Visualisasi Interaktif Hasil Segmentasi")
            x_slider = st.slider("Pilih kolom x (segmentasi)", min_value=min(y_ranges.keys()), max_value=max(y_ranges.keys()), step=interval)
            profile = img_float[:, x_slider]
            y_start, y_end, longest_start, longest_len = y_ranges[x_slider]

            fig, axs = plt.subplots(1, 2, figsize=(10, 3))
            axs[0].imshow(filtered_img, cmap='gray')
            axs[0].axvline(x=x_slider, color='red', linestyle='--', label=f'x = {x_slider}')
            axs[0].axhline(y=y_start, color='green', linestyle='--', label='y_start')
            axs[0].axhline(y=y_end, color='blue', linestyle='--', label='y_end')
            axs[0].set_title(f"Citra Filtered")
            axs[0].legend()

            axs[1].plot(profile, label='Intensitas')
            axs[1].axvspan(longest_start, longest_start + longest_len, color='gray', alpha=0.3, label='Segmen Gelap')
            axs[1].axvline(x=y_start, color='green', linestyle='--', label='y_start')
            axs[1].axvline(x=y_end, color='blue', linestyle='--', label='y_end')
            axs[1].set_title(f"Profil Intensitas x = {x_slider}")
            axs[1].set_xlabel('Y (baris)')
            axs[1].set_ylabel('Intensitas')
            axs[1].legend()
            axs[1].grid(True)
            st.pyplot(fig)

            # ==== Visualisasi Kontur dan Masking ====
            st.subheader("🔺 Kontur dan Masking Area Dinding Arteri")
            xs_sorted = sorted(y_ranges.keys())
            upper_curve = [[y_ranges[x][0], x] for x in xs_sorted]
            lower_curve = [[y_ranges[x][1], x] for x in reversed(xs_sorted)]
            init_snake_closed = np.array(upper_curve + lower_curve)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.imshow(filtered_img, cmap='gray')
            ax2.plot(init_snake_closed[:, 1], init_snake_closed[:, 0], '--r', label='Kontur Area Crop')
            ax2.set_title("Kontur Area Crop dari y_ranges")
            ax2.legend()
            st.pyplot(fig2)

            mask = polygon2mask(filtered_img.shape, init_snake_closed)
            masked_image = np.zeros_like(filtered_img)
            masked_image[mask] = filtered_img[mask]

            fig3, ax3 = plt.subplots(1, 2, figsize=(10, 4))
            ax3[0].imshow(mask, cmap='gray')
            ax3[0].set_title("Mask dari Kontur Tertutup")

            ax3[1].imshow(masked_image, cmap='gray')
            ax3[1].set_title("Citra Hasil Potong (dalam kontur)")
            st.pyplot(fig3)

            ys, xs = np.where(mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            cropped_result = masked_image[y_min:y_max+1, x_min:x_max+1]

            st.subheader("📤 Hasil Akhir Segmentasi")
            st.image(cropped_result, caption="Citra Crop Area Kontur", width=400)

            # ====================
            # === EKSTRAKSI FITUR
            # ====================
            st.subheader("📊 Ekstraksi Fitur Tekstur dari Area Segmentasi")

            # Pastikan cropped_result tipe uint8
            if cropped_result.dtype != np.uint8:
                img_gray = (cropped_result / np.max(cropped_result) * 255).astype(np.uint8)
            else:
                img_gray = cropped_result

            total_median = 0
            total_contrast = 0
            n = len(y_ranges)

            for x in y_ranges:
                y_start, y_end, _, _ = y_ranges[x]
                intensities = img_float[y_start:y_end, x]
                total_median += np.median(intensities)
                total_contrast += np.var(intensities)

            avg_median = total_median / n
            avg_contrast = total_contrast / n

            asm_val = sgldm_asm(img_gray, levels=8, radius=1)
            coarse_val = coarseness_ngtdm(img_gray)
            fdta_h2 = fractal_dimension(img_float, threshold=0.7)
            fdta_h4 = fractal_dimension(img_float, threshold=0.9)

            st.write(f"- SF GSM : **{avg_median:.4f}**")
            st.write(f"- GLDS Contrast : **{avg_contrast:.4f}**")
            st.write(f"- SGLDM ASM : **{asm_val:.4f}**")
            st.write(f"- Coarseness (NGTDM)    : **{coarse_val:.4f}**")
            st.write(f"- H2FDTA      : **{fdta_h2:.4f}**")
            st.write(f"- H4FDTA      : **{fdta_h4:.4f}**")

            # =================
            # Siapkan fitur untuk prediksi
            # =================
            # Persiapkan fitur tekstur dalam urutan sama seperti training
            fitur_tekstur = [asm_val, avg_median, avg_contrast]

            # Data klinis input
            if metode == "Dengan data klinis":
                smoking_num = 1 if perokok == "Ya" else 0
                hipertensi_num = 1 if hipertensi == "Ya" else 0
                diabetes_num = 1 if diabetes == "Ya" else 0
                fitur_klinis = [usia, bungkus_per_tahun, smoking_num, hipertensi_num, diabetes_num]
            else:
                fitur_klinis = []

            fitur_tekstur_2 = [coarse_val, fdta_h2, fdta_h4]
            fitur_tekstur_3 = [asm_val, coarse_val, fdta_h2, fdta_h4]

            if metode == "Tanpa data klinis":
                model, scaler = model_tekstur, scaler_tekstur
                fitur_input = fitur_tekstur_3

            else:
                if modelSVM == "Model gabungan (3 fitur)":
                    model, scaler = model_gabungan, scaler_gabungan
                    fitur_input = fitur_klinis + fitur_tekstur_3

                elif modelSVM == "Model 6 fitur":
                    model, scaler = model_6fitur, scaler_6fitur
                    fitur_input = fitur_klinis + fitur_tekstur + fitur_tekstur_2


            X_input = np.array(fitur_input).reshape(1, -1)
            X_scaled = scaler.transform(X_input)

            # Prediksi dengan model SVM
            probas = model.predict_proba(X_scaled)[0]
            prediksi = model.predict(X_scaled)[0]
            confidence = probas[prediksi]

            st.write(f"Prediksi Potensi Aterosklerosis: **{'Positif' if prediksi == 1 else 'Negatif'}**")
            st.write(f"Probabilitas Deteksi (keyakinan model): **{confidence*100:.2f}%**")

            if confidence >= 0.7:
                st.success(f"✅ Model yakin pasien {'berpotensi CVD' if prediksi == 1 else 'tidak menunjukkan gejala CVD'}.")
            elif confidence <= 0.3:
                st.info(f"🟢 Model cukup yakin pasien {'tidak menunjukkan gejala CVD' if prediksi == 0 else 'berpotensi CVD'}.")
            else:
                st.warning("⚠️ Model tidak terlalu yakin terhadap hasil ini. Pertimbangkan pemeriksaan lanjutan.")


# =============================
# --------- ROUTER -----------
# =============================
def main():
    st.set_page_config(page_title="Aterosklerosis Detector", layout="wide")

    pages = {
        "Home": page_home,
        "Deteksi": page_deteksi,
    }

    with st.sidebar:
        st.title("📌 Menu")
        selected = st.selectbox("Navigasi", list(pages.keys()))

    pages[selected]()

if __name__ == "__main__":
    main()
