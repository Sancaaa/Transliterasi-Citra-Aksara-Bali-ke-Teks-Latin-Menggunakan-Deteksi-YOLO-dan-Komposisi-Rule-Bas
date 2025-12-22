#   teyvatLontar.py
"""
    @TeyvatLontar
    © Copyright 2025 Narendera Sancaya && Tim Teyvat101
    * Sebuah program Transliterasi AKsara Bali ke Latin menggunakan
    * YOLO untuk deteksi objek karakter Aksara Bali
    * Pre-processing dengan openCV dan kawan-kawannya
    * Rule-based transliteration
    * Post-processing denegan model LSTM (TO-DO)
    * Semoga mempermudah pejuang Aksara Bali lainnya di dunia ini
=====================================================================
                    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⣿⣷⣤⣿⣤⣨⣿⡀⠀⠀⠉⠙⠷⣤⠀⠈⢻⡇
                    ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⠴⠚⠋⠉⠉⠉⠉⠉⠛⠒⠦⣤⡀⠀⠈⢻⣆⠀⠹
                    ⠀⠀⠀⣴⢶⣶⡶⠖⢋⡽⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⠶⣾⣿⣠⡤
                    ⠀⠀⠀⣻⠦⢀⣠⣴⡿⠁⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣄⠀
                    ⠀⠀⠉⠀⢰⣿⣿⣿⣇⠀⢀⣴⡟⠀⠀⠀⠀⠀⠀⠀⣸⠀⠀⠀⠀⠀⠀⠀⠘⣷
                    ⢧⣀⠀⠀⢸⣿⡿⠟⢋⣴⠋⣸⠁⠀⠀⠀⠀⠀⠀⣴⢻⡶⡄⠀⠀⠀⠀⠀⠀⠘
                    ⠀⢨⠃⠀⡿⠋⢀⡴⠋⠀⢀⡟⠀⡀⠀⠀⠀⣹⣾⣅⠈⣇⡇⠀⠀⠀⠀⠀⠀⠀
                    ⣾⠃⠀⠀⠀⣰⠏⣀⡀⠀⢸⢷⡾⠁⣀⣤⠾⠋⠀⠀⠀⢹⣇⠀⠀⠀⠀⠁⠀⠀
                    ⡇⠀⠀⠀⣰⠏⣾⣿⣿⠀⠀⠘⠓⠛⠉⠁⢀⣴⣶⣄⠀⠀⣿⡄⠀⠀⠀⠀⠀⠀
                    ⠀⠀⠀⣠⡟⠀⠙⠛⠋⠀⠀⠀⠀⠀⢀⠀⠺⣿⣿⡿⠂⠀⡟⠁⠀⠀⠀⠀⢠⠀
                    ⠀⠀⠀⣿⠁⠀⠀⠀⠀⠸⠶⠶⡶⣾⠋⠀⠀⠈⠉⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⡄
                    ⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠀⠀⠀⢿⡄⠀⠀⠀⠀⠀⠀⢰⡇⠀⠀⠀⠀⠀⠀⡇
                    ⠀⠀⠀⢸⣍⣻⣶⣤⣤⣤⣤⣤⣀⣀⣀⡀⠀⠀⠀⠀⠀⣾⠁⠀⠀⠀⠀⠀⠀⡇
                    ⠀⢰⡄⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣷⣶⣿⠏⠀⠀⠀⠀⠀⠀⢠⡇
                    ⠀⠈⢿⣷⣦⣿⣿⣿⣿⣿⣿⣿⠟⠙⠿⠋⠈⢿⣿⣿⠏⠀⠀⠀⠀⠀⠀⢀⡞
=====================================================================
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import pandas as pd
import cv2
import os

# Import Rule Aksara Bali
from ruleAksara import ruleAksara
from majorLinesAksara import *
from debugSpatial import draw_spatial_debug, print_debug_log
from preprocess import PREPROCESSORS

#   ==========================================================================================================================================
#       Berjam-jam train model, jangan lupa untuk dipake yah
#       kalao modelnya ga sempurna, maafkan yah
#       satu-satunya yang semmpurna adalah senyummu v(⌒o⌒)v♪
#   ==========================================================================================================================================


#   Cek Model
MODEL_PATH = 'bestClassificationDetection.pt'
# MODEL_PATH = 'best.pt'

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' tidak ditemukan!")
    st.info("Pastikan file 'best.pt' berada di folder yang sama dengan script ini")
    st.stop()

#   Memuat model YOLOv8
@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()

#   Ambil image RGB menjadi BGR melalui YOLO model, di store via array np
#   Kenapa ke BGR? karena openCV hanya store BGR.
def process_image(image):

    img = np.array(image)
    # Convert RGB to BGR untuk YOLO
    img_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model(img_np)
    return results

#   Gambar bounding box dengan nomor urutan deteksi
def draw_detections_with_numbers(image, detections, sorted_indices=None):

    img_with_numbers = image.copy()

    for i, detection in enumerate(detections):
        xmin, ymin, xmax, ymax = detection['position']
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # Tentukan warna
        color = (77, 16, 4)  # Merah untuk urutan asli

        # Gambar bounding box
        cv2.rectangle(img_with_numbers, (xmin, ymin), (xmax, ymax), color, 2)

        # Gambar background untuk text

        label = f"{i + 1}"

        font_scale = 0.5
        thickness = 1

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        text_x = xmin + 5
        text_y = ymin + text_size[1] + 5

        cv2.rectangle( img_with_numbers,
            (text_x - 3, text_y - text_size[1] - 3),
            (text_x + text_size[0] + 3, text_y + 3),
            color,
            -1
        )

        # Tambahkan nomor
        cv2.putText(img_with_numbers, label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return img_with_numbers

def draw_major_lines(image, major_lines, color=(0,0,255), thickness=2):
    img = image.copy()
    for line in major_lines:
        # hitung y tengah line
        ys = [(d['position'][1] + d['position'][3]) / 2 for d in line]
        cy = int(sum(ys) / len(ys))

        xs_left  = [int(d['position'][0]) for d in line]
        xs_right = [int(d['position'][2]) for d in line]

        # biar tidak mepet
        min_x = min(xs_left)
        max_x = max(xs_right)
        # min_x = int(min(xs_left)*1.05)
        # max_x = int(max(xs_right)*0.95)

        cv2.line(img, (min_x, cy), (max_x, cy), color, thickness)
    return img

#   ==========================================================================================================================================
#       Sebenarnya admin besar Sanca tidak terlalu peduli sama ui inih
#       Ya pokoknya jalan dan bisa dipake debug sajah
#       Peace ehe (◕‿◕✿)
#   ==========================================================================================================================================


st.set_page_config(page_title="TeyvatLontar", page_icon="🕉", layout="wide")
st.title("🕉 TeyvatLontar: Transliterasi Aksara Bali ke Teks Latin")
st.markdown("""
**Convert gambar Aksara Bali menjadi Teks Latin menggunakan YOLO detection + Rule-based Composition**
""")

# Sidebar
st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.0, 0.05) # min, max, def, interval

st.sidebar.title("🖼️ Preprocessing")
preprocess_mode = st.sidebar.selectbox(
    "Preprocessing Mode",
    [
        "none",
        "gray",
        "gray+clahe",
        "gray+median+clahe"
    ]
)

st.sidebar.title("🐛 Debug Mode")
debug_spatial = st.sidebar.checkbox("Tampilkan Debug Spasial", True)
show_detection_details = st.sidebar.checkbox("Tampilkan Detail Deteksi", True)
show_composition_log = st.sidebar.checkbox("Tampilkan Log Komposisi", True)
show_numbered_detection = st.sidebar.checkbox("Tampilkan Gambar dengan Nomor Urutan", True)
show_major_line = st.sidebar.checkbox("Tampilkan Major Line Plot", True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Gambar")
    uploaded_file = st.file_uploader("Pilih gambar aksara Bali...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        original_array_image = np.array(original_image)
        st.image(original_image, caption="Gambar Input", use_container_width=True)
        st.write(f"**Info Gambar:** {original_image.size[0]} x {original_image.size[1]} pixels")
    else:
        original_image = None
        st.info("👆 Upload gambar aksara Bali untuk memulai konversi")


with col2:
    st.subheader("🔤 Hasil Konversi Latin")

    if original_image is not None and model is not None:
        with st.spinner("🔄 Memproses gambar dengan YOLO + Rule Engine..."):
            try:
                preprocess_fn = PREPROCESSORS[preprocess_mode]
                preprocess_image = preprocess_fn(original_array_image)
                results = process_image(preprocess_image)

                if len(results[0].boxes) > 0:
                    composer = ruleAksara('phonologyRulesAksara.yaml')
                    all_detections = []
                    result_itterate = []
                    boxes_data = results[0].boxes.data.cpu().numpy()

                    for box in boxes_data:
                        xmin, ymin, xmax, ymax, conf, class_id = box
                        if conf >= confidence_threshold:
                            class_id = int(class_id)
                            all_detections.append({
                                'class_id': class_id,
                                'name': composer.class_mapping[class_id],
                                'confidence': float(conf),
                                'position': (float(xmin), float(ymin), float(xmax), float(ymax))
                            })

                    # Sort berdasarkan aturan baca Aksara Bali
                    # sorted_detections = advanced_balinese_sorting(detections)
                    img_width, img_height = original_image.size
                    original_image_size = (img_width, img_height)



                    majorLines_detections, majorLines_plot = major_lines_aksara(
                        all_detections,
                        debug=True
                    )

                    for wianjana in majorLines_detections:
                        candidates, evaluated, chosen_modifiers, latin = composer.process_element(
                            wianjana, all_detections
                        )

                        result_itterate.append({
                            'wianjana': wianjana,
                            'candidates': candidates,
                            'evaluated': evaluated,
                            'chosen_modifiers': chosen_modifiers,
                            'latin': latin
                        })

                    # composer.reset()
    

                    # Dapatkan hasil akhir
                    latin_text = composer.get_latin_texts()

                    # Tampilkan hasil utama
                    st.success("**🎯 TEKS LATIN HASIL KONVERSI:**")

                    # Main output box
                    st.markdown(f"""
                    <div style='background-color: #f0f8ff; padding: 25px; border-radius: 10px; border-left: 5px solid #041677; margin: 20px 0;'>
                        <h3 style='margin: 0; color: #041677; font-size: 24px;'>{latin_text}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # Detection image dengan bounding boxes
                    result_image = results[0].plot()
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_image_rgb, caption="Hasil Deteksi YOLO", use_container_width=True)

                    # Gambar dengan nomor urutan deteksi
                    if show_numbered_detection and all_detections:
                        st.subheader("🔢 Deteksi dengan Nomor Urutan")


                        # Gambar deteksi dengan nomor urutan
                        min_idx = st.number_input("Min Index", min_value=0, value=0, step=1)
                        max_idx = st.number_input("Max Index", min_value=0, value=len(all_detections) - 1, step=1)

                        if max_idx >= len(all_detections):
                            max_idx = len(all_detections) - 1

                        if min_idx >= max_idx:
                            max_idx = min_idx

                        # sorted_detections = sorted(sorted_detections, key=lambda d: (d["position"][1], d["position"][0]))

                        filtered_sorted = majorLines_detections[min_idx: max_idx + 1]

                        # Convert original image to OpenCV format
                        original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

                        numbered_image = draw_detections_with_numbers(original_cv, filtered_sorted )
                        numbered_image_rgb = cv2.cvtColor(numbered_image, cv2.COLOR_BGR2RGB)

                        major_line_img = draw_major_lines(numbered_image, majorLines_plot)
                        major_line_rgb = cv2.cvtColor(major_line_img, cv2.COLOR_BGR2RGB)

                        if show_major_line == True:
                            st.image(major_line_rgb,
                                 caption="Deteksi dengan Nomor Urutan",
                                 use_container_width=True)
                        else:
                            st.image(numbered_image_rgb,
                                     caption="Deteksi dengan Nomor Urutan",
                                     use_container_width=True)

                    if debug_spatial == True:
                        st.subheader("🐛 Debug Spatial View")
                        max_wianjana = len(majorLines_detections) - 1
                        debug_idx = st.sidebar.number_input(
                            "Pilih Wianjana",
                            min_value=0,
                            max_value=max(0, max_wianjana),
                            value=0,
                            step=1
                        )

                        if debug_idx < len(majorLines_detections):
                            # Process ulang hanya untuk wianjana yang dipilih (for visualization)
                            result = result_itterate[debug_idx]

                            original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                            # print_debug_log(i, wianjana, neighbors, evaluated, chosen, latin)

                            debug_img = draw_spatial_debug(
                                original_cv,
                                result['wianjana'],
                                result['candidates'],
                                result['chosen_modifiers'],
                                debug_idx
                            )

                            debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

                            st.image(debug_img_rgb,
                                     caption=f"Debug Wianjana #{debug_idx}: {result['wianjana']['name']} → '{result['latin']}'", # 5 berarti latin
                                     use_container_width=True)
                        else:
                            st.warning(f"Index {debug_idx} melebihi jumlah wianjana ({len(result_itterate)})")



                    # Detail deteksi
                    if show_detection_details and all_detections:
                        st.subheader("📊 Detail Deteksi")
                        df_data = []
                        for i, det in enumerate(all_detections):
                            element_type = composer.classify_element(det['class_id'])
                            df_data.append({
                                'No': i + 1,
                                'Aksara': det['name'],
                                'Tipe': element_type,
                                'Confidence': f"{det['confidence']:.3f}",
                                'Posisi X': f"{det['position'][0]:.1f}",
                                'Posisi Y': f"{det['position'][1]:.1f}"
                            })

                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)

                        # Statistics
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        with col_stat1:
                            st.metric("Total Karakter", len(all_detections))
                        with col_stat2:
                            avg_conf = np.mean([d['confidence'] for d in all_detections])
                            st.metric("Rata-rata Confidence", f"{avg_conf:.3f}")
                        with col_stat3:
                            wianjana_count = sum(
                                1 for d in all_detections if composer.classify_element(d['class_id']) == 'wianjana')
                            st.metric("Wianjana", wianjana_count)
                        with col_stat4:
                            special_count = sum(1 for d in all_detections if
                                                any(special in d['name'] for special in ['gantungan', 'pengangge']))
                            st.metric("Modifier", special_count)

                    # Composition log untuk debugging
                    if show_composition_log:
                        st.subheader("🔍 Log Proses Komposisi")

                        log_df_data = []

                        for log_entry in composer.get_log():

                            # Log untuk wianjana compose
                            if log_entry.get('action') == 'compose':
                                # Ambil nama-nama modifier dari chosen_modifiers
                                modifier_names = []
                                for mod in log_entry.get('chosen_modifiers', []):
                                    mod_name = mod.get('name', 'unknown')
                                    mod_role = mod.get('role', '')
                                    modifier_names.append(f"{mod_name} ({mod_role})")

                                log_df_data.append({
                                    'Aksi': log_entry['action'],
                                    'Base': log_entry['element'],
                                    'Tipe': log_entry['type'],
                                    'Modifier': ', '.join(modifier_names) if modifier_names else '-',
                                    'Latin': log_entry.get('latin', '-')
                                })

                        if log_df_data:
                            log_df = pd.DataFrame(log_df_data)
                            st.dataframe(log_df, use_container_width=True)


                        raw_logs = composer.get_log()
                        with st.expander("🛠️ Debug Raw Logs"):
                            st.write("Raw logs dari composer:", raw_logs)

                    # Download section
                    st.subheader("💾 Download Hasil")

                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        # Download gambar hasil
                        result_pil = Image.fromarray(result_image_rgb)
                        buffered = io.BytesIO()
                        result_pil.save(buffered, format="PNG")

                        st.download_button(
                            label="📥 Download Gambar Hasil",
                            data=buffered.getvalue(),
                            file_name="hasil_deteksi.png",
                            mime="image/png",
                            use_container_width=True
                        )

                    with col_dl2:
                        # Download teks Latin
                        text_result = f"TEKS LATIN HASIL KONVERSI\n"
                        text_result += f"{'=' * 40}\n"
                        text_result += f"{latin_text}\n"
                        text_result += f"{'=' * 40}\n"
                        text_result += f"File: {uploaded_file.name}\n"
                        text_result += f"Total Karakter: {len(all_detections)}\n"
                        text_result += f"Confidence Rata-rata: {avg_conf:.3f}\n"

                        st.download_button(
                            label="📄 Download Teks Latin",
                            data=text_result,
                            file_name="teks_latin.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                else:
                    st.warning(
                        "❌ Tidak ada aksara Bali yang terdeteksi. Coba turunkan confidence threshold atau gunakan gambar yang lebih jelas.")

            except Exception as e:
                st.error(f"❌ Error saat memproses: {e}")

# Information Section
st.markdown("---")
st.subheader("ℹ️ Cara Kerja Sistem")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.write("**🔍 Proses Deteksi:**")
    st.markdown("""
    1. **Pre-Processing (TO DO)** - Meningkatkan Akurasi Deteksi Objek 
    2. **YOLO Detection** - Mendeteksi setiap karakter aksara Bali
    2. **Confidence Filter** - Menyaring deteksi dengan confidence tinggi
    3. **Position Sorting** - Mengurutkan Karakter berdasarkan posisi
    4. **Rule-Based Transliteration** - Melakukan transliterasi dengan aturan yang benar  
    4. **Post-Processing (TO DO)** - Menyusun tiap kata latin dengan benar
    """)

with col_info2:
    st.write("**🔄 Proses Komposisi:**")
    st.markdown("""
    1. **Rule-based Combining** - Menggabungkan karakter berdasarkan hierarki
    2. **Latin Conversion** - Konversi ke teks Latin
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p><b>🔤 Bali Script to Latin Converter</b></p>
        <p>Menggunakan YOLO detection + Rule-based composition untuk konversi yang akurat</p>
    </div>
    """,
    unsafe_allow_html=True
)