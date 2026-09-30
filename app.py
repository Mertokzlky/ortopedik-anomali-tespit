# Dosya adı: app.py
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from PIL import Image
from toplu_tahmin import csv_oku, ozellikleri_hazirla, tahmin_et, CSVHatasi, OZELLIKLER

# --- AYARLAR ---
st.set_page_config(page_title="Fizik Tedavi KDS", page_icon="🏥", layout="wide")

# 2 sınıflı ve 3 sınıflı mod için gereken tüm dosya adları ve Türkçe
# etiket eşleştirmeleri burada toplanıyor. Yeni bir mod eklemek istersen
# (ör. farklı bir veri seti) sadece bu sözlüğe bir kayıt eklemen yeterli.
MOD_AYARLARI = {
    "2sinif": {
        "etiket": "2 Sınıf (Normal / Anormal)",
        "model_dosya": "fiziktedavi_model.pkl",
        "skor_dosya": "model_skorlari.pkl",
        "veri_dosya": "column_2C.csv",
        "turkce": {"Normal": "Normal", "Abnormal": "Anormal"},
    },
    "3sinif": {
        "etiket": "3 Sınıf (Normal / Disk Hernisi / Spondilolistezis)",
        "model_dosya": "fiziktedavi_model_3sinif.pkl",
        "skor_dosya": "model_skorlari_3sinif.pkl",
        "veri_dosya": "column_3C.csv",
        "turkce": {
            "Normal": "Normal",
            "Disk_Hernia": "Disk Hernisi",
            "Spondylolisthesis": "Spondilolistezis",
        },
    },
}

# RESİM YÜKLEME FONKSİYONU
def resim_goster(dosya_adi, genislik=None, altyazi=None):
    # Görsel dosyası (banner.jpg / anatomi.jpg) projeye eklenmemişse
    # sayfa hata vermeden devam eder; bu görseller isteğe bağlıdır.
    if os.path.exists(dosya_adi):
        img = Image.open(dosya_adi)
        if genislik:
            st.image(img, width=genislik, caption=altyazi)
        else:
            st.image(img, use_container_width=True, caption=altyazi)

def araligi_kontrol_et(df, kolon_adi, deger, etiket):
    if df is None:
        return
    p05 = df[kolon_adi].quantile(0.05)
    p95 = df[kolon_adi].quantile(0.95)
    if deger < p05 or deger > p95:
        st.warning(f"⚠️ {etiket} değeri, hastaların %90'ının bulunduğu tipik aralığın ({p05:.1f} - {p95:.1f}) dışında. Uç bir değer.")


@st.cache_data
def veriyi_yukle(dosya_yolu):
    df = pd.read_csv(dosya_yolu)
    df.columns = ['Pelvik_İnsidans', 'Pelvik_Eğim', 'Lumbar_Lordoz_Açısı',
                  'Sakral_Eğim', 'Pelvik_Yarıçap', 'Spondilolistezis_Derecesi', 'Durum']
    return df

# Modelleri yükle (cache: sayfa her etkileşimde yeniden yüklemez)
@st.cache_resource
def modeli_yukle(model_dosya, skor_dosya):
    model = joblib.load(model_dosya)
    skorlar = joblib.load(skor_dosya)
    return model, skorlar

# --- BAŞLIK KISMI ---
col_logo, col_baslik = st.columns([1, 4])
with col_logo:
    resim_goster("banner.jpg", genislik=150)
with col_baslik:
    st.title("🏥 Ortopedik Anomali Tespit Sistemi")
    st.markdown("**Makine Öğrenmesi Destekli Karar Destek Sistemi**")

# --- MOD SEÇİMİ (2 SINIF / 3 SINIF) ---
# Seçilen moda göre farklı model ve veri dosyaları kullanılır; aşağıdaki
# 3 sekmenin (Tahmin, Analiz, Toplu Tahmin) hepsi bu seçime göre çalışır.
mod_anahtarlari = list(MOD_AYARLARI.keys())
secilen_etiket = st.radio(
    "Sınıflandırma Modu",
    options=[MOD_AYARLARI[k]["etiket"] for k in mod_anahtarlari],
    horizontal=True,
)
mod = mod_anahtarlari[[MOD_AYARLARI[k]["etiket"] for k in mod_anahtarlari].index(secilen_etiket)]
ayar = MOD_AYARLARI[mod]

try:
    model, skorlar = modeli_yukle(ayar["model_dosya"], ayar["skor_dosya"])
except Exception as e:
    st.error(f"Model dosyaları yüklenemedi: {e}\n\nLütfen önce 'model_egit.py' dosyasını çalıştırın.")
    st.stop()

try:
    df_referans = veriyi_yukle(ayar["veri_dosya"]) if os.path.exists(ayar["veri_dosya"]) else None
except Exception:
    df_referans = None

tab1, tab2, tab3 = st.tabs(["🩺 Tahmin Sistemi", "📊 Veri Analizi ve Performans", "📁 Toplu Tahmin"])

# ==========================================
# SEKME 1: TAHMİN SİSTEMİ
# ==========================================
with tab1:
    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("Hasta Verileri")
        resim_goster("anatomi.jpg", altyazi="Omurga Açıları Referans Görseli")

        st.info("Lütfen hastanın radyolojik ölçümlerini giriniz:")

        p_insidans = st.slider('Pelvik İnsidans', 26.0, 130.0, 60.0)
        p_egim = st.slider('Pelvik Eğim', -6.0, 50.0, 20.0)
        l_lordoz = st.slider('Lumbar Lordoz Açısı', 14.0, 126.0, 50.0)
        s_egim = st.slider('Sakral Eğim', 13.0, 122.0, 40.0)
        p_yaricap = st.slider('Pelvik Yarıçap', 70.0, 164.0, 110.0)
        s_derece = st.slider('Spondilolistezis Derecesi', -11.0, 419.0, 10.0)

        araligi_kontrol_et(df_referans, 'Pelvik_İnsidans', p_insidans, 'Pelvik İnsidans')
        araligi_kontrol_et(df_referans, 'Pelvik_Eğim', p_egim, 'Pelvik Eğim')
        araligi_kontrol_et(df_referans, 'Lumbar_Lordoz_Açısı', l_lordoz, 'Lumbar Lordoz Açısı')
        araligi_kontrol_et(df_referans, 'Sakral_Eğim', s_egim, 'Sakral Eğim')
        araligi_kontrol_et(df_referans, 'Pelvik_Yarıçap', p_yaricap, 'Pelvik Yarıçap')
        araligi_kontrol_et(df_referans, 'Spondilolistezis_Derecesi', s_derece, 'Spondilolistezis Derecesi')

        input_df = pd.DataFrame({
            'Pelvik_İnsidans': [p_insidans],
            'Pelvik_Eğim': [p_egim],
            'Lumbar_Lordoz_Açısı': [l_lordoz],
            'Sakral_Eğim': [s_egim],
            'Pelvik_Yarıçap': [p_yaricap],
            'Spondilolistezis_Derecesi': [s_derece]
        })

    with col_result:
        st.subheader("Analiz Sonucu")

        if st.button("Hastalığı Tahmin Et", type="primary"):
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            durum_ham = prediction[0]
            durum_tr = ayar["turkce"].get(durum_ham, durum_ham)

            # Her ham sınıf etiketi için gösterilecek renk/mesaj burada
            # tanımlı; 3 sınıflı modelde iki farklı "anormal" türü olduğu
            # için ayrı ayrı ele alınıyor.
            if durum_ham == "Normal":
                st.success(f"✅ SONUÇ: {durum_tr}")
                st.write("Hastanın omurga yapısı **Sağlıklı** sınıfında değerlendirilmiştir.")
            elif durum_ham == "Abnormal":
                st.error(f"⚠️ SONUÇ: {durum_tr.upper()} (Riskli)")
                st.write("Hastada **Disk Kayması veya Fıtık** riski tespit edilmiştir. Uzman hekim kontrolü önerilir.")
            elif durum_ham == "Disk_Hernia":
                st.warning(f"⚠️ SONUÇ: {durum_tr}")
                st.write("Hastada **Disk Hernisi (Fıtık)** riski tespit edilmiştir. Uzman hekim kontrolü önerilir.")
            elif durum_ham == "Spondylolisthesis":
                st.error(f"🚨 SONUÇ: {durum_tr}")
                st.write("Hastada **Spondilolistezis (Omur Kayması)** riski tespit edilmiştir. Uzman hekim kontrolü önerilir.")
            else:
                st.info(f"SONUÇ: {durum_tr}")

            st.write("---")
            st.write("**Yapay Zeka Güven Oranı:**")
            probs_df = pd.DataFrame(probability, columns=model.classes_)
            probs_df = probs_df.rename(columns=ayar["turkce"])
            st.bar_chart(probs_df.T)

    st.divider()
    st.subheader("📈 Algoritma Performans Karşılaştırması")
    skor_df = pd.DataFrame(list(skorlar.items()), columns=['Algoritma', 'Başarı Oranı'])
    skor_df = skor_df.set_index('Algoritma')
    st.bar_chart(skor_df)
    st.caption("Bu grafik, eğitim sırasında farklı algoritmaların test verisi üzerindeki başarı oranlarını gösterir.")

# ==========================================
# SEKME 2: VERİ ANALİZİ, KORELASYON VE CONFUSION MATRIX
# ==========================================
with tab2:
    st.header("Veri Seti Analizi ve Model Performansı")

    dosya_yolu = ayar["veri_dosya"]

    if os.path.exists(dosya_yolu):
        df = veriyi_yukle(dosya_yolu)

        # 1. BÖLÜM: GENEL BAKIŞ
        st.subheader("1. Veri Setine Genel Bakış")
        st.write(f"Toplam Kayıt: **{df.shape[0]}** | Özellik Sayısı: **{df.shape[1]}**")
        st.dataframe(df.head(10))
        st.caption("ℹ️ Tabloda veri setinin ilk 10 satırı örnek olarak gösterilmektedir.")

        # 2. BÖLÜM: İSTATİSTİKLER
        st.subheader("2. İstatistiksel Özellikler")
        st.write(df.describe())
        st.caption("ℹ️ **count:** Veri sayısı, **mean:** Ortalama, **std:** Standart sapma, **min-max:** En düşük ve en yüksek değerler.")

        # 3. BÖLÜM: HASTA DAĞILIMI
        st.subheader("3. Hasta Dağılımı")
        col_pie1, col_pie2 = st.columns([1, 2])
        dagilim = df['Durum'].value_counts().rename(index=ayar["turkce"])
        with col_pie1: st.dataframe(dagilim)
        with col_pie2: st.bar_chart(dagilim)
        st.caption("ℹ️ Veri setindeki sınıfların sayısal dağılımı.")

        # 4. BÖLÜM: DEĞİŞKEN İLİŞKİLERİ
        st.subheader("4. Değişken İlişkileri (Scatter Plot)")
        ozellikler = df.columns[:-1].tolist()
        c1, c2 = st.columns(2)
        x_val = c1.selectbox("X Ekseni", ozellikler, index=0)
        y_val = c2.selectbox("Y Ekseni", ozellikler, index=5)
        st.scatter_chart(df, x=x_val, y=y_val, color='Durum', size=20)
        st.caption(f"ℹ️ Grafikte **{x_val}** ile **{y_val}** arasındaki ilişki gösterilmektedir. Renkler hastalık durumunu belirtir.")

        st.divider()

        # 5. BÖLÜM: KORELASYON MATRİSİ
        st.subheader("5. Korelasyon Matrisi (İlişki Analizi)")
        st.markdown("""
        Bu matris, özelliklerin birbirleriyle ne kadar ilişkili olduğunu gösterir.
        *   **+1'e yakın (Kırmızı):** Güçlü Pozitif İlişki.
        *   **-1'e yakın (Mavi):** Güçlü Negatif İlişki.
        """)

        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr_matrix = numeric_df.corr()

        col_corr1, col_corr2 = st.columns([1, 1])

        with col_corr1:
            fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
            st.pyplot(fig_corr)

        with col_corr2:
            st.info("""
            **💡 Analiz İpucu:**
            Matrise dikkatli bakarsanız **Pelvik İnsidans** ile **Sakral Eğim** arasında çok yüksek bir ilişki (Kırmızı renk) görürsünüz.

            Bunun sebebi tıbbi olarak formülün şu olmasıdır:
            `Pelvik İnsidans = Pelvik Eğim + Sakral Eğim`
            """)

        st.divider()

        # 6. BÖLÜM: CONFUSION MATRIX
        st.subheader("6. Karmaşıklık Matrisi (Performans Analizi)")
        st.markdown("Modelin **Tüm Veri Seti** üzerindeki Doğru/Yanlış tahminleri:")

        X_all = df.drop('Durum', axis=1)
        y_all = df['Durum']
        y_pred_all = model.predict(X_all)
        cm = confusion_matrix(y_all, y_pred_all, labels=model.classes_)

        etiket_gorunum = [ayar["turkce"].get(s, s) for s in model.classes_]

        col_cm1, col_cm2 = st.columns([1, 2])

        with col_cm1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etiket_gorunum, yticklabels=etiket_gorunum, ax=ax)
            plt.ylabel('Gerçek Durum')
            plt.xlabel('Modelin Tahmini')
            st.pyplot(fig)

        st.caption("ℹ️ Koyu mavi kutular modelin doğru bildiği hasta sayılarını gösterir.")

        st.divider()

        # 7. BÖLÜM: ÖZELLİK ÖNEMİ (AÇIKLANABİLİRLİK)
        st.subheader("7. Özellik Önemi (Açıklanabilirlik)")
        st.markdown("""
        KNN ve SVM gibi modellerin (Random Forest'ın aksine) doğrudan bir "özellik önemi" değeri yoktur.
        Bunu ölçmek için **Permutation Importance** yöntemi kullanılıyor: bir özelliğin değerleri
        veri setinde rastgele karıştırılıyor, modelin başarısı ne kadar düşerse o özellik tahmin için
        o kadar önemli demektir. Bu yöntem her model türüyle (KNN, SVM, Random Forest) çalışır.
        """)

        with st.spinner("Özellik önemleri hesaplanıyor..."):
            onem = permutation_importance(
                model, X_all, y_all, n_repeats=10, random_state=42, scoring="accuracy")

        onem_df = pd.DataFrame({
            "Özellik": X_all.columns,
            "Önem": onem.importances_mean,
        }).sort_values("Önem", ascending=False).set_index("Özellik")

        col_onem1, col_onem2 = st.columns([2, 1])
        with col_onem1:
            st.bar_chart(onem_df)
            st.caption("ℹ️ Çubuk ne kadar uzunsa, o ölçüm modelin doğru tahmin yapabilmesi için o kadar "
                       "kritik demektir (özellik karıştırıldığında doğruluk o kadar düşüyor).")
        with col_onem2:
            en_onemli = onem_df.index[0]
            st.info(f"""
            **💡 Analiz İpucu:**
            **{en_onemli}**, diğer tüm özelliklerden açık ara daha belirleyici çıkıyor.

            Bu tıbbi olarak da beklenen bir sonuç: bu ölçüm, adından da (Spondilolistezis = omur
            kayması) anlaşılacağı gibi doğrudan hastalığın kendisini tanımlıyor.
            """)

    else:
        st.error(f"'{dosya_yolu}' dosyası bulunamadı! Lütfen CSV dosyasını klasöre atın.")

# ==========================================
# SEKME 3: TOPLU TAHMİN (CSV YÜKLEME)
# ==========================================
with tab3:
    st.header("Toplu Tahmin (CSV Yükleme)")
    st.write("Birden fazla hastanın ölçümlerini içeren bir CSV dosyası yükleyin; her satır için tahmin üretilir.")
    st.markdown("**Dosyada şu 6 sütun bulunmalıdır** (Türkçe başlıklar veya orijinal veri setinin İngilizce başlıkları kabul edilir):")
    st.code(", ".join(OZELLIKLER), language=None)
    st.caption("Ek sütunlar (ör. hasta numarası) silinmez, sonuç tablosunda korunur. "
               "Virgül veya noktalı virgülle ayrılmış dosyalar okunabilir.")

    if df_referans is not None:
        ornek_csv = df_referans.drop(columns="Durum").sample(5)
        st.download_button(
            "📄 Örnek CSV şablonunu indir",
            data=ornek_csv.to_csv(index=False).encode("utf-8-sig"),
            file_name="ornek_hasta_verisi.csv",
            mime="text/csv",
        )

    yuklenen = st.file_uploader("CSV dosyası seçin", type=["csv"])

    if yuklenen is not None:
        try:
            df_ham = csv_oku(yuklenen.getvalue())
            X_toplu, atilan = ozellikleri_hazirla(df_ham)
        except CSVHatasi as hata:
            st.error(f"❌ {hata}")
        else:
            sonuc = tahmin_et(model, X_toplu, df_ham, etiket_haritasi=ayar["turkce"])

            if atilan:
                gosterilen = ", ".join(str(n) for n in atilan[:10])
                fazla = f" ve {len(atilan) - 10} satır daha" if len(atilan) > 10 else ""
                st.warning(f"⚠️ {len(atilan)} satır eksik veya sayısal olmayan değer içerdiği için atlandı "
                           f"(CSV satır no: {gosterilen}{fazla}).")

            sayilar = sonuc["Tahmin"].value_counts()
            kolonlar = st.columns(len(sayilar) + 1)
            kolonlar[0].metric("Toplam Hasta", len(sonuc))
            for kolon, (sinif_adi, adet) in zip(kolonlar[1:], sayilar.items()):
                kolon.metric(sinif_adi, int(adet))

            st.dataframe(sonuc, hide_index=True)
            st.download_button(
                "⬇️ Sonuçları CSV olarak indir",
                data=sonuc.to_csv(index=False).encode("utf-8-sig"),
                file_name="toplu_tahmin_sonuclari.csv",
                mime="text/csv",
                type="primary",
            )
            st.caption("Bu sonuçlar karar destek amaçlıdır, tanı yerine geçmez. Uzman hekim kontrolü önerilir.")
