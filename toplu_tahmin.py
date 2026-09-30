# Dosya adı: toplu_tahmin.py
# Toplu (CSV ile) tahmin için yardımcı fonksiyonlar.
# Streamlit'e bağımlı değildir; bu sayede pytest ile kolayca test edilebilir.
import csv
import io

import numpy as np
import pandas as pd

# Modelin eğitildiği sütun adları (sıra önemli)
OZELLIKLER = [
    'Pelvik_İnsidans',
    'Pelvik_Eğim',
    'Lumbar_Lordoz_Açısı',
    'Sakral_Eğim',
    'Pelvik_Yarıçap',
    'Spondilolistezis_Derecesi',
]

# Orijinal veri setindeki (column_2C.csv) İngilizce başlıklar da kabul edilir
INGILIZCE_ESLESME = {
    'pelvic_incidence': 'Pelvik_İnsidans',
    'pelvic_tilt': 'Pelvik_Eğim',
    'lumbar_lordosis_angle': 'Lumbar_Lordoz_Açısı',
    'sacral_slope': 'Sakral_Eğim',
    'pelvic_radius': 'Pelvik_Yarıçap',
    'degree_spondylolisthesis': 'Spondilolistezis_Derecesi',
}

# Ücretsiz sunucuyu korumak için üst sınır
MAKS_SATIR = 10000


class CSVHatasi(Exception):
    """Yüklenen dosya kullanılamıyorsa kullanıcıya gösterilecek mesajı taşır."""


def csv_oku(ham_bayt):
    """Yüklenen dosyanın baytlarını DataFrame'e çevirir.

    - UTF-8 (BOM'lu olabilir) ve Windows-1254 (Türkçe Excel) kodlamalarını dener
    - Ayırıcıyı (virgül / noktalı virgül / sekme) kendisi tahmin eder
    """
    metin = None
    for kodlama in ('utf-8-sig', 'cp1254'):
        try:
            metin = ham_bayt.decode(kodlama)
            break
        except UnicodeDecodeError:
            continue
    if metin is None:
        raise CSVHatasi("Dosyanın karakter kodlaması okunamadı. "
                        "CSV'yi UTF-8 olarak kaydedip tekrar deneyin.")

    try:
        df = pd.read_csv(io.StringIO(metin), sep=None, engine='python')
    except (pd.errors.EmptyDataError, pd.errors.ParserError, csv.Error):
        raise CSVHatasi("Dosya CSV olarak okunamadı. Sütunların virgül veya "
                        "noktalı virgülle ayrıldığından emin olun.")

    if df.empty:
        raise CSVHatasi("Dosyada hiç veri satırı yok.")
    return df


def ozellikleri_hazirla(df):
    """Yüklenen tablodan modelin beklediği 6 sütunu çıkarır.

    Döndürür: (X, atilan_satirlar)
      X               -> sayısal, eksiksiz satırlardan oluşan DataFrame
      atilan_satirlar -> geçersiz/eksik değer yüzünden atlanan satırların
                         CSV'deki numaraları (başlık satırı 1 sayılır)
    """
    df = df.copy()
    df.columns = [INGILIZCE_ESLESME.get(str(k).strip().lower(), str(k).strip())
                  for k in df.columns]

    eksik = [k for k in OZELLIKLER if k not in df.columns]
    if eksik:
        raise CSVHatasi("Şu sütun(lar) dosyada bulunamadı: " + ", ".join(eksik))

    if len(df) > MAKS_SATIR:
        raise CSVHatasi(f"Dosya çok büyük: en fazla {MAKS_SATIR} satır yüklenebilir "
                        f"(dosyada {len(df)} satır var).")

    # Ondalık virgülü ("63,02") de kabul et, sayıya çevrilemeyenleri NaN yap
    X = df[OZELLIKLER].apply(
        lambda s: pd.to_numeric(s.astype(str).str.replace(',', '.', regex=False),
                                errors='coerce'))
    X = X.replace([np.inf, -np.inf], np.nan)

    gecersiz = X.isna().any(axis=1)
    atilan_satirlar = (df.index[gecersiz] + 2).tolist()
    X = X[~gecersiz]

    if X.empty:
        raise CSVHatasi("Geçerli sayısal değer içeren hiç satır bulunamadı.")
    return X, atilan_satirlar


def tahmin_et(model, X, kaynak_df, etiket_haritasi=None):
    """Her geçerli satır için tahmin ve güven oranı üretir.

    Kullanıcının dosyasındaki tüm orijinal sütunlar (ör. hasta numarası)
    korunur; sağa 'Tahmin' ve 'Güven Oranı (%)' sütunları eklenir.

    etiket_haritasi: modelin ürettiği ham sınıf adını (ör. 'Abnormal',
    'Disk_Hernia') ekranda gösterilecek Türkçe adla eşleştirir. Hem 2
    sınıflı hem 3 sınıflı model için kullanılabilir.
    """
    tahmin = model.predict(X)
    guven = model.predict_proba(X).max(axis=1) * 100

    sonuc = kaynak_df.loc[X.index].copy()
    tahmin_serisi = pd.Series(tahmin, index=X.index)
    if etiket_haritasi:
        tahmin_serisi = tahmin_serisi.replace(etiket_haritasi)
    sonuc['Tahmin'] = tahmin_serisi
    sonuc['Güven Oranı (%)'] = np.round(guven, 1)
    return sonuc
