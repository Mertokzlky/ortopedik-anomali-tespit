# Dosya adı: tests/test_model.py
# Eğitilmiş modellerin temel "sağlık kontrolleri".
# Amaç: model_egit.py'de yapılacak bir değişikliğin modeli yanlışlıkla
# bozup bozmadığını (ör. sütun sırasının karışması, yanlış veri verilmesi)
# erken yakalamak. Bilimsel bir doğrulama değil, bir "duman testi"dir.
import pandas as pd
import pytest

OZELLIKLER = [
    'Pelvik_İnsidans',
    'Pelvik_Eğim',
    'Lumbar_Lordoz_Açısı',
    'Sakral_Eğim',
    'Pelvik_Yarıçap',
    'Spondilolistezis_Derecesi',
]


def hasta(pelvik_insidans, pelvik_egim, lumbar_lordoz, sakral_egim, pelvik_yaricap, spondilolistezis):
    return pd.DataFrame([[pelvik_insidans, pelvik_egim, lumbar_lordoz,
                           sakral_egim, pelvik_yaricap, spondilolistezis]],
                         columns=OZELLIKLER)


# ---------- 2 SINIFLI MODEL ----------

def test_2sinif_model_dogru_siniflari_biliyor(model_2sinif):
    assert set(model_2sinif.classes_) == {"Normal", "Abnormal"}


def test_2sinif_bilinen_anormal_hasta(model_2sinif):
    # column_2C.csv satır 9 — model bu hastayı %100 güvenle doğru biliyor
    x = hasta(36.6864, 5.0109, 41.9488, 31.6755, 84.2414, 0.6644)
    assert model_2sinif.predict(x)[0] == "Abnormal"


def test_2sinif_bilinen_normal_hasta(model_2sinif):
    # column_2C.csv satır 212 — model bu hastayı %100 güvenle doğru biliyor
    x = hasta(44.3625, 8.9454, 46.9021, 35.4171, 129.2207, 4.9942)
    assert model_2sinif.predict(x)[0] == "Normal"


def test_2sinif_genel_dogruluk_makul_seviyede(model_2sinif, veri_2sinif):
    # Veri setinin tamamında en azından %70 doğruluk bekleniyor (eğitimde ~%84 ölçülmüştü).
    # Bu bir regresyon alarmı: skor aniden çok düşerse bir şey bozulmuş demektir.
    X = veri_2sinif[OZELLIKLER]
    y = veri_2sinif["Durum"]
    dogruluk = (model_2sinif.predict(X) == y).mean()
    assert dogruluk > 0.70


# ---------- 3 SINIFLI MODEL ----------

def test_3sinif_model_dogru_siniflari_biliyor(model_3sinif):
    assert set(model_3sinif.classes_) == {"Normal", "Disk_Hernia", "Spondylolisthesis"}


def test_3sinif_bilinen_normal_hasta(model_3sinif):
    # column_3C.csv satır 257
    x = hasta(50.1601, -2.97, 42.0, 53.1301, 131.8025, -8.2902)
    assert model_3sinif.predict(x)[0] == "Normal"


def test_3sinif_bilinen_disk_hernisi_hasta(model_3sinif):
    # column_3C.csv satır 24
    x = hasta(36.1257, 22.7588, 29.0, 13.3669, 115.5771, -3.2376)
    assert model_3sinif.predict(x)[0] == "Disk_Hernia"


def test_3sinif_bilinen_spondilolistezis_hasta(model_3sinif):
    # column_3C.csv satır 67 — en belirleyici özellik olan Spondilolistezis
    # Derecesi burada çok yüksek (69.55), bu yüzden güvenle test edilebilir.
    x = hasta(75.6497, 19.3398, 64.1487, 56.3099, 95.9036, 69.5513)
    assert model_3sinif.predict(x)[0] == "Spondylolisthesis"


def test_3sinif_genel_dogruluk_makul_seviyede(model_3sinif, veri_3sinif):
    X = veri_3sinif[OZELLIKLER]
    y = veri_3sinif["Durum"]
    dogruluk = (model_3sinif.predict(X) == y).mean()
    assert dogruluk > 0.65


# ---------- ORTAK DAVRANIŞ ----------

@pytest.mark.parametrize("model_adi", ["model_2sinif", "model_3sinif"])
def test_model_olasilik_tahmini_destekler(model_adi, request):
    # app.py güven oranı göstermek için predict_proba kullanıyor;
    # bu her iki modelde de çalışmalı.
    model = request.getfixturevalue(model_adi)
    x = hasta(60.0, 20.0, 50.0, 40.0, 110.0, 10.0)
    olasiliklar = model.predict_proba(x)[0]
    assert olasiliklar.sum() == pytest.approx(1.0)
    assert len(olasiliklar) == len(model.classes_)
