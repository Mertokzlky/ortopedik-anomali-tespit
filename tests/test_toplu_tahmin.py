# Dosya adı: tests/test_toplu_tahmin.py
# toplu_tahmin.py'deki CSV okuma / doğrulama / tahmin fonksiyonlarının testleri.
import pandas as pd
import pytest

from toplu_tahmin import (CSVHatasi, OZELLIKLER, csv_oku, ozellikleri_hazirla,
                           tahmin_et)


# ---------- csv_oku ----------

def test_csv_oku_normal_utf8():
    ham = "a,b\n1,2\n3,4\n".encode("utf-8")
    df = csv_oku(ham)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_csv_oku_turkce_excel_noktali_virgul_cp1254():
    # Türkçe Excel genelde noktalı virgülle ayırır ve cp1254 ile kaydeder
    ham = "Pelvik_İnsidans;Pelvik_Eğim\n63,02;22,55\n".encode("cp1254")
    df = csv_oku(ham)
    assert df.shape == (1, 2)


def test_csv_oku_bos_dosya_hata_verir():
    with pytest.raises(CSVHatasi):
        csv_oku(b"")


def test_csv_oku_okunamayan_kodlama_hata_verir():
    with pytest.raises(CSVHatasi):
        csv_oku(bytes(range(256)) * 3)


# ---------- ozellikleri_hazirla ----------

def ornek_df(satir_sayisi=3):
    return pd.DataFrame({
        "pelvic_incidence": [60.0] * satir_sayisi,
        "pelvic_tilt": [20.0] * satir_sayisi,
        "lumbar_lordosis_angle": [50.0] * satir_sayisi,
        "sacral_slope": [40.0] * satir_sayisi,
        "pelvic_radius": [110.0] * satir_sayisi,
        "degree_spondylolisthesis": [10.0] * satir_sayisi,
    })


def test_ozellikleri_hazirla_ingilizce_basliklari_tanir():
    X, atilan = ozellikleri_hazirla(ornek_df())
    assert list(X.columns) == OZELLIKLER
    assert atilan == []
    assert len(X) == 3


def test_ozellikleri_hazirla_eksik_sutun_hata_verir():
    df = ornek_df().drop(columns=["pelvic_radius"])
    with pytest.raises(CSVHatasi):
        ozellikleri_hazirla(df)


def test_ozellikleri_hazirla_bozuk_satirlari_atlar():
    df = ornek_df(3).astype(object)
    df.iloc[1, 0] = "abc"  # sayısal olmayan değer
    X, atilan = ozellikleri_hazirla(df)
    assert len(X) == 2
    assert atilan == [3]  # CSV'de 2. satır (başlık=1) atlandı


def test_ozellikleri_hazirla_hepsi_bozuksa_hata_verir():
    df = pd.DataFrame({k: ["x", "y"] for k in
                        ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
                         "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]})
    with pytest.raises(CSVHatasi):
        ozellikleri_hazirla(df)


def test_ozellikleri_hazirla_ondalik_virgulu_destekler():
    df = ornek_df(1).astype(object)
    df.iloc[0, 0] = "63,02"
    X, atilan = ozellikleri_hazirla(df)
    assert atilan == []
    assert X.iloc[0]["Pelvik_İnsidans"] == pytest.approx(63.02)


# ---------- tahmin_et ----------

def test_tahmin_et_etiket_haritasi_uygular(model_2sinif, veri_2sinif):
    X = veri_2sinif[OZELLIKLER].iloc[:5]
    sonuc = tahmin_et(model_2sinif, X, veri_2sinif, etiket_haritasi={"Abnormal": "Anormal"})
    assert "Tahmin" in sonuc.columns
    assert "Güven Oranı (%)" in sonuc.columns
    assert set(sonuc["Tahmin"].unique()) <= {"Normal", "Anormal"}
    assert len(sonuc) == 5


def test_tahmin_et_kaynak_sutunlarini_korur(model_2sinif, veri_2sinif):
    df = veri_2sinif.iloc[:3].copy()
    df.insert(0, "Hasta No", ["H1", "H2", "H3"])
    X = df[OZELLIKLER]
    sonuc = tahmin_et(model_2sinif, X, df)
    assert list(sonuc["Hasta No"]) == ["H1", "H2", "H3"]
