"""Các kiểm tra theo góp ý G5 (17/10/2026) của thầy, hạn Thứ Tư 21/10.

Cách chạy (đặt cùng thư mục với g4_kiem_tra.py, g3_kiem_tra_bo_sung.py, r4_thanh_phan.py, r5_v2_phan_loai_clear.py;
chạy ở thư mục có sẵn ket_qua_g3/, ket_qua_r4/, ket_qua_g4/ để loại các case đã đọc):
    python src/g5_kiem_tra.py --tu-kiem        # chạy thử trên vết giả lập
    python src/g5_kiem_tra.py

Kết quả ghi vào ket_qua_g5/ (đổi bằng --ra); tom_tat_g5.txt là bản tóm tắt, gửi lại file này.
Mục trong G5 được ghi ở đầu mỗi phần: 2.3 (trạng thái hoàn tất), 3 (M / N1 / N2), 5 (đối chiếu số liệu).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import g3_kiem_tra_bo_sung as g3
import g4_kiem_tra as g4
import r5_v2_phan_loai_clear as r5
from g4_kiem_tra import DELTA, SEED, VENDOR_INV, dem_r5, pct, r5_v22
from r5_v2_phan_loai_clear import CLEAR, IR, MEMO, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear, sap_xep
import cau_hinh

THU_MUC_DA_DOC = ["ket_qua_g3/mau_kiem_chung", "ket_qua_r4/mau_kiem_chung", "ket_qua_g4/mau_kiem_chung"]


def don_mua_hang(case):
    """Mã đơn mua hàng (Purchasing Document) = phần trước dấu '_' cuối cùng của mã dòng hàng."""
    return pd.Series(case, index=case).astype(str).str.rsplit("_", n=1).str[0]


def so_lieu_case(ev, pv, loai):
    """P: số thời điểm thanh toán khác nhau; I: max(0, #IR − #Cancel IR); V: số Vendor creates invoice; M: số debit memo."""
    d = r5.dem_theo_case(ev, pv, loai)
    tt = loai.index[loai == "thanh_toan"]
    d["P"] = ev.loc[tt].drop_duplicates(["case", "ts"]).groupby("case").size().reindex(d.index).fillna(0).astype(int)
    d["I"] = (d["n_ir"] - d["n_cancel"]).clip(lower=0)
    d["V"] = ev[ev["act"] == VENDOR_INV].groupby("case").size().reindex(d.index).fillna(0).astype(int)
    d["M"] = d["n_memo"]
    d["po"] = don_mua_hang(d.index)
    return d


def trang_thai_cac_dinh_nghia(d):
    """cu: định nghĩa hiện tại (số Clear thanh toán ≥ #IR − #Cancel IR + #memo);
    moi: G5 2.3 (số thời điểm thanh toán ≥ max(0, #IR − #Cancel IR), không đòi debit memo);
    moi_su_kien: như moi nhưng đếm sự kiện Clear (để xem riêng ảnh hưởng của việc đếm theo thời điểm)."""
    hd = d["n_ir"] - d["n_cancel"]
    return {
        "cu": r5.trang_thai(d, d["n_clear_tt"], hd + d["n_memo"], d["cuoi_clear_tt"]),
        "moi": r5.trang_thai(d, d["P"], d["I"], d["cuoi_clear_tt"]),
        "moi_su_kien": r5.trang_thai(d, d["n_clear_tt"], d["I"], d["cuoi_clear_tt"]),
        "tho": r5.trang_thai(d, d["n_clear"], hd, d["cuoi_clear"]),  # trước khi phân loại Clear (Bảng 3)
    }


def km_trung_vi(t, co_su_kien):
    """Trung vị Kaplan-Meier (ngày). t: thời gian; co_su_kien: 1 nếu hoàn tất, 0 nếu bị cắt."""
    x = pd.DataFrame({"t": t, "e": co_su_kien}).dropna()
    g = x.groupby("t")["e"].agg(["sum", "count"]).sort_index()
    con = len(x) - g["count"].cumsum().shift(fill_value=0)
    s = (1 - g["sum"] / con).cumprod()
    duoi = s[s <= 0.5]
    return float(duoi.index[0]) if len(duoi) else float("nan")


def thoi_gian(d, tt, moc):
    """Trung vị, P90 trên dòng hàng hoàn tất và trung vị Kaplan-Meier (còn mở bị cắt tại thời điểm trích xuất)."""
    ht = tt == "Hoàn tất"
    mo = tt == "Còn mở"
    ngay_ht = (d["cuoi_clear_tt"] - d["dau_po"]).dt.total_seconds() / 86400
    ngay_mo = (pd.Timestamp(moc) - d["dau_po"]).dt.total_seconds() / 86400
    t = pd.concat([ngay_ht[ht], ngay_mo[mo]])
    e = pd.concat([pd.Series(1, index=ngay_ht[ht].index), pd.Series(0, index=ngay_mo[mo].index)])
    return dict(n_ht=int(ht.sum()), n_mo=int(mo.sum()), trung_vi=ngay_ht[ht].median(), p90=ngay_ht[ht].quantile(0.9),
                km=km_trung_vi(t.values, e.values))


def phan_loai_mn(d, s22):
    """G5 Mục 3, mã giả P.1. Dòng hàng trong cùng đơn có #IR > #Vendor creates invoice là 'dư ghi nhận'."""
    du = d.loc[d["n_ir"] > d["V"]]
    du_theo_don = du.groupby("po").size()
    x = d.loc[sorted(s22), ["po", "P", "I", "V", "M", "n_ir", "n_cancel"]].copy()
    tu_du = (x["n_ir"] > x["V"]).astype(int)  # bản thân dòng này có dư ghi nhận thì không tính là "dòng khác"
    dong_khac = x["po"].map(du_theo_don).fillna(0).astype(int) - tu_du
    so_dong_don = d.groupby("po").size()
    x["so_dong_trong_don"] = x["po"].map(so_dong_don)
    x["dong_khac_du_ghi_nhan"] = dong_khac
    x["loai"] = np.select(
        [x["P"] > x["V"] + x["M"], x["P"] <= x["I"] + x["M"], dong_khac > 0],
        ["M", "khác (P ≤ I + M khi xét cả dòng hàng)", "N1"], "N2")
    return x


def doc_case_da_doc():
    da_doc = set(g3.DA_DOC)
    for th in THU_MUC_DA_DOC:
        for f in Path(th).glob("*.csv"):
            try:
                da_doc |= set(pd.read_csv(f, encoding="utf-8-sig", usecols=["case"])["case"].astype(str))
            except Exception:
                pass
    return da_doc


# ---------------------------------------------------------------- phân tích
def phan_tich(ev, pv, moc, ra, p, rng):
    N = len(pv)
    loai, _, _ = phan_loai_clear(ev, DELTA)
    ev = ev.copy()
    ev["loai_clear"] = loai.reindex(ev.index)
    d = so_lieu_case(ev, pv, loai)
    p(f"Phạm vi {N:,} dòng hàng. Δ = {DELTA} phút. Thời điểm trích xuất: {moc}")

    # ======================= Mục 2.3. Trạng thái hoàn tất không đòi debit memo
    p("\n=== G5 Mục 2.3. Trạng thái hoàn tất không đòi debit memo ===")
    st = trang_thai_cac_dinh_nghia(d)
    for k, ten in [("cu", "Định nghĩa hiện tại (đếm sự kiện Clear, tính debit memo)"),
                   ("moi", "Định nghĩa mới G5 (đếm thời điểm thanh toán, không đòi debit memo)"),
                   ("moi_su_kien", "Như định nghĩa mới nhưng đếm sự kiện Clear")]:
        vc = st[k].value_counts()
        p(f"  {ten}: hoàn tất {pct(int(vc.get('Hoàn tất', 0)), N)}; còn mở {pct(int(vc.get('Còn mở', 0)), N)}; "
          f"bị xóa {pct(int(vc.get('Bị xóa', 0)), N)}")
    bang = pd.crosstab(st["cu"].rename("hiện tại"), st["moi"].rename("mới G5"))
    p("Bảng chéo trạng thái (hàng: hiện tại, cột: mới G5):\n" + bang.to_string())
    doi = st["cu"] != st["moi"]
    p(f"Số dòng hàng đổi trạng thái: {pct(int(doi.sum()), N)}")
    mo_sang_ht = (st["cu"] == "Còn mở") & (st["moi"] == "Hoàn tất")
    ht_sang_mo = (st["cu"] == "Hoàn tất") & (st["moi"] == "Còn mở")
    p(f"  Còn mở → hoàn tất: {int(mo_sang_ht.sum()):,}; trong đó có ≥ 1 debit memo: {int((mo_sang_ht & (d['M'] > 0)).sum()):,}")
    p(f"  Hoàn tất → còn mở: {int(ht_sang_mo.sum()):,} (do đếm theo thời điểm: nhiều Clear cùng timestamp tính là một lần trả)")
    p(f"  Riêng ảnh hưởng của đếm theo thời điểm (mới G5 so với mới đếm sự kiện): {int((st['moi'] != st['moi_su_kien']).sum()):,} dòng hàng khác trạng thái")
    hai096 = (st["tho"] == "Hoàn tất") & (st["cu"] == "Còn mở")
    p(f"  Trong {int(hai096.sum()):,} dòng hàng hoàn tất → còn mở khi phân loại Clear (Mục 5.2): theo định nghĩa mới, "
      f"hoàn tất trở lại {int((hai096 & (st['moi'] == 'Hoàn tất')).sum()):,}, vẫn còn mở {int((hai096 & (st['moi'] == 'Còn mở')).sum()):,}")
    p("Thời gian thanh toán (Create Purchase Order Item → Clear thanh toán cuối), ngày:")
    rows = []
    for k, ten in [("tho", "trước phân loại Clear (Bảng 3)"), ("cu", "hiện tại"), ("moi", "mới G5"), ("moi_su_kien", "mới, đếm sự kiện")]:
        r_ = thoi_gian(d, st[k], moc)
        rows.append({"dinh_nghia": ten, **r_})
        so = lambda v, k=1: f"{v:,.{k}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        p(f"  {ten}: hoàn tất {so(r_['n_ht'], 0)}, còn mở {so(r_['n_mo'], 0)}; trung vị (hoàn tất) {so(r_['trung_vi'])}; "
          f"P90 {so(r_['p90'])}; trung vị Kaplan-Meier {so(r_['km'])}")
    pd.DataFrame(rows).to_csv(ra / "g5_2_thoi_gian_theo_dinh_nghia.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(st)[doi].to_csv(ra / "g5_2_dong_hang_doi_trang_thai.csv", encoding="utf-8-sig")

    # ======================= Mục 3. Phân loại M / N1 / N2
    p("\n=== G5 Mục 3. Phân loại R5 v2.2 theo chứng từ nhà cung cấp và đơn mua hàng ===")
    bd = dem_r5(ev, loai)
    s22, nhanh, tra_truoc = r5_v22(bd)
    x = phan_loai_mn(d, s22)
    x["nhanh"] = nhanh.reindex(x.index)
    x["nhom"] = np.select([(x["n_cancel"] > 0) & (x["M"] > 0), (x["n_cancel"] == 0) & (x["M"] == 0), x["n_cancel"] == 0],
                          ["3. có hủy và debit memo", "4. không hủy, không debit memo", "5. có debit memo, không hủy"],
                          "6. có hủy, không debit memo")
    n22 = len(x)
    p(f"R5 v2.2: {n22:,} dòng hàng")
    for k in ["M", "N1", "N2", "khác (P ≤ I + M khi xét cả dòng hàng)"]:
        p(f"  {k}: {pct(int((x['loai'] == k).sum()), n22)}")
    mot_dong = x["so_dong_trong_don"] == 1
    p(f"  Dòng hàng R5 thuộc đơn mua hàng chỉ có một dòng (kiểm tra cấp đơn không áp dụng, mặc định N2 nếu không phải M): "
      f"{pct(int(mot_dong.sum()), n22)}; trong N2: {int((mot_dong & (x['loai'] == 'N2')).sum()):,}")
    p("Bảng chéo loại × nhóm tầng (G4):\n" + pd.crosstab(x["loai"], x["nhom"], margins=True, margins_name="Tổng").to_string())
    p("Bảng chéo loại × nhánh:\n" + pd.crosstab(x["loai"], x["nhanh"], margins=True, margins_name="Tổng").to_string())
    x.to_csv(ra / "g5_3_r5_phan_loai.csv", encoding="utf-8-sig")

    # đối chiếu với 37 case đã đọc ở G4
    f = Path("ket_qua_g4/mau_kiem_chung/r5v22_theo_tang_nhan_xet.csv")
    if f.exists():
        nx = pd.read_csv(f, encoding="utf-8-sig").set_index("case")
        nx["loai"] = x["loai"].reindex(nx.index)
        p("Đối chiếu 37 case đọc ở G4 (kết luận đọc tay × loại):\n"
          + pd.crosstab(nx["ket_luan"], nx["loai"].fillna("không còn trong R5"), margins=True, margins_name="Tổng").to_string())

    # mẫu kiểm chứng mới: 10 case N2 và toàn bộ nhóm M (tối đa 10)
    da_doc = doc_case_da_doc()
    p(f"\nMẫu kiểm chứng (đã loại {len(da_doc):,} case đọc trước đây):")
    phan = []
    for k in ["M", "N2"]:
        ds = sorted(set(x.index[x["loai"] == k]) - da_doc)
        chon = list(rng.choice(ds, size=min(10, len(ds)), replace=False)) if ds else []
        p(f"  {k}: {len(ds):,} case đủ điều kiện, lấy {len(chon)}")
        phan.append(pd.DataFrame({"case": chon, "loai": k}))
    chon = pd.concat(phan)
    vet = ev[ev["case"].isin(chon["case"])][["case", "ts", "act", "loai_clear", "user"]].merge(chon, on="case")
    vet = vet.merge(x[["P", "I", "V", "M", "so_dong_trong_don"]], left_on="case", right_index=True)
    vet["ket_qua"] = ""  # đúng / báo nhầm do dữ liệu / không xác định
    vet["nhan_xet"] = ""
    vet.to_csv(ra / "mau_kiem_chung" / "r5_m_n2.csv", index=False, encoding="utf-8-sig")
    # các dòng hàng khác trong cùng đơn, để đọc ngữ cảnh
    cung_don = d.loc[d["po"].isin(x.loc[chon["case"], "po"]), ["po", "P", "I", "V", "M", "n_ir", "n_cancel"]]
    cung_don.to_csv(ra / "mau_kiem_chung" / "r5_m_n2_cung_don.csv", encoding="utf-8-sig")

    # ======================= Mục 5. Đối chiếu số liệu
    p("\n=== G5 Mục 5. Đối chiếu 744 / 681 / 126 / 63 ===")
    stg = set(bd.loc[bd["vuot_tg"], "case"])
    p(f"Đếm theo thời điểm (bước trung gian): {len(stg):,}; R5 v2.2: {len(s22):,}; chênh {len(stg - s22):,}")
    p(f"Chỉ báo 'trả trước khi ghi nhận hóa đơn' (dòng hàng có ít nhất một lần vượt mà sau đó có Record Invoice Receipt): {len(tra_truoc):,}")
    p(f"  trong đó vẫn còn lần vượt khác nên vẫn ở R5 v2.2: {len(tra_truoc & s22):,}; rời khỏi R5: {len(tra_truoc - s22):,}")
    p(f"  Kiểm tra: mọi dòng hàng rời khỏi R5 (744 → 681) đều thuộc chỉ báo này: {(stg - s22) <= tra_truoc}")
    s22_km, _, _ = r5_v22(dem_r5(ev, loai, tinh_memo=False))
    p(f"R5 v2.2 dạng khoảng: {pct(len(s22), N)} (tính debit memo) đến {pct(len(s22_km), N)} (không tính debit memo)")
    r5a = int((nhanh == "R5a").sum())
    p(f"R5a: {pct(r5a, N)} (con số cuối của chuỗi R4: 2.562 → 1.566 → R5a)")
    for k, n in [("M", int((x['loai'] == 'M').sum())), ("N2", int((x['loai'] == 'N2').sum()))]:
        p(f"  {k} trên toàn phạm vi: {pct(n, N)}")


# ---------------------------------------------------------------- tự kiểm
def tu_kiem():
    t = pd.Timestamp("2018-03-01 08:00")
    m, ngay = pd.Timedelta(minutes=1), pd.Timedelta(days=1)
    PO, GR = r5.PO, r5.GR
    vet = {
        # vượt cả chứng từ phía nhà cung cấp: P = 2 > V + M = 1 → M
        "4500000001_00010": [(PO, 0 * m), (GR, ngay), (VENDOR_INV, ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay), (CLEAR, 40 * ngay)],
        # NCC gửi 2 hóa đơn, dòng này ghi nhận 1; dòng 00020 cùng đơn ghi nhận dư (#IR > V) → N1
        "4500000002_00010": [(PO, 0 * m), (GR, ngay), (VENDOR_INV, ngay), (VENDOR_INV, 2 * ngay), (IR, 2 * ngay),
                             (CLEAR, 30 * ngay), (CLEAR, 40 * ngay)],
        "4500000002_00020": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay)],
        # như trên nhưng đơn chỉ có một dòng → N2
        "4500000003_00010": [(PO, 0 * m), (GR, ngay), (VENDOR_INV, ngay), (VENDOR_INV, 2 * ngay), (IR, 2 * ngay),
                             (CLEAR, 30 * ngay), (CLEAR, 40 * ngay)],
        # debit memo không được trả riêng: hiện tại còn mở, mới hoàn tất
        "4500000004_00010": [(PO, 0 * m), (GR, ngay), (MEMO, ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay)],
        # hai hóa đơn trả cùng một lúc: đếm sự kiện hoàn tất, đếm thời điểm còn mở
        "4500000005_00010": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (IR, 3 * ngay), (CLEAR, 30 * ngay), (CLEAR, 30 * ngay)],
    }
    df = pd.DataFrame([{"case": c, "act": a, "ts": t + dt, "user": "user_1"} for c, ds in vet.items() for a, dt in ds])
    ev = sap_xep(df)
    loai, _, _ = phan_loai_clear(ev, DELTA)
    d = so_lieu_case(ev, set(ev["case"]), loai)
    s22, _, _ = r5_v22(dem_r5(ev, loai))
    x = phan_loai_mn(d, s22)
    st = trang_thai_cac_dinh_nghia(d)
    kq = {
        "R5 v2.2": sorted(s22),
        "M": sorted(x.index[x["loai"] == "M"]),
        "N1": sorted(x.index[x["loai"] == "N1"]),
        "N2": sorted(x.index[x["loai"] == "N2"]),
        "còn mở (hiện tại)": sorted(st["cu"].index[st["cu"] == "Còn mở"]),
        "còn mở (mới G5)": sorted(st["moi"].index[st["moi"] == "Còn mở"]),
        "còn mở (mới, đếm sự kiện)": sorted(st["moi_su_kien"].index[st["moi_su_kien"] == "Còn mở"]),
    }
    ky_vong = {
        "R5 v2.2": ["4500000001_00010", "4500000002_00010", "4500000003_00010"],
        "M": ["4500000001_00010"],
        "N1": ["4500000002_00010"],
        "N2": ["4500000003_00010"],
        "còn mở (hiện tại)": ["4500000004_00010"],
        "còn mở (mới G5)": ["4500000005_00010"],
        "còn mở (mới, đếm sự kiện)": [],
    }
    loi = 0
    for k in kq:
        dat = kq[k] == ky_vong[k]
        loi += not dat
        print(f"{k:28s} {kq[k]}  {'ĐẠT' if dat else 'SAI, kỳ vọng ' + str(ky_vong[k])}")
    km = km_trung_vi(np.array([1, 2, 3, 4, 5.0]), np.array([1, 1, 0, 1, 1]))
    dat = km == 4.0
    loi += not dat
    print(f"{'trung vị Kaplan-Meier':28s} {km}  {'ĐẠT' if dat else 'SAI, kỳ vọng 4.0'}")
    print("\nTất cả đều đạt." if loi == 0 else f"\n{loi} kiểm tra SAI.")
    return loi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(cau_hinh.LOG), help="đường dẫn log; mặc định trong cau_hinh.py")
    ap.add_argument("--cong-ty", default="companyID_0000")
    ap.add_argument("--ra", default="ket_qua_g5")
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        sys.exit(tu_kiem())
    if not a.log:
        ap.error("cần đường dẫn log hoặc --tu-kiem")
    if tu_kiem():
        sys.exit("Kiểm thử giả lập không đạt, dừng.")
    ra = Path(a.ra)
    (ra / "mau_kiem_chung").mkdir(parents=True, exist_ok=True)
    out = []

    def p(*x):
        s = " ".join(str(i) for i in x)
        print(s)
        out.append(s)

    raw = g3.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    phan_tich(ev, pv, moc, ra, p, np.random.default_rng(SEED))
    (ra / "tom_tat_g5.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_g5.txt.")


if __name__ == "__main__":
    main()
