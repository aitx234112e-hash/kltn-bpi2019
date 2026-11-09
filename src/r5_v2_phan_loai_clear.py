"""Phân loại Clear Invoice và tính R5 v2 theo góp ý G2 (04/10/2026).

Đặt file này cùng thư mục với kiem_tra_trang_thai_case.py rồi chạy:
    python src/r5_v2_phan_loai_clear.py --tu-kiem                       # kiểm thử trên 8 vết giả lập
    python src/r5_v2_phan_loai_clear.py duong_dan/BPI_Challenge_2019.xes
    python src/r5_v2_phan_loai_clear.py duong_dan/BPI_Challenge_2019.xes --delta-phut 60

Lần chạy đầu không cần --delta-phut: script đề xuất Δ từ histogram (điểm tách hai cụm).
Xem hinh_khoang_cach_cancel_clear.png, nếu đồng ý thì giữ, nếu không thì chạy lại với --delta-phut.

Kết quả ghi vào thư mục ket_qua_r5_v2/ (đổi bằng --ra):
    tom_tat.txt                       toàn bộ số liệu in ra màn hình (gửi file này lại)
    bang4_kiem_chung.csv              dòng kiểm chứng Bảng 4 (phạm vi sau phễu lọc)
    bang2_luong_doi_chieu.csv         số case / sự kiện theo Item Category trên toàn log (Bảng 5)
    hoat_dong_42.csv                  42 hoạt động: tần suất trong phạm vi, khâu, có đổi số chứng từ phải trả không
    khoang_cach_cancel_clear.csv      mỗi lần hủy: Clear kế tiếp, khoảng cách (phút)
    hinh_khoang_cach_cancel_clear.png histogram thang log (cần matplotlib)
    bang_p1_pheu_r5.csv               Bảng P1
    luat_truoc_sau.csv                R1, R4, R5 trước/sau phân loại; độ nhạy Δ
    trang_thai_truoc_sau.csv          bảng chéo trạng thái case cũ × mới
    rework_theo_ncc.csv, rework_theo_hang_muc.csv
    mau_kiem_chung/nhom1_r5v2.csv, nhom2_bi_loai.csv, nhom3_huy_khong_ghep.csv
        (20 case mỗi nhóm, kèm toàn bộ vết; cột nhan_xet để bạn ghi kết quả đọc tay)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cau_hinh

try:
    from kiem_tra_trang_thai_case import buoc2_moc_trich_xuat, chuan_hoa_cot, doc_xes_gio_dia_phuong
except ImportError:
    sys.exit("Đặt file này cùng thư mục với kiem_tra_trang_thai_case.py")

PO, GR, CANCEL_GR = "Create Purchase Order Item", "Record Goods Receipt", "Cancel Goods Receipt"
IR, CANCEL_IR, CLEAR = "Record Invoice Receipt", "Cancel Invoice Receipt", "Clear Invoice"
MEMO, SUB_IR, CANCEL_SUB = "Vendor creates debit memo", "Record Subsequent Invoice", "Cancel Subsequent Invoice"
DELETE, REACT = "Delete Purchase Order Item", "Reactivate Purchase Order Item"
SEED = 2026
CASE_DA_DOC = {"2000000172_00001", "2000000005_00001", "2000000131_00001", "2000000157_00001"}

# Thứ tự khi trùng timestamp: nhận hàng và ghi nhận hóa đơn đứng trước Clear, các lần hủy đứng sau Clear.
# Nhờ vậy "trùng thời điểm = không vi phạm thứ tự" (R1: GR(<=t), Cancel GR(<t); R4: Cancel IR(<t), IR trong (cancel, t]).
THU_TU = {GR: 0, IR: 1, MEMO: 1, CLEAR: 2, CANCEL_IR: 3, CANCEL_GR: 3}

KHAU = {
    "Mua hàng": ["Create Purchase Requisition Item", "Release Purchase Requisition", PO, "Release Purchase Order",
                 "Change Approval for Purchase Order", "Receive Order Confirmation", "Update Order Confirmation",
                 "Change Quantity", "Change Price", "Change Delivery Indicator", "Change Storage Location",
                 "Change Currency", "Change Rejection Indicator"],
    "Mua hàng (SRM)": ["SRM: Created", "SRM: Awaiting Approval", "SRM: Complete", "SRM: Ordered",
                       "SRM: In Transfer to Execution Syst.", "SRM: Document Completed", "SRM: Change was Transmitted",
                       "SRM: Transfer Failed (E.Sys.)", "SRM: Transaction Completed", "SRM: Held", "SRM: Incomplete"],
    "Kho": [GR, "Record Service Entry Sheet"],
    "Kế toán phải trả": [IR, CLEAR, "Remove Payment Block", "Set Payment Block", SUB_IR,
                         "Change Final Invoice Indicator", "Change payment term"],
    "Nhà cung cấp": ["Vendor creates invoice", MEMO],
    "Hủy – sửa": [CANCEL_IR, CANCEL_GR, CANCEL_SUB, DELETE, REACT, "Block Purchase Order Item", "SRM: Deleted"],
}
CHUNG_TU = {IR: "+1 (ghi nhận hóa đơn)", CANCEL_IR: "−1 (hủy hóa đơn)", MEMO: "+1 (chỉ có phía NCC, không có hoạt động ghi nhận)",
            SUB_IR: "+1 nếu tính (độ nhạy)", CANCEL_SUB: "−1 nếu tính (độ nhạy)",
            "Vendor creates invoice": "Không đếm (đã có Record Invoice Receipt tương ứng)",
            CLEAR: "Không đổi số chứng từ; là thanh toán hoặc bù trừ"}


# ---------------------------------------------------------------- đọc log, giữ thêm Vendor / Spend area
def doc_log_day_du(duong_dan):
    p = Path(duong_dan)
    raw = pd.read_csv(p, encoding="latin-1", low_memory=False) if p.suffix.lower() == ".csv" else doc_xes_gio_dia_phuong(p)
    df = chuan_hoa_cot(raw)
    for chuan, ds in {"vendor": ["case:Vendor", "case Vendor"],
                      "spend_area": ["case:Spend area text", "case Spend area text"]}.items():
        for c in ds:
            if c in raw.columns:
                df[chuan] = raw.loc[df.index, c].astype(str).values
                break
    return df


def loc_pham_vi(df, cong_ty, moc):
    attr = df.drop_duplicates("case").set_index("case")
    dau_po = df[df["act"] == PO].groupby("case")["ts"].min().reindex(attr.index)
    loi = set(df.loc[(df["ts"] < "2017-01-01") | (df["ts"] > moc), "case"])
    ok = (attr["item_category"].astype(str).str.contains("invoice before GR", case=False)
          & attr["doc_type"].astype(str).eq("Standard PO")
          & attr["item_type"].astype(str).eq("Standard")
          & attr["company"].astype(str).eq(cong_ty)
          & dau_po.dt.year.eq(2018)
          & ~attr.index.isin(loi))
    return set(attr.index[ok.fillna(False)])


def sap_xep(ev):
    ev = ev.copy()
    ev["_tt"] = ev["act"].map(THU_TU).fillna(5)
    ev = ev.sort_values(["case", "ts", "_tt"], kind="stable").reset_index(drop=True)
    return ev.drop(columns="_tt")


# ---------------------------------------------------------------- phân loại Clear
def khoang_cach_cancel_clear(ev):
    c = ev.loc[ev["act"] == CANCEL_IR, ["case", "ts"]].sort_values("ts")
    k = ev.loc[ev["act"] == CLEAR, ["case", "ts"]].rename(columns={"ts": "ts_clear"}).sort_values("ts_clear")
    m = pd.merge_asof(c, k, left_on="ts", right_on="ts_clear", by="case", direction="forward")
    m["phut"] = (m["ts_clear"] - m["ts"]).dt.total_seconds() / 60
    return m.sort_values(["case", "ts"]).reset_index(drop=True)


def de_xuat_delta(phut):
    """Điểm thấp nhất của histogram log10(phút) giữa đỉnh <= 1 ngày và đỉnh > 1 ngày."""
    x = np.log10(phut[phut > 0].dropna())
    if len(x) < 20:
        return None
    bins = np.arange(0, 6.01, 0.2)  # 1 phút .. ~1,9 năm
    h, _ = np.histogram(x, bins)
    tam = (bins[:-1] + bins[1:]) / 2
    trai, phai = tam <= np.log10(1440), tam > np.log10(1440)
    if h[trai].max() == 0 or h[phai].max() == 0:
        return None
    i1 = np.where(trai)[0][h[trai].argmax()]
    i2 = np.where(phai)[0][h[phai].argmax()]
    j = i1 + h[i1:i2 + 1].argmin()
    return float(10 ** bins[j + 1])  # cạnh phải của ô thấp nhất


def phan_loai_clear(ev, delta_phut):
    """Trả về Series loai_clear (index = chỉ số dòng Clear) và DataFrame các lần hủy không ghép được."""
    delta = pd.Timedelta(minutes=delta_phut)
    loai = pd.Series("thanh_toan", index=ev.index[ev["act"] == CLEAR])
    nguoi_cung, khong_ghep = [], []
    co_huy = set(ev.loc[ev["act"] == CANCEL_IR, "case"])
    sub = ev[ev["case"].isin(co_huy) & ev["act"].isin([CANCEL_IR, CLEAR])]
    for case, g in sub.groupby("case", sort=False):
        clears = g[g["act"] == CLEAR]
        da_ghep = set()
        for ci, c in g[g["act"] == CANCEL_IR].iterrows():
            ghep = None
            for ki, kts in zip(clears.index, clears["ts"]):
                if ki in da_ghep or kts < c["ts"]:
                    continue
                if kts <= c["ts"] + delta:
                    ghep = ki
                break  # Clear gần nhất sau lần hủy: nằm trong Δ thì ghép, không thì thôi
            if ghep is None:
                khong_ghep.append({"case": case, "ts_cancel": c["ts"]})
            else:
                da_ghep.add(ghep)
                loai[ghep] = "bu_tru_huy"
                if "user" in ev:
                    nguoi_cung.append(ev.at[ghep, "user"] == c.get("user"))
    return loai, pd.DataFrame(khong_ghep, columns=["case", "ts_cancel"]), nguoi_cung


# ---------------------------------------------------------------- luật
def r1_theo_clear(ev, chi_thanh_toan):
    e = ev[ev["act"].isin([GR, CANCEL_GR, CLEAR])].copy()
    e["d"] = e["act"].map({GR: 1, CANCEL_GR: -1, CLEAR: 0})
    e["cum"] = e.groupby("case")["d"].cumsum()
    m = (e["act"] == CLEAR) & (e["cum"] <= 0)
    if chi_thanh_toan is not None:
        m &= e.index.isin(chi_thanh_toan)
    return set(e.loc[m, "case"])


def r4_theo_clear(ev, chi_thanh_toan):
    e = ev[ev["act"].isin([IR, CANCEL_IR, CLEAR])].copy()
    # Cancel IR gần nhất xếp trước Clear (Cancel cùng thời điểm với Clear xếp sau nên không tính);
    # IR cùng thời điểm với lần hủy được coi là "xen giữa" (trùng thời điểm = không vi phạm).
    e["tc"] = e["ts"].where(e["act"] == CANCEL_IR)
    e["ti"] = e["ts"].where(e["act"] == IR)
    e[["tc", "ti"]] = e.groupby("case")[["tc", "ti"]].ffill()
    m = (e["act"] == CLEAR) & e["tc"].notna() & (e["ti"].isna() | (e["ti"] < e["tc"]))
    if chi_thanh_toan is not None:
        m &= e.index.isin(chi_thanh_toan)
    return set(e.loc[m, "case"])


def r5_theo_thoi_diem(ev, chi_thanh_toan):
    """R5 v2.1: tại mỗi Clear thanh toán, số lần đã thanh toán > max(0, #IR − #Cancel IR) + #memo tính đến thời điểm đó.
    Lần hủy hóa đơn xảy ra SAU khi đã thanh toán không biến lần thanh toán trước đó thành trả thừa,
    và lần hủy không có hóa đơn nào để hủy không làm số chứng từ âm."""
    e = ev[ev["act"].isin([IR, CANCEL_IR, MEMO, CLEAR])]
    e = e[(e["act"] != CLEAR) | e.index.isin(chi_thanh_toan)]
    lk = {k: (e["act"] == a).groupby(e["case"]).cumsum() for k, a in
          {"ir": IR, "huy": CANCEL_IR, "memo": MEMO, "tra": CLEAR}.items()}
    hop_le = (lk["ir"] - lk["huy"]).clip(lower=0) + lk["memo"]
    return set(e.loc[(e["act"] == CLEAR) & (lk["tra"] > hop_le), "case"])


def dem_theo_case(ev, cases, loai):
    d = pd.DataFrame(index=pd.Index(sorted(cases), name="case"))
    for k, a in {"ir": IR, "cancel": CANCEL_IR, "clear": CLEAR, "memo": MEMO, "sub": SUB_IR,
                 "cancel_sub": CANCEL_SUB, "delete": DELETE}.items():
        d["n_" + k] = ev[ev["act"] == a].groupby("case").size()
    d = d.fillna(0).astype(int)
    tt = ev.loc[loai.index[loai == "thanh_toan"]]
    d["n_clear_tt"] = tt.groupby("case").size().reindex(d.index).fillna(0).astype(int)
    d["cuoi_clear"] = ev[ev["act"] == CLEAR].groupby("case")["ts"].max()
    d["cuoi_clear_tt"] = tt.groupby("case")["ts"].max()
    d["dau_clear_tt"] = tt.groupby("case")["ts"].min()
    d["cuoi_ir"] = ev[ev["act"] == IR].groupby("case")["ts"].max()
    d["dau_po"] = ev[ev["act"] == PO].groupby("case")["ts"].min()
    d["cuoi_delete"] = ev[ev["act"] == DELETE].groupby("case")["ts"].max()
    d["cuoi_react"] = ev[ev["act"] == REACT].groupby("case")["ts"].max()
    return d


def trang_thai(d, n_clear, hop_le, cuoi_clear):
    hoan_tat = (n_clear >= 1) & (n_clear >= hop_le) & (d["cuoi_ir"].isna() | (cuoi_clear >= d["cuoi_ir"]))
    bi_xoa = (d["n_delete"] >= 1) & (n_clear == 0) & (d["cuoi_react"].isna() | (d["cuoi_delete"] > d["cuoi_react"]))
    return pd.Series(np.where(hoan_tat, "Hoàn tất", np.where(bi_xoa, "Bị xóa", "Còn mở")), index=d.index)


# ---------------------------------------------------------------- tự kiểm
def log_tu_kiem():
    t = pd.Timestamp("2018-03-01 08:00")
    m, ngay = pd.Timedelta(minutes=1), pd.Timedelta(days=1)
    vet = {  # (hoạt động, thời điểm); kỳ vọng: (R1v2, R4v1, R4v2, R5v1, R5v2, R5v2.1)
        "A_sua_hoa_don": ([(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CANCEL_IR, 30 * ngay), (IR, 30 * ngay + 2 * m),
                           (CLEAR, 30 * ngay + 3 * m), (CLEAR, 60 * ngay)], (0, 0, 0, 1, 0, 0)),
        "B_tra_trung": ([(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay), (CLEAR, 40 * ngay)], (0, 0, 0, 1, 1, 1)),
        "C_debit_memo": ([(PO, 0 * m), (MEMO, ngay), (GR, ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay), (CLEAR, 31 * ngay)], (0, 0, 0, 1, 0, 0)),
        "D_huy_khong_bu_tru": ([(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CANCEL_IR, 3 * ngay), (IR, 3 * ngay + m),
                                (CLEAR, 20 * ngay), (CLEAR, 40 * ngay)], (0, 0, 0, 1, 1, 1)),
        "E_trung_timestamp": ([(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CANCEL_IR, 5 * ngay), (IR, 5 * ngay),
                               (CLEAR, 5 * ngay), (CLEAR, 30 * ngay)], (0, 0, 0, 1, 0, 0)),
        "F_huy_hai_lan": ([(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CANCEL_IR, 3 * ngay), (IR, 3 * ngay + m), (CLEAR, 3 * ngay + 2 * m),
                           (CANCEL_IR, 9 * ngay), (IR, 9 * ngay + m), (CLEAR, 9 * ngay + 2 * m), (CLEAR, 30 * ngay)], (0, 0, 0, 1, 0, 0)),
        "G_huy_bu_tru_xong": ([(PO, 0 * m), (IR, ngay), (CANCEL_IR, 5 * ngay), (CLEAR, 5 * ngay + 3 * m)], (0, 1, 0, 1, 0, 0)),
        "H_tra_truoc_nhan_hang": ([(PO, 0 * m), (IR, ngay), (CLEAR, 10 * ngay), (GR, 20 * ngay)], (1, 0, 0, 0, 0, 0)),
        # hai mẫu gặp khi đọc tay 13/10: hủy hóa đơn sau khi đã trả (batch), hủy khi không có hóa đơn
        "I_huy_sau_khi_tra": ([(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 10 * ngay), (CANCEL_IR, 100 * ngay)], (0, 0, 0, 1, 1, 0)),
        "J_huy_khong_co_hd": ([(PO, 0 * m), (CANCEL_GR, ngay), (CANCEL_IR, 100 * ngay)], (0, 0, 0, 1, 1, 0)),
    }
    dong = [{"case": c, "act": a, "ts": t + dt} for c, (ds, _) in vet.items() for a, dt in ds]
    return pd.DataFrame(dong), {c: kq for c, (_, kq) in vet.items()}


def tu_kiem():
    df, ky_vong = log_tu_kiem()
    ev = sap_xep(df)
    loai, _, _ = phan_loai_clear(ev, 60)
    tt = loai.index[loai == "thanh_toan"]
    d = dem_theo_case(ev, set(ev["case"]), loai)
    r1v2, r4v1, r4v2 = r1_theo_clear(ev, tt), r4_theo_clear(ev, None), r4_theo_clear(ev, tt)
    r5v1 = d["n_clear"] > d["n_ir"] - d["n_cancel"]
    r5v2 = d["n_clear_tt"] > d["n_ir"] - d["n_cancel"] + d["n_memo"]
    r5v21 = r5_theo_thoi_diem(ev, tt)
    loi = 0
    for c, kv in ky_vong.items():
        kq = (int(c in r1v2), int(c in r4v1), int(c in r4v2), int(r5v1[c]), int(r5v2[c]), int(c in r5v21))
        dat = kq == kv
        loi += not dat
        print(f"{c:24s} R1v2,R4v1,R4v2,R5v1,R5v2,R5v2.1 = {kq}  kỳ vọng {kv}  {'ĐẠT' if dat else 'SAI'}")
    print("\nTất cả đều đạt." if loi == 0 else f"\n{loi} vết SAI, xem lại trước khi chạy log thật.")
    return loi


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(cau_hinh.LOG), help="đường dẫn log; mặc định trong cau_hinh.py")
    ap.add_argument("--cong-ty", default="companyID_0000")
    ap.add_argument("--moc", help="thời điểm trích xuất đặt tay, dạng YYYY-MM-DD")
    ap.add_argument("--delta-phut", type=float, help="Δ (phút); bỏ trống để script đề xuất từ histogram")
    ap.add_argument("--da-doc", help="thư mục chứa các file CSV case đã đọc (vd ket_qua_5_2/buoc5_case_bien) để loại khỏi mẫu")
    ap.add_argument("--ra", default="ket_qua_r5_v2")
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

    def pct(n, N):
        return f"{n:,} ({n / N:.2%})".replace(",", "X").replace(".", ",").replace("X", ".")

    df = doc_log_day_du(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(df, a.moc)
    attr = df.drop_duplicates("case").set_index("case")

    # Bảng 5: luồng đối chiếu trên toàn log
    b5 = df.groupby(df["case"].map(attr["item_category"]))["case"].agg(so_case="nunique", so_su_kien="size")
    b5.to_csv(ra / "bang2_luong_doi_chieu.csv", encoding="utf-8-sig")
    p("=== Bảng 5. Luồng đối chiếu (toàn log) ===\n" + b5.to_string())

    pv = loc_pham_vi(df, a.cong_ty, moc)
    ev = sap_xep(df[df["case"].isin(pv)])
    N = len(pv)
    p(f"\nThời điểm trích xuất: {moc}. Phạm vi: {N:,} case, {len(ev):,} sự kiện")

    # Bảng 4: dòng kiểm chứng
    giay = (ev["ts"].dt.second != 0).mean()
    nua_dem = (ev["ts"].dt.strftime("%H:%M") == "00:00")
    cung_case = ev["case"].eq(ev["case"].shift())
    trung = cung_case & ev["ts"].eq(ev["ts"].shift())
    b4 = pd.DataFrame([{
        "so_case": N, "so_su_kien": len(ev), "so_hoat_dong": ev["act"].nunique(),
        "ty_le_su_kien_co_giay_khac_0": round(giay, 4),
        "ty_le_su_kien_luc_00_00": round(nua_dem.mean(), 4),
        "so_cap_lien_tiep": int(cung_case.sum()), "so_cap_trung_thoi_diem": int(trung.sum()),
        "ty_le_cap_trung": round(trung.sum() / cung_case.sum(), 4),
        "so_case_co_cap_trung": ev.loc[trung, "case"].nunique()}])
    b4.to_csv(ra / "bang4_kiem_chung.csv", index=False, encoding="utf-8-sig")
    p("\n=== Bảng 4. Dòng kiểm chứng (trong phạm vi) ===\n" + b4.T.to_string(header=False))
    theo_hd_nua_dem = nua_dem.groupby(ev["act"]).mean()
    p("Hoạt động có >= 90% sự kiện lúc 00:00 (chỉ ghi đến ngày): " + ", ".join(theo_hd_nua_dem[theo_hd_nua_dem >= 0.9].index))

    # 42 hoạt động
    hd = df.groupby("act").agg(su_kien_toan_log=("case", "size"))
    hd["su_kien_pham_vi"] = ev.groupby("act").size()
    hd["case_pham_vi"] = ev.groupby("act")["case"].nunique()
    hd = hd.fillna(0).astype(int)
    hd["ty_le_case_pham_vi"] = (hd["case_pham_vi"] / N).round(4)
    k = {act: khau for khau, ds in KHAU.items() for act in ds}
    hd["khau"] = hd.index.map(lambda x: k.get(x, "Khác (chưa phân loại)"))
    hd["anh_huong_so_chung_tu_phai_tra"] = hd.index.map(lambda x: CHUNG_TU.get(x, ""))
    hd = hd.sort_values("su_kien_pham_vi", ascending=False)
    hd.to_csv(ra / "hoat_dong_42.csv", encoding="utf-8-sig")
    p(f"\n=== Danh sách {len(hd)} hoạt động ===\n" + hd.to_string())

    # Histogram Cancel -> Clear kế tiếp
    kc = khoang_cach_cancel_clear(ev)
    kc.to_csv(ra / "khoang_cach_cancel_clear.csv", index=False, encoding="utf-8-sig")
    ph = kc["phut"]
    moc_kc = [("cùng phút", ph == 0), ("1–10 phút", (ph > 0) & (ph <= 10)), ("10–60 phút", (ph > 10) & (ph <= 60)),
              ("1–24 giờ", (ph > 60) & (ph <= 1440)), ("1–7 ngày", (ph > 1440) & (ph <= 10080)),
              ("> 7 ngày", ph > 10080), ("không có Clear sau", ph.isna())]
    p(f"\n=== Khoảng cách Cancel Invoice Receipt → Clear Invoice kế tiếp ({len(kc):,} lần hủy) ===")
    for ten, m in moc_kc:
        p(f"  {ten:20s} {pct(int(m.sum()), len(kc))}")
    p("  Phân vị (phút): " + ", ".join(f"P{q}={ph.quantile(q / 100):,.0f}" for q in (10, 25, 50, 75, 90)))
    dx = de_xuat_delta(ph)
    p(f"  Δ đề xuất từ histogram: {dx:,.0f} phút" if dx else "  Không thấy hai cụm rõ, cần chọn Δ bằng tay.")
    delta = a.delta_phut or dx or 60
    p(f"  => Δ dùng cho kết quả chính: {delta:,.0f} phút" + ("" if a.delta_phut else " (tự động; đặt lại bằng --delta-phut)"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        x = ph.dropna().clip(lower=0.5)
        ax.hist(x, bins=np.logspace(np.log10(0.5), np.log10(max(x.max(), 2)), 60), color="#4472c4")
        ax.set_xscale("log")
        for v, nhan in [(delta, f"Δ = {delta:,.0f} phút"), (60, "1 giờ"), (1440, "1 ngày")]:
            ax.axvline(v, ls="--", color="k" if v == delta else "gray", lw=1)
            ax.text(v, ax.get_ylim()[1] * 0.95, nhan, rotation=90, va="top", ha="right", fontsize=8)
        ax.set_xlabel("Khoảng cách từ Cancel Invoice Receipt đến Clear Invoice kế tiếp (phút, thang log; 0 phút vẽ ở 0,5)")
        ax.set_ylabel("Số lần hủy")
        fig.tight_layout()
        fig.savefig(ra / "hinh_khoang_cach_cancel_clear.png", dpi=150)
    except ImportError:
        p("  (Chưa có matplotlib nên không vẽ hình: pip install matplotlib)")

    # Phân loại Clear và các luật
    tt_cu = None
    d_cuoi, loai_cuoi, kg_cuoi, ng_cuoi = None, None, None, None
    dong_luat = []
    for nhan, dp in [("Δ chính", delta), ("Δ = 1 giờ", 60), ("Δ = 1 ngày", 1440)]:
        loai, kg, ng = phan_loai_clear(ev, dp)
        tt = loai.index[loai == "thanh_toan"]
        d = dem_theo_case(ev, pv, loai)
        r5v2 = d.index[d["n_clear_tt"] > d["n_ir"] - d["n_cancel"] + d["n_memo"]]
        r5v2_sub = d.index[d["n_clear_tt"] > d["n_ir"] - d["n_cancel"] + d["n_memo"] + d["n_sub"]]
        r5v21 = r5_theo_thoi_diem(ev, tt) & set(pv)
        so_huy = int((ev["act"] == CANCEL_IR).sum())
        dong_luat.append({"phien_ban": nhan, "delta_phut": dp, "so_lan_huy": so_huy,
                          "so_lan_huy_ghep_duoc": so_huy - len(kg), "ty_le_ghep": round((so_huy - len(kg)) / so_huy, 4) if so_huy else None,
                          "so_clear_bu_tru": int((loai == "bu_tru_huy").sum()), "R5_v2": len(r5v2),
                          "R5_v2_tinh_ca_hd_bo_sung": len(r5v2_sub), "R5_v2_1": len(r5v21),
                          "R1_v2": len(r1_theo_clear(ev, tt)), "R4_v2": len(r4_theo_clear(ev, tt)),
                          "ty_le_bu_tru_cung_nguoi_huy": round(float(np.mean(ng)), 4) if ng else None})
        if nhan == "Δ chính":
            d_cuoi, loai_cuoi, kg_cuoi, ng_cuoi, r5v2_cuoi, r5v21_cuoi = d, loai, kg, ng, set(r5v2), r5v21
    luat = pd.DataFrame(dong_luat)
    r1v1, r4v1 = r1_theo_clear(ev, None), r4_theo_clear(ev, None)
    luat.insert(0, "R1_v1", len(r1v1))
    luat.insert(1, "R4_v1", len(r4v1))
    luat.to_csv(ra / "luat_truoc_sau.csv", index=False, encoding="utf-8-sig")
    d = d_cuoi
    p("\n=== Phân loại Clear và luật trước/sau (số case) ===\n" + luat.T.to_string(header=False))
    p(f"R1: v1 {pct(len(r1v1), N)} → v2 {pct(int(luat.loc[0, 'R1_v2']), N)}")
    p(f"R4: v1 {pct(len(r4v1), N)} → v2 {pct(int(luat.loc[0, 'R4_v2']), N)}")

    # Bảng P1
    r5v1 = set(d.index[d["n_clear"] > d["n_ir"] - d["n_cancel"]])
    de_xuat = d["n_clear"] > d["n_ir"]
    co_memo = d["n_memo"] > 0
    giai_thich = de_xuat & co_memo & (d["n_clear"] <= d["n_ir"] + d["n_memo"])
    van_du = de_xuat & co_memo & (d["n_clear"] > d["n_ir"] + d["n_memo"])
    p1 = pd.DataFrame([
        ("v1", "#Clear > #IR − #Cancel IR", len(r5v1)),
        ("Đề xuất của SV", "#Clear > #IR", int(de_xuat.sum())),
        ("", "trong đó: không có debit memo", int((de_xuat & ~co_memo).sum())),
        ("", "trong đó: có debit memo", int((de_xuat & co_memo).sum())),
        ("", "– dư được giải thích bằng debit memo", int(giai_thich.sum())),
        ("", "– vẫn dư sau khi tính debit memo", int(van_du.sum())),
        ("Trước phân loại Clear", "#Clear > #IR + #memo (= 'không memo' + 'vẫn dư')", int((d["n_clear"] > d["n_ir"] + d["n_memo"]).sum())),
        ("v2", f"Clear thanh toán > #IR − #Cancel IR + #memo, Δ = {delta:,.0f} phút", len(r5v2_cuoi)),
        ("Độ nhạy", "R5 v2 với Δ = 1 giờ", int(luat.loc[1, "R5_v2"])),
        ("Độ nhạy", "R5 v2 với Δ = 1 ngày", int(luat.loc[2, "R5_v2"])),
        ("Độ nhạy", "R5 v2 có tính Record Subsequent Invoice", int(luat.loc[0, "R5_v2_tinh_ca_hd_bo_sung"])),
        ("v2.1", f"Theo thời điểm: tại Clear thanh toán thứ k, k > max(0, #IR − #Cancel IR) + #memo tính đến lúc đó, Δ = {delta:,.0f} phút", len(r5v21_cuoi)),
        ("Độ nhạy", "R5 v2.1 với Δ = 1 ngày", int(luat.loc[2, "R5_v2_1"])),
        ("", "trong đó: cũng bị v2 bắt", len(r5v21_cuoi & r5v2_cuoi)),
        ("", "bị v2 bắt nhưng v2.1 không (hủy sau khi trả, hủy khi không có hóa đơn)", len(r5v2_cuoi - r5v21_cuoi)),
    ], columns=["buoc", "dinh_nghia", "so_case"])
    p1["phan_tram_pham_vi"] = (p1["so_case"] / N * 100).round(2)
    p1.to_csv(ra / "bang_p1_pheu_r5.csv", index=False, encoding="utf-8-sig")
    p("\n=== Bảng P1. Phễu tinh chỉnh R5 ===\n" + p1.to_string(index=False))

    # Trạng thái trước/sau
    cu = trang_thai(d, d["n_clear"], d["n_ir"] - d["n_cancel"], d["cuoi_clear"])
    moi = trang_thai(d, d["n_clear_tt"], d["n_ir"] - d["n_cancel"] + d["n_memo"], d["cuoi_clear_tt"])
    cheo = pd.crosstab(cu.rename("trang_thai_cu"), moi.rename("trang_thai_moi"), margins=True)
    cheo.to_csv(ra / "trang_thai_truoc_sau.csv", encoding="utf-8-sig")
    p("\n=== Trạng thái case: cũ (mọi Clear) × mới (Clear thanh toán, có debit memo) ===\n" + cheo.to_string())
    p(f"Số case đổi trạng thái: {pct(int((cu != moi).sum()), N)}")
    ht = moi == "Hoàn tất"
    ngay_cu = (d.loc[ht, "cuoi_clear"] - d.loc[ht, "dau_po"]).dt.total_seconds() / 86400
    ngay_moi = (d.loc[ht, "cuoi_clear_tt"] - d.loc[ht, "dau_po"]).dt.total_seconds() / 86400
    ngay_dau = (d.loc[ht, "dau_clear_tt"] - d.loc[ht, "dau_po"]).dt.total_seconds() / 86400
    p(f"Thời gian thanh toán (case hoàn tất mới, trung vị ngày): đến Clear cuối (mọi loại) {ngay_cu.median():.1f}; "
      f"đến Clear thanh toán cuối {ngay_moi.median():.1f}; đến Clear thanh toán đầu {ngay_dau.median():.1f}")

    # Rework
    co_huy = d["n_cancel"] > 0
    p(f"\n=== Rework: dòng hàng có hủy hóa đơn: {pct(int(co_huy.sum()), N)} ===")
    for cot, ten in [("vendor", "rework_theo_ncc.csv"), ("spend_area", "rework_theo_hang_muc.csv")]:
        if cot in ev:
            g = pd.DataFrame({"nhom": attr.loc[d.index, cot] if cot in attr else ev.groupby("case")[cot].first().reindex(d.index),
                              "co_huy": co_huy})
            bang = g.groupby("nhom")["co_huy"].agg(so_case="size", so_case_co_huy="sum")
            bang["ty_le"] = (bang["so_case_co_huy"] / bang["so_case"]).round(4)
            bang = bang.sort_values("so_case", ascending=False)
            bang.to_csv(ra / ten, encoding="utf-8-sig")
            p(f"{ten}: 10 nhóm lớn nhất\n" + bang.head(10).to_string())

    # Ba nhóm mẫu kiểm chứng
    da_doc = set(CASE_DA_DOC)
    if a.da_doc and Path(a.da_doc).exists():
        for f in Path(a.da_doc).glob("*.csv"):
            try:
                da_doc |= set(pd.read_csv(f, usecols=["case"])["case"].astype(str))
            except Exception:
                pass
    p(f"\nLoại {len(da_doc):,} case đã đọc khi thiết kế luật khỏi mẫu kiểm chứng.")
    rng = np.random.default_rng(SEED)
    nhom = {
        "nhom1_r5v2": sorted(r5v2_cuoi - da_doc),
        "nhom2_bi_loai": sorted((r5v1 - r5v2_cuoi) - da_doc),
        "nhom3_huy_khong_ghep": sorted(set(kg_cuoi["case"]) - da_doc),
        "nhom4_r5v2_1": sorted(r5v21_cuoi - da_doc),
    }
    ev_loai = ev.copy()
    ev_loai["loai_clear"] = loai_cuoi.reindex(ev.index)
    for ten, ds in nhom.items():
        chon = list(rng.choice(ds, size=min(20, len(ds)), replace=False)) if ds else []
        vet = ev_loai[ev_loai["case"].isin(chon)][["case", "ts", "act", "loai_clear"] + (["user"] if "user" in ev else [])]
        vet = vet.assign(nhan_xet="")
        vet.to_csv(ra / "mau_kiem_chung" / f"{ten}.csv", index=False, encoding="utf-8-sig")
        p(f"{ten}: {len(ds):,} case đủ điều kiện, lấy {len(chon)}")

    (ra / "tom_tat.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat.txt, hình histogram và thư mục mau_kiem_chung.")


if __name__ == "__main__":
    main()
