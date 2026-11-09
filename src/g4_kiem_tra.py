"""Các kiểm tra theo góp ý G4 (16/10/2026) của thầy, hạn Chủ nhật 18/10.

Cách chạy (đặt cùng thư mục với r5_v2_phan_loai_clear.py, g3_kiem_tra_bo_sung.py, r4_thanh_phan.py
và kiem_tra_trang_thai_case.py, chạy ở thư mục có sẵn ket_qua_g3/ và ket_qua_r4/ để loại các case đã đọc):
    python src/g4_kiem_tra.py
    python src/g4_kiem_tra.py --tu-kiem        # chạy thử trên vết giả lập

Kết quả ghi vào ket_qua_g4/ (đổi bằng --ra); tom_tat_g4.txt là bản tóm tắt, gửi lại file này.
Mục trong G4 được ghi ở đầu mỗi phần: 0 (múi giờ), 2, 3, 4, 5.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import g3_kiem_tra_bo_sung as g3
import r4_thanh_phan as r4tp
import r5_v2_phan_loai_clear as r5
from r5_v2_phan_loai_clear import CANCEL_IR, CLEAR, IR, MEMO, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear, sap_xep
import cau_hinh

SEED = 2026
DELTA = 60
DELTA_MO_RONG = 1000  # G4 Mục 3: độ nhạy, Clear trong 1.000 phút do chính người hủy thực hiện
NGAY_NGHI = 90        # G4 Mục 2: dòng hàng "đã kết thúc" = không có hoạt động trong 90 ngày trước thời điểm trích xuất
MUI_GIO = "Europe/Amsterdam"
VENDOR_INV = "Vendor creates invoice"
NHOM_TANG = ["có hủy và debit memo", "không hủy, không debit memo", "có debit memo, không hủy", "có hủy, không debit memo"]


def pct(n, N):
    s = f"{n:,} ({n / N:.2%})" if N else f"{n:,} (–)"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------- 0. múi giờ
def gio_dia_phuong(ev, p):
    """Log đọc vào đang ở UTC hay giờ địa phương? Nhận ra bằng giờ của Vendor creates invoice (23:59 địa phương).
    Trả về Series giờ địa phương (không múi giờ), chỉ dùng cho phân tích theo giờ trong ngày / thứ trong tuần."""
    hm = ev.loc[ev["act"] == VENDOR_INV, "ts"].dt.strftime("%H:%M")
    ty_le_2359 = (hm == "23:59").mean() if len(hm) else 0
    ty_le_utc = hm.isin(["21:59", "22:59"]).mean() if len(hm) else 0
    if ty_le_utc > 0.9:
        ts_dp = ev["ts"].dt.tz_localize("UTC").dt.tz_convert(MUI_GIO).dt.tz_localize(None)
        p(f"Log đang ở UTC (Vendor creates invoice lúc 21:59/22:59: {ty_le_utc:.1%}). Đổi sang {MUI_GIO} cho phân tích theo giờ/thứ.")
        hm2 = ts_dp[ev["act"] == VENDOR_INV].dt.strftime("%H:%M")
        p(f"  Sau khi đổi: Vendor creates invoice lúc 23:59 giờ địa phương: {(hm2 == '23:59').mean():.1%}")
        p("  Thứ tự sự kiện và khoảng cách thời gian không đổi; mọi luật vẫn tính trên timestamp gốc.")
        return ts_dp
    p(f"Log đã ở giờ địa phương (Vendor creates invoice lúc 23:59: {ty_le_2359:.1%}); không cần đổi múi giờ.")
    return ev["ts"].copy()


# ---------------------------------------------------------------- phân loại Clear có mở rộng (độ nhạy G4 Mục 3)
def phan_loai_clear_mo_rong(ev, delta_phut, mo_rong_phut):
    """Bước 1 như phan_loai_clear (Δ). Bước 2: lần hủy chưa ghép được, nếu Clear "thanh toán" kế tiếp
    (chưa ghép) nằm trong mo_rong_phut và do chính người hủy thực hiện thì cũng là "bù trừ do hủy"."""
    loai, khong_ghep, _ = phan_loai_clear(ev, delta_phut)
    loai = loai.copy()
    if "user" not in ev or khong_ghep.empty:
        return loai
    mr = pd.Timedelta(minutes=mo_rong_phut)
    huy = ev[(ev["act"] == CANCEL_IR)].set_index(["case", "ts"])["user"]
    huy = huy[~huy.index.duplicated()]
    clr = ev[ev["act"] == CLEAR]
    clr = {k: g for k, g in clr[clr["case"].isin(set(khong_ghep["case"]))].groupby("case")}
    for case, g in khong_ghep.groupby("case"):
        c = clr.get(case)
        if c is None:
            continue
        for tsc in g["ts_cancel"]:
            u = huy.get((case, tsc))
            sau = c[(c["ts"] >= tsc) & (loai.reindex(c.index) == "thanh_toan")]
            if sau.empty:
                continue
            ki = sau.index[0]
            if sau.at[ki, "ts"] <= tsc + mr and sau.at[ki, "user"] == u:
                loai[ki] = "bu_tru_huy"
    return loai


# ---------------------------------------------------------------- bộ đếm R5 v2.2 (mã giả P.2 của thầy)
def dem_r5(ev, loai, tinh_memo=True):
    """Mỗi Clear "thanh toán" một dòng: hop_le, n_tra (đếm sự kiện), n_tra_tg (đếm thời điểm khác nhau),
    vuot (v2.1), vuot_tg, ir_sau (có Record Invoice Receipt sau lần Clear này), da_huy (đã có Cancel IR trước đó).
    Sắp xếp sự kiện như sap_xep: IR/memo trước Clear, hủy sau Clear khi cùng timestamp."""
    e = ev[ev["act"].isin([IR, CANCEL_IR, MEMO, CLEAR])].copy()
    tt = set(loai.index[loai == "thanh_toan"])
    e = e[(e["act"] != CLEAR) | e.index.isin(tt)]
    lk = {k: (e["act"] == a).astype(int).groupby(e["case"]).cumsum()
          for k, a in {"ir": IR, "huy": CANCEL_IR, "memo": MEMO, "tra": CLEAR}.items()}
    hop_le = (lk["ir"] - lk["huy"]).clip(lower=0) + (lk["memo"] if tinh_memo else 0)
    la_tra = e["act"] == CLEAR
    ts_tra = e["ts"].where(la_tra)
    tra_truoc_ts = ts_tra.groupby(e["case"]).ffill().groupby(e["case"]).shift()  # ts của Clear thanh toán liền trước
    moi_tg = la_tra & ~(e["ts"] == tra_truoc_ts)
    tong_ir = (e["act"] == IR).astype(int).groupby(e["case"]).transform("sum")
    out = e.loc[la_tra, ["case", "ts"] + [c for c in ("user",) if c in e]].copy()
    out["hop_le"] = hop_le[la_tra]
    out["n_tra"] = lk["tra"][la_tra]
    out["n_tra_tg"] = moi_tg.astype(int).groupby(e["case"]).cumsum()[la_tra]
    out["vuot"] = out["n_tra"] > out["hop_le"]
    out["vuot_tg"] = out["n_tra_tg"] > out["hop_le"]
    out["ir_sau"] = (tong_ir - lk["ir"])[la_tra] > 0
    out["da_huy"] = lk["huy"][la_tra] > 0
    return out


def r5_v22(bd):
    """Trả về (tập R5 v2.2, nhánh R5a/R5b theo lần vi phạm đầu tiên, tập 'trả trước khi ghi nhận hóa đơn')."""
    vp = bd[bd["vuot_tg"] & ~bd["ir_sau"]]
    dau = vp.sort_values("ts").groupby("case").first()
    nhanh = pd.Series(np.where(dau["da_huy"], "R5a", "R5b"), index=dau.index, name="nhanh")
    tra_truoc = set(bd.loc[bd["vuot_tg"] & bd["ir_sau"], "case"])
    return set(vp["case"]), nhanh, tra_truoc


def k_memo(d, cases, ten, p):
    """G4 Mục 2: k = số Clear thanh toán − max(0, #IR − #Cancel IR), so với số debit memo."""
    x = d.loc[sorted(cases)].copy()
    x["hd_con"] = (x["n_ir"] - x["n_cancel"]).clip(lower=0)
    for cot_tra, nhan in (("n_clear_tt", "đếm sự kiện Clear"), ("n_tt_tg", "đếm thời điểm thanh toán khác nhau")):
        x["k"] = x[cot_tra] - x["hd_con"]
        m = x["n_memo"]
        ds = [("k = số debit memo", x["k"] == m), ("k = 0", (x["k"] == 0) & (m > 0)),
              ("0 < k < số debit memo", (x["k"] > 0) & (x["k"] < m)), ("k > số debit memo", x["k"] > m), ("k < 0", x["k"] < 0)]
        if (m == 0).all():
            ds = [("k = 0", x["k"] == 0), ("k > 0", x["k"] > 0), ("k < 0", x["k"] < 0)]
        p(f"  {ten}, {nhan} (n = {len(x):,}): " + "; ".join(f"{t} {pct(int(s.sum()), len(x))}" for t, s in ds))
    return x


# ---------------------------------------------------------------- phân tích
def phan_tich(ev, pv, moc, ra, p, rng):
    N = len(pv)
    ev = ev.copy()
    ev["ts_dp"] = gio_dia_phuong(ev, p)
    loai, khong_ghep, _ = phan_loai_clear(ev, DELTA)
    ev["loai_clear"] = loai.reindex(ev.index)
    tt = loai.index[loai == "thanh_toan"]
    d = r5.dem_theo_case(ev, pv, loai)
    d["n_tt_tg"] = ev.loc[tt].drop_duplicates(["case", "ts"]).groupby("case").size().reindex(d.index).fillna(0).astype(int)
    cuoi_sk = ev.groupby("case")["ts"].max().reindex(d.index)
    so_ngay_nghi = (pd.Timestamp(moc) - cuoi_sk).dt.days
    p(f"Phạm vi {N:,} case. Δ = {DELTA} phút. Thời điểm trích xuất: {moc}")

    # ======================= Mục 2. Debit memo có được trả riêng không
    p("\n=== G4 Mục 2. Debit memo có được trả bằng Clear riêng không ===")
    nghi = so_ngay_nghi > NGAY_NGHI
    co_memo = set(d.index[(d["n_memo"] >= 1) & nghi])
    khong_memo = set(d.index[(d["n_memo"] == 0) & nghi & (d["n_ir"] >= 1)])
    p(f"Dòng hàng có ≥ 1 debit memo và không có hoạt động trong {NGAY_NGHI} ngày trước thời điểm trích xuất: {len(co_memo):,}")
    x = k_memo(d, co_memo, "Có debit memo", p)
    p("  Đối chứng: dòng hàng không có debit memo, cũng đã nghỉ, có ≥ 1 IR (k kỳ vọng = 0 nếu mỗi hóa đơn một lần trả):")
    k_memo(d, khong_memo, "Không có debit memo", p)
    x["k"] = x["n_clear_tt"] - x["hd_con"]
    bang = pd.crosstab(x["k"].clip(-2, 5).rename("k (cắt ở -2 và 5)"),
                       x["n_memo"].clip(upper=3).map({1: "1 memo", 2: "2 memo", 3: "3+ memo"}).rename("so_memo"))
    bang.to_csv(ra / "g4_2_bang_k_theo_so_memo.csv", encoding="utf-8-sig")
    p("Bảng chéo k × số debit memo (đếm sự kiện Clear):\n" + bang.to_string())
    dung = (x["k"] == x["n_memo"]).mean()
    bang0 = ((x["k"] == 0)).mean()
    if dung > 0.5:
        ket_luan = "PHẦN LỚN k = số debit memo → debit memo được trả bằng Clear riêng; GIỮ cách tính hiện tại."
    elif bang0 > 0.5:
        ket_luan = "PHẦN LỚN k = 0 → debit memo được bù trừ chung; BỎ debit memo khỏi vế phải và tính lại (báo thầy)."
    else:
        ket_luan = "Không có nhóm nào chiếm quá nửa → chưa kết luận được, xem bảng chéo."
    p(f"Kết luận theo quy tắc của thầy: {ket_luan}")
    x.to_csv(ra / "g4_2_k_tung_dong_hang.csv", encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6))
        bang.plot(kind="bar", ax=ax)
        ax.set_xlabel("k = số Clear thanh toán − max(0, #IR − #Cancel IR)")
        ax.set_ylabel("Số dòng hàng")
        ax.set_title("Phân bố k theo số debit memo (dòng hàng đã nghỉ ≥ 90 ngày)")
        fig.tight_layout()
        fig.savefig(ra / "g4_2_hinh_k_theo_so_memo.png", dpi=200)
        plt.close(fig)
    except Exception as loi:  # không có matplotlib vẫn chạy tiếp
        p(f"  (Không vẽ được hình: {loi})")

    # tách 2.096 dòng hàng đổi trạng thái theo nguyên nhân
    hd = d["n_ir"] - d["n_cancel"]
    cu = r5.trang_thai(d, d["n_clear"], hd, d["cuoi_clear"])
    chi_pl = r5.trang_thai(d, d["n_clear_tt"], hd, d["cuoi_clear_tt"])          # chỉ phân loại lại Clear
    chi_memo = r5.trang_thai(d, d["n_clear"], hd + d["n_memo"], d["cuoi_clear"])  # chỉ cộng debit memo
    moi = r5.trang_thai(d, d["n_clear_tt"], hd + d["n_memo"], d["cuoi_clear_tt"])
    doi = (cu == "Hoàn tất") & (moi == "Còn mở")
    a, b = chi_pl != "Hoàn tất", chi_memo != "Hoàn tất"
    nn = pd.Series(np.select([a & ~b, ~a & b, a & b], ["chỉ do phân loại lại Clear bù trừ", "chỉ do cộng debit memo",
                                                       "cả hai, mỗi cái đều đủ"], "chỉ khi kết hợp cả hai"), index=d.index)
    p(f"\nDòng hàng đổi Hoàn tất → Còn mở: {int(doi.sum()):,}. Nguyên nhân:")
    rows = []
    for k in ["chỉ do phân loại lại Clear bù trừ", "chỉ do cộng debit memo", "cả hai, mỗi cái đều đủ", "chỉ khi kết hợp cả hai"]:
        m = doi & (nn == k)
        rows.append({"nguyen_nhan": k, "so_dong": int(m.sum()),
                     "nghi_90": int((m & (so_ngay_nghi > 90)).sum()), "nghi_180": int((m & (so_ngay_nghi > 180)).sum()),
                     "trung_vi_ngay_nghi": so_ngay_nghi[m].median()})
        p(f"  {k}: {pct(int(m.sum()), int(doi.sum()))}; trong đó không hoạt động > 90 ngày {pct(int((m & (so_ngay_nghi > 90)).sum()), int(m.sum()))}, "
          f"> 180 ngày {pct(int((m & (so_ngay_nghi > 180)).sum()), int(m.sum()))}")
    pd.DataFrame(rows).to_csv(ra / "g4_2_2096_theo_nguyen_nhan.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"nguyen_nhan": nn, "so_ngay_khong_hoat_dong": so_ngay_nghi})[doi].to_csv(
        ra / "g4_2_2096_tung_dong_hang.csv", encoding="utf-8-sig")

    # ======================= Mục 3. R4 gộp vào R5
    p("\n=== G4 Mục 3. R4: ba con số và nhánh R5a/R5b ===")
    r4v1 = r5.r4_theo_clear(ev, None)
    r4v2 = r5.r4_theo_clear(ev, tt)
    xr4 = r4tp.thanh_phan_r4(ev, loai)
    dau_r4 = xr4.sort_values("ts").groupby("case").first()
    dung_nghia_dau = set(dau_r4.index[dau_r4["loai_r4"].str[:2].isin(["A3", "B:"])])
    dung_nghia_bat_ky = set(xr4.loc[xr4["con_phai_tra"] <= 0, "case"])
    p(f"R4 cách đếm thô (mọi Clear): {pct(len(r4v1), N)}; R4 v2 (chỉ Clear thanh toán): {pct(len(r4v2), N)}; "
      f"R4 đúng nghĩa (lần Clear bị bắt đầu tiên là A3 hoặc B): {pct(len(dung_nghia_dau), N)}; "
      f"(có ít nhất một lần Clear bị bắt là A3 hoặc B): {pct(len(dung_nghia_bat_ky), N)}")

    # ======================= Mục 4. R5 v2.2
    p("\n=== G4 Mục 4. R5 v2.2 ===")
    bd = dem_r5(ev, loai)
    s21 = set(bd.loc[bd["vuot"], "case"])
    stg = set(bd.loc[bd["vuot_tg"], "case"])
    s22, nhanh, tra_truoc = r5_v22(bd)
    p(f"R5 v2.1: {pct(len(s21), N)}; đếm theo thời điểm: {pct(len(stg), N)}; R5 v2.2 (thêm: bỏ lần vượt có Record Invoice "
      f"Receipt ghi sau): {pct(len(s22), N)}")
    p(f"  Kiểm tra: case đếm theo thời điểm mà v2.1 không bắt: {len(stg - s21)} (kỳ vọng 0)")
    p(f"  Chỉ báo riêng 'trả trước khi ghi nhận hóa đơn' (có lần vượt mà sau đó có Record Invoice Receipt): {pct(len(tra_truoc), N)}; "
      f"trong đó vẫn còn lần vượt khác nên vẫn ở R5 v2.2: {len(tra_truoc & s22):,}")
    p(f"  Nhánh: R5a (vi phạm đầu tiên xảy ra sau ít nhất một lần hủy) {pct(int((nhanh == 'R5a').sum()), len(s22))}; "
      f"R5b {pct(int((nhanh == 'R5b').sum()), len(s22))}")
    p(f"  R4 đúng nghĩa (lần đầu A3/B) nằm trong R5 v2.2: {pct(len(dung_nghia_dau & s22), len(dung_nghia_dau))}; "
      f"trong R5a: {pct(len(dung_nghia_dau & set(nhanh.index[nhanh == 'R5a'])), len(dung_nghia_dau))}")

    # chênh lệch 112 và 135
    v = bd[bd["vuot"]]
    so_cung = bd.groupby(["case", "ts"]).size().rename("so_cung_ts")
    v = v.join(so_cung, on=["case", "ts"])
    vuot_cung = set(v.loc[v["so_cung_ts"] > 1, "case"])
    vuot_rieng = set(v.loc[v["so_cung_ts"] == 1, "case"])
    chi_cung = vuot_cung - vuot_rieng
    mat = s21 - stg
    p(f"\nĐối chiếu: bảng phân loại G3 có {len(chi_cung):,} case 'chỉ vượt ở Clear cùng timestamp'; đếm theo thời điểm loại {len(mat):,} case.")
    p(f"  {len(mat & chi_cung):,} case loại thuộc nhóm 'chỉ vượt cùng timestamp'; {len(mat - chi_cung):,} case còn lại có lần vượt ở một "
      f"timestamp riêng, nhưng lần vượt đó chỉ xuất hiện vì có 2 Clear cùng timestamp ở TRƯỚC đó làm số lần trả bị đếm dư.")
    cung_truoc = bd.join(so_cung, on=["case", "ts"])
    cung_truoc = cung_truoc[cung_truoc["so_cung_ts"] > 1].groupby("case")["ts"].min()
    vuot_dau = v[v["so_cung_ts"] == 1].groupby("case")["ts"].min()
    xac_nhan = [c for c in mat - chi_cung if c in cung_truoc.index and c in vuot_dau.index and cung_truoc[c] < vuot_dau[c]]
    p(f"  Xác nhận: {len(xac_nhan):,}/{len(mat - chi_cung):,} case có nhóm Clear cùng timestamp xảy ra trước lần vượt riêng đầu tiên.")
    pd.Series(sorted(mat - chi_cung), name="case").to_csv(ra / "g4_4_chenh_lech_23_case.csv", index=False, encoding="utf-8-sig")

    # phân loại loại trừ nhau theo thứ tự ưu tiên
    dd = d.loc[sorted(s21)]
    co_huy, co_m = dd["n_cancel"] > 0, dd["n_memo"] > 0
    lop = np.select([~dd.index.isin(stg), ~dd.index.isin(s22), co_huy & co_m, ~co_huy & ~co_m, co_m & ~co_huy, co_huy & ~co_m],
                    ["1. chỉ vượt do đếm Clear cùng timestamp", "2. trả trước khi ghi nhận hóa đơn",
                     "3. " + NHOM_TANG[0], "4. " + NHOM_TANG[1], "5. " + NHOM_TANG[2], "6. " + NHOM_TANG[3]], "khác")
    lop = pd.Series(lop, index=dd.index, name="loai")
    bang4 = lop.value_counts().sort_index()
    bang4.loc["Tổng (= R5 v2.1)"] = int(bang4.sum())
    bang4 = bang4.rename("so_case").to_frame()
    bang4["ty_le"] = (bang4["so_case"] / len(s21)).round(4)
    p("\nPhân loại R5 v2.1 theo thứ tự ưu tiên (loại trừ nhau; nhóm 3–6 cộng lại = R5 v2.2):\n" + bang4.to_string())
    p(f"  Nhóm 3–6 cộng lại: {int(lop.str[0].isin(list('3456')).sum()):,} (R5 v2.2 = {len(s22):,})")
    p("  Nhánh R5a/R5b theo nhóm:\n" + pd.crosstab(lop[lop.index.isin(s22)], nhanh).to_string())
    bang4.to_csv(ra / "g4_4_phan_loai_uu_tien.csv", encoding="utf-8-sig")
    pd.DataFrame({"loai": lop, "nhanh": nhanh.reindex(lop.index)}).to_csv(ra / "g4_4_r5_tung_case.csv", encoding="utf-8-sig")

    # mẫu kiểm chứng theo tầng: 10 case mỗi nhóm 3–6, loại các case đã đọc
    da_doc = set(g3.DA_DOC)
    for f in list(Path("ket_qua_g3/mau_kiem_chung").glob("*.csv")) + list(Path("ket_qua_r4/mau_kiem_chung").glob("*.csv")):
        try:
            da_doc |= set(pd.read_csv(f, encoding="utf-8-sig", usecols=["case"])["case"].astype(str))
        except Exception:
            pass
    p(f"\nMẫu theo tầng (10 case mỗi nhóm), đã loại {len(da_doc):,} case đọc trước đây:")
    phan = []
    for i, ten in enumerate(NHOM_TANG, start=3):
        ds = sorted(set(lop.index[lop == f"{i}. {ten}"]) - da_doc)
        chon = list(rng.choice(ds, size=min(10, len(ds)), replace=False)) if ds else []
        p(f"  {i}. {ten}: {len(ds):,} case đủ điều kiện, lấy {len(chon)}")
        phan.append(pd.DataFrame({"case": chon, "nhom": f"{i}. {ten}"}))
    chon = pd.concat(phan)
    vet = ev[ev["case"].isin(chon["case"])][["case", "ts", "act", "loai_clear", "user"]].merge(chon, on="case")
    vet["nhanh"] = vet["case"].map(nhanh)
    vet["nhan_xet"] = ""
    vet.to_csv(ra / "mau_kiem_chung" / "r5v22_theo_tang.csv", index=False, encoding="utf-8-sig")
    if "PHẦN LỚN k = 0" in ket_luan:
        p("  CẢNH BÁO: kiểm tra Mục 2 cho thấy debit memo bù trừ chung, R5 sẽ đổi; mẫu này cần lấy lại sau khi tính lại.")

    # độ nhạy
    p("\nĐộ nhạy R5 v2.2:")
    s22_khong_memo, _, _ = r5_v22(dem_r5(ev, loai, tinh_memo=False))
    p(f"  Không tính debit memo: {pct(len(s22_khong_memo), N)} (thêm {len(s22_khong_memo - s22):,}, bớt {len(s22 - s22_khong_memo):,})")
    loai_mr = phan_loai_clear_mo_rong(ev, DELTA, DELTA_MO_RONG)
    them_bu_tru = int(((loai_mr == "bu_tru_huy") & (loai == "thanh_toan")).sum())
    s22_mr, nhanh_mr, _ = r5_v22(dem_r5(ev, loai_mr))
    p(f"  Bù trừ mở rộng (thêm Clear trong {DELTA_MO_RONG:,} phút do chính người hủy): thêm {them_bu_tru:,} Clear bù trừ; "
      f"R5 v2.2 còn {pct(len(s22_mr), N)} (bớt {len(s22 - s22_mr):,}, thêm {len(s22_mr - s22):,}); R5a {int((nhanh_mr == 'R5a').sum()):,}")
    xr4_mr = r4tp.thanh_phan_r4(ev.assign(loai_clear=loai_mr.reindex(ev.index)), loai_mr)
    p(f"  R4 v2 với bù trừ mở rộng: {pct(xr4_mr['case'].nunique(), N)}")

    # Wilson gộp hai mẫu R5
    p("\nKhoảng tin cậy Wilson 95%, gộp hai mẫu R5 v2.1 (13/10 và 16/10):")
    for nhan, k, n in [("đáng nghi cao", 11, 40), ("đáng nghi cao + cần số tiền", 22, 40)]:
        lo, hi = g3.wilson(k, n)
        p(f"  {nhan}: {k}/{n} = {k / n:.1%} (KTC 95%: {lo:.0%}–{hi:.0%})")

    # ======================= Mục 5. Hủy do tài khoản batch, chỉ báo làm lại, múi giờ
    p("\n=== G4 Mục 5. Hủy do tài khoản batch và chỉ báo làm lại ===")
    huy = ev[ev["act"] == CANCEL_IR].copy()
    huy["batch"] = g3.la_batch(huy["user"]) if "user" in huy else False
    # lần hủy có ghép được Clear bù trừ hay không: so theo (case, ts) với danh sách không ghép
    kg = set(zip(khong_ghep["case"], khong_ghep["ts_cancel"]))
    huy["ghep_duoc"] = [(c, t) not in kg for c, t in zip(huy["case"], huy["ts"])]
    # hủy sau khi đã thanh toán (giống G3 2.5)
    e = ev[ev["act"].isin([IR, CANCEL_IR, MEMO, CLEAR])].copy()
    e = e[(e["act"] != CLEAR) | (e["loai_clear"] == "thanh_toan")]
    lk = {k: (e["act"] == a).astype(int).groupby(e["case"]).cumsum() for k, a in {"ir": IR, "huy": CANCEL_IR, "memo": MEMO, "tra": CLEAR}.items()}
    hl_truoc = (lk["ir"] - (lk["huy"] - 1)).clip(lower=0) + lk["memo"]
    sau_tra = (e["act"] == CANCEL_IR) & (lk["tra"] >= hl_truoc) & (lk["tra"] > 0)
    huy["sau_khi_tra"] = huy.index.isin(e.index[sau_tra])
    p(f"Tổng số lần Cancel Invoice Receipt: {len(huy):,}; do tài khoản batch: {pct(int(huy['batch'].sum()), len(huy))}")
    for ten, m in [("hủy sau khi đã thanh toán", huy["sau_khi_tra"]), ("hủy khác (chưa thanh toán đủ)", ~huy["sau_khi_tra"]),
                   ("ghép được Clear bù trừ (Δ = 60 phút)", huy["ghep_duoc"]), ("không ghép được", ~huy["ghep_duoc"])]:
        p(f"  {ten}: {int(m.sum()):,} lần, do batch {pct(int((m & huy['batch']).sum()), int(m.sum()))}")
    if "user" in huy:
        p("  Tài khoản batch hủy nhiều nhất: " + ", ".join(f"{k} {v:,}" for k, v in huy.loc[huy["batch"], "user"].value_counts().head(5).items()))
        p("  Giờ (địa phương) của lần hủy do batch: " + ", ".join(
            f"{k}h {v:,}" for k, v in huy.loc[huy["batch"], "ts_dp"].dt.hour.value_counts().head(5).items()))
        p("  Thứ (địa phương) của lần hủy do batch: " + ", ".join(
            f"{k} {v:,}" for k, v in huy.loc[huy["batch"], "ts_dp"].dt.day_name().value_counts().items()))
    # (iv) làm lại: hủy rồi ghi lại hóa đơn (có Record Invoice Receipt sau lần hủy, cùng dòng hàng)
    ir_cuoi = ev[ev["act"] == IR].groupby("case")["ts"].max()
    huy["ghi_lai"] = huy["ts"] < huy["case"].map(ir_cuoi)
    lam_lai_tat_ca = set(huy.loc[huy["ghi_lai"], "case"])
    lam_lai_nguoi = set(huy.loc[huy["ghi_lai"] & ~huy["batch"], "case"])
    p(f"(iv) Làm lại (hủy rồi ghi lại hóa đơn): {pct(len(lam_lai_tat_ca), N)} dòng hàng tính mọi lần hủy; "
      f"{pct(len(lam_lai_nguoi), N)} dòng hàng chỉ tính lần hủy do người dùng thật")
    p(f"Danh sách chỉ báo phụ: (i) hủy sau thanh toán {huy.loc[huy['sau_khi_tra'], 'case'].nunique():,} dòng hàng; "
      f"(ii) #Cancel IR > #IR {int((d['n_cancel'] > d['n_ir']).sum()):,}, #Cancel Sub > #Record Sub {int((d['n_cancel_sub'] > d['n_sub']).sum()):,}; "
      f"(iii) trả trước khi ghi nhận hóa đơn {len(tra_truoc):,}; (iv) làm lại {len(lam_lai_nguoi):,} (người dùng thật) / {len(lam_lai_tat_ca):,} (tất cả)")
    huy.drop(columns=["ts_dp"]).to_csv(ra / "g4_5_moi_lan_huy.csv", index=False, encoding="utf-8-sig")

    p("\nNhịp thanh toán theo giờ địa phương:")
    if "user" in ev:
        c = ev.loc[tt]
        u2 = c[c["user"] == "user_002"]
        p(f"  Clear thanh toán do user_002: {pct(len(u2), len(c))}; theo thứ: "
          + ", ".join(f"{k} {v:,}" for k, v in u2["ts_dp"].dt.day_name().value_counts().items()))
        p("  Theo giờ (5 giờ nhiều nhất): " + ", ".join(f"{k}h {v:,}" for k, v in u2["ts_dp"].dt.hour.value_counts().head(5).items()))
        so_ngay = u2["ts_dp"].dt.normalize().nunique()
        p(f"  user_002 làm Clear trong {so_ngay:,} ngày khác nhau; trung bình {len(u2) / max(1, so_ngay):,.0f} Clear mỗi ngày có chạy")
        p("  So sánh tất cả Clear thanh toán theo thứ: " + ", ".join(f"{k} {v:,}" for k, v in c["ts_dp"].dt.day_name().value_counts().items()))
    return dict(s22=len(s22))


# ---------------------------------------------------------------- tự kiểm
def tu_kiem():
    t = pd.Timestamp("2018-03-01 08:00")
    m, ngay = pd.Timedelta(minutes=1), pd.Timedelta(days=1)
    PO, GR = r5.PO, r5.GR
    vet = {
        "A_cung_ts": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay), (CLEAR, 30 * ngay)],
        "B_tra_trung": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay), (CLEAR, 40 * ngay)],
        "C_tra_truoc_hd": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 10 * ngay), (CLEAR, 20 * ngay), (IR, 25 * ngay)],
        "D_r5a": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CANCEL_IR, 5 * ngay), (CLEAR, 12 * ngay)],
        "E_memo_rieng": [(PO, 0 * m), (GR, ngay), (MEMO, 2 * ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay), (CLEAR, 35 * ngay)],
        # cùng timestamp trước, rồi một lần trả riêng: v2.1 bắt, đếm theo thời điểm không bắt (nhóm chênh lệch 23 case)
        "F_cung_ts_truoc": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (IR, 3 * ngay), (CLEAR, 30 * ngay), (CLEAR, 30 * ngay),
                            (CLEAR, 40 * ngay)],
        # bù trừ thủ công 3 giờ sau khi hủy, cùng người: chỉ bù trừ ở độ nhạy 1.000 phút
        "G_bu_tru_3h": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CANCEL_IR, 5 * ngay), (CLEAR, 5 * ngay + 180 * m)],
    }
    df = pd.DataFrame([{"case": c, "act": a, "ts": t + dt, "user": "user_1"} for c, ds in vet.items() for a, dt in ds])
    ev = sap_xep(df)
    loai, _, _ = phan_loai_clear(ev, DELTA)
    bd = dem_r5(ev, loai)
    s22, nhanh, tra_truoc = r5_v22(bd)
    loai_mr = phan_loai_clear_mo_rong(ev, DELTA, DELTA_MO_RONG)
    s22_mr, _, _ = r5_v22(dem_r5(ev, loai_mr))
    s22_km, _, _ = r5_v22(dem_r5(ev, loai, tinh_memo=False))
    kq = {
        "R5 v2.1": sorted(set(bd.loc[bd["vuot"], "case"])),
        "R5 đếm theo thời điểm": sorted(set(bd.loc[bd["vuot_tg"], "case"])),
        "R5 v2.2": sorted(s22),
        "R5a": sorted(nhanh.index[nhanh == "R5a"]),
        "trả trước khi ghi nhận hóa đơn": sorted(tra_truoc),
        "R5 v2.2, bù trừ mở rộng": sorted(s22_mr),
        "R5 v2.2, không tính memo": sorted(s22_km),
    }
    ky_vong = {
        "R5 v2.1": ["A_cung_ts", "B_tra_trung", "C_tra_truoc_hd", "D_r5a", "F_cung_ts_truoc", "G_bu_tru_3h"],
        "R5 đếm theo thời điểm": ["B_tra_trung", "C_tra_truoc_hd", "D_r5a", "G_bu_tru_3h"],
        "R5 v2.2": ["B_tra_trung", "D_r5a", "G_bu_tru_3h"],
        "R5a": ["D_r5a", "G_bu_tru_3h"],
        "trả trước khi ghi nhận hóa đơn": ["C_tra_truoc_hd"],
        "R5 v2.2, bù trừ mở rộng": ["B_tra_trung", "D_r5a"],
        "R5 v2.2, không tính memo": ["B_tra_trung", "D_r5a", "E_memo_rieng", "G_bu_tru_3h"],
    }
    # k của Mục 2: E_memo_rieng có 1 IR, 1 memo, 2 Clear → k = 1 = số memo
    d = r5.dem_theo_case(ev, set(ev["case"]), loai)
    k_e = int(d.at["E_memo_rieng", "n_clear_tt"] - max(0, d.at["E_memo_rieng", "n_ir"] - d.at["E_memo_rieng", "n_cancel"]))
    loi = 0
    for k in kq:
        dat = kq[k] == ky_vong[k]
        loi += not dat
        print(f"{k:32s} {kq[k]}  {'ĐẠT' if dat else 'SAI, kỳ vọng ' + str(ky_vong[k])}")
    dat = k_e == 1
    loi += not dat
    print(f"{'k của E_memo_rieng':32s} {k_e}  {'ĐẠT' if dat else 'SAI, kỳ vọng 1'}")
    print("\nTất cả đều đạt." if loi == 0 else f"\n{loi} kiểm tra SAI.")
    return loi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(cau_hinh.LOG), help="đường dẫn log; mặc định trong cau_hinh.py")
    ap.add_argument("--cong-ty", default="companyID_0000")
    ap.add_argument("--ra", default="ket_qua_g4")
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
    (ra / "tom_tat_g4.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Đã chạy xong.")


if __name__ == "__main__":
    main()
