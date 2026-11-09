"""SQ3: các dấu hiệu bất thường liên quan thế nào đến thời gian thanh toán (thiết kế Mục 3.7.3, ghi ngày 23/10/2026).

Thời gian = Create Purchase Order Item đầu tiên → Clear Invoice "thanh toán" cuối cùng (Δ = 60 phút). Dòng hàng còn mở bị
kiểm duyệt phải tại thời điểm trích xuất; dòng hàng bị xóa bị loại và được đếm. Thời gian tính theo ngày nguyên (làm tròn
lên, nhỏ nhất 1 ngày) vì thanh toán chạy theo đợt hằng tuần; nhờ vậy mô hình Cox có biến thay đổi theo thời gian chạy nhanh.

  1. Mô tả: Kaplan-Meier toàn bộ và theo nhánh, trung vị, P90; thời gian chờ theo giai đoạn tách nhánh (a)/(b);
     tỷ lệ Clear thanh toán theo thứ trong tuần.
  2. Cox có biến thay đổi theo thời gian (R2a, R3, làm lại) và biến ban đầu (R6, biến thể phụ của R7), có biến kiểm soát;
     HR, KTC 95%, p hiệu chỉnh Holm trên 5 luật. Kiểm tra độ vững: R6 thay đổi theo thời gian tại lần nhận hàng do
     chính người tạo dòng đơn hàng thực hiện.
  3. Phân tích mốc ngày 30 cho R2a, R3, làm lại (Kaplan-Meier, log-rank, Cox); log-rank toàn bộ dòng hàng cho R6, R7b.
     Kiểm tra nguy cơ tỷ lệ bằng phần dư Schoenfeld; RMST đến 180 ngày (mốc ngày 30: 150 ngày còn lại).
  4. Ba phân tích độ nhạy: trạng thái theo định nghĩa có debit memo; loại dòng còn mở không hoạt động > 180 ngày;
     loại dòng có Cancel Invoice Receipt hoặc Cancel Goods Receipt.
  Hình Kaplan-Meier theo nhóm cuối cùng chỉ để minh họa (có thiên lệch thời gian bất tử), không dùng để kiểm định.

Cần lifelines (python -m pip install lifelines); matplotlib để vẽ hình (thiếu thì bỏ qua hình).
Cách chạy (cùng thư mục với sq2_tuan_thu.py, kiem_chung_bo_sung.py và các file đi kèm):
    python src/sq3_thoi_gian.py --tu-kiem
    python src/sq3_thoi_gian.py
Kết quả ghi vào ket_qua_sq3/; gửi lại tom_tat_sq3.txt.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import g3_kiem_tra_bo_sung as g3
import g4_kiem_tra as g4
import g5_kiem_tra as g5
import kiem_chung_cuoi as kc
import sq1_kham_pha as sq1
import sq2_bo_sung as sq2b
import sq2_tuan_thu as sq2
from kiem_chung_bo_sung import holm
from r5_v2_phan_loai_clear import (CANCEL_GR, CANCEL_IR, GR, IR, PO, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear,
                                   sap_xep)
import cau_hinh

MOC_LANDMARK = 30
RMST_NGAY = 180
NGAY_NGHI = 180
SO_HANG_MUC = 8  # số hạng mục chi tiêu giữ riêng; còn lại gộp "Khác"
LUAT_TV = ["R2a", "R3", "lam_lai"]
LUAT_CO_DINH = ["R6", "R7b"]
LUAT_SQ3 = LUAT_TV + LUAT_CO_DINH
TEN = {"R2a": "R2a", "R3": "R3", "lam_lai": "Làm lại", "R6": "R6", "R7b": "R7 (biến thể phụ)",
       "R6_tv": "R6 (thay đổi theo thời gian)"}
KY_VONG = {"R2a": 508, "R3": 2008, "R6": 650, "R7b": 2430, "lam_lai": 2982}  # trên toàn phạm vi 210.887 dòng hàng
MOC_KM = [30, 60, 90, 120, 180, 365]
MAU = ["#2a78d6", "#eb6834", "#1baf7a"]  # xanh, cam, xanh ngọc
MOT_NGAY = pd.Timedelta(days=1)
vn = sq2b.vn


def pct(n, N, so_le=2):
    return f"{vn(int(n))} ({vn(n / N, f'.{so_le}%')})" if N else f"{vn(int(n))} (–)"


def so(x, k=1):
    return "–" if x is None or not np.isfinite(x) else vn(float(x), f",.{k}f")


def so_p(x):
    if x is None or not np.isfinite(x):
        return "–"
    return "< 0,001" if x < 0.001 else vn(float(x), ".3f")


# ---------------------------------------------------------------- 1. thời điểm dấu hiệu
def thoi_diem_dau_hieu(ev, kq):
    """Thời điểm dấu hiệu đầu tiên của mỗi dòng hàng (Series theo case, chỉ dòng có dấu hiệu)."""
    s = {}
    rpb1 = kc.dau(ev, kc.RPB)
    s["R2a"] = rpb1[rpb1.index.isin(kq["R2a"][0])]  # R2a xét lần gỡ khóa đầu tiên
    ir1 = kc.dau(ev, IR)
    cp = ev.loc[ev["act"] == kc.CHANGE_PRICE, ["case", "ts"]]
    cp = cp[cp["ts"] > cp["case"].map(ir1)]
    s["R3"] = cp.groupby("case")["ts"].min()
    # làm lại: lần ghi nhận hóa đơn đầu tiên sau lần hủy đầu tiên do người dùng thật (như g4: chỉ loại tài khoản batch)
    huy = ev[(ev["act"] == CANCEL_IR) & ~g3.la_batch(ev["user"])]
    c0 = huy.groupby("case")["ts"].min()
    ir = ev.loc[(ev["act"] == IR) & ev["case"].isin(c0.index), ["case", "ts"]]
    ir = ir[ir["ts"] > ir["case"].map(c0)]
    s["lam_lai"] = ir.groupby("case")["ts"].min()
    # R6 thay đổi theo thời gian: lần nhận hàng đầu tiên do chính người (thật) tạo dòng đơn hàng thực hiện
    u = ev[ev["act"].isin([PO, GR]) & kc.la_nguoi_that(ev["user"])]
    nguoi_po = u.loc[u["act"] == PO, ["case", "user"]].drop_duplicates()
    gr = u.loc[u["act"] == GR, ["case", "user", "ts"]].merge(nguoi_po, on=["case", "user"])
    s["R6"] = gr.groupby("case")["ts"].min()
    return s


def ngay_chia(s, t):
    """Ngày bắt đầu phơi nhiễm trên thang ngày nguyên. s, t: số ngày (thực) từ khi tạo dòng đơn hàng đến dấu hiệu và đến
    thanh toán/kiểm duyệt. Không có dấu hiệu hoặc dấu hiệu từ thời điểm t trở đi → inf (không phơi nhiễm trong thời gian
    theo dõi). Còn lại → floor(s), nhỏ nhất 0 (0 = phơi nhiễm từ đầu). Khi s < t thì floor(s) < ceil(t), nên sự kiện
    luôn rơi vào đoạn đã phơi nhiễm."""
    s, t = np.asarray(s, dtype=float), np.asarray(t, dtype=float)
    S = np.floor(s).clip(min=0)
    return np.where(np.isnan(s) | (s >= t), np.inf, S)


def dang_dai(x, luat_tv):
    """Chia mỗi dòng hàng thành các đoạn (start, stop] tại ngày bắt đầu phơi nhiễm của các luật thay đổi theo thời gian.
    x: có cột id, T, E, S_<luật> và các biến ban đầu. Sự kiện chỉ ở đoạn cuối."""
    diem = [x[["id"]].assign(d=0.0), x[["id"]].assign(d=x["T"].astype(float))]
    for l in luat_tv:
        m = (x["S_" + l] > 0) & (x["S_" + l] < x["T"])
        diem.append(x.loc[m, ["id"]].assign(d=x.loc[m, "S_" + l].astype(float)))
    dm = pd.concat(diem, ignore_index=True).drop_duplicates().sort_values(["id", "d"])
    dm["stop"] = dm.groupby("id")["d"].shift(-1)
    seg = dm.dropna(subset=["stop"]).rename(columns={"d": "start"})
    out = seg.merge(x, on="id", how="left")
    out["event"] = ((out["stop"] == out["T"]) & (out["E"] == 1)).astype(int)
    for l in luat_tv:
        out[l] = (out["S_" + l] <= out["start"]).astype(float)
    return out.drop(columns=["T", "E"] + [c for c in out.columns if c.startswith("S_")]).reset_index(drop=True)


# ---------------------------------------------------------------- 2. bảng dòng hàng
def chuan_bi(ev, pv, moc, p):
    """Bảng theo dòng hàng (toàn phạm vi): mốc thời gian, trạng thái hai định nghĩa, dấu hiệu, biến kiểm soát."""
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    ts_dp = g4.gio_dia_phuong(ev, p)
    kq, _ = kc.cac_luat(ev, pv, loai, ts_dp)
    so_lieu = g5.so_lieu_case(ev, pv, loai)
    tt = g5.trang_thai_cac_dinh_nghia(so_lieu)
    idx = so_lieu.index
    d = so_lieu[["dau_po", "dau_clear_tt", "cuoi_clear_tt", "n_ir", "n_cancel"]].copy()
    d["tt_moi"], d["tt_cu"] = tt["moi"], tt["cu"]
    tg = thoi_diem_dau_hieu(ev, kq)
    for l in LUAT_SQ3:
        d["co_" + l] = idx.isin(tg[l].index) if l in ("lam_lai",) else idx.isin(kq[l][0])
    for l in LUAT_TV + ["R6"]:
        d["s_" + l] = (tg[l].reindex(idx) - d["dau_po"]) / MOT_NGAY
    # đối chiếu số dòng hàng
    lech = {l: (int(d["co_" + l].sum()), v) for l, v in KY_VONG.items() if int(d["co_" + l].sum()) != v}
    p("Số dòng hàng theo luật khớp kiểm chứng (R2a 508, R3 2.008, R6 650, R7b 2.430, làm lại 2.982)."
      if not lech else f"!!! Lệch số dòng hàng (tính được, kỳ vọng): {lech}")
    lech_tg = [l for l in LUAT_TV + ["R6"] if set(tg[l].index) != set(d.index[d["co_" + l]])]
    p("Mọi dòng hàng bị đánh dấu đều có thời điểm dấu hiệu." if not lech_tg else f"!!! Thiếu thời điểm dấu hiệu: {lech_tg}")
    # nhánh
    _, nh = sq1.nhanh(ev, pv)
    d["nhanh"] = nh["nhanh"].reindex(idx)
    d["nhanh3"] = np.select([d["nhanh"] == sq1.NHOM_NHANH[0], d["nhanh"] == sq1.NHOM_NHANH[1]], ["(a)", "(b)"], "khác")
    d["gr1"], d["ir1"], d["rpb1"] = nh["gr_dau"], nh["ir_dau"], kc.dau(ev, kc.RPB).reindex(idx)
    # biến kiểm soát
    n_gr = ev[ev["act"] == GR].groupby("case").size().reindex(idx).fillna(0)
    d["nhieu_gr_ir"] = ((n_gr > 1) | (d["n_ir"] > 1)).astype(float)
    v = sq2.gia_tri_dong_hang(ev, idx)
    d["gia_tri_0"] = (~(v > 0)).astype(float)
    d["log_gia_tri"] = np.log(v.where(v > 0)).fillna(0.0)
    dau_case = ev.drop_duplicates("case").set_index("case").reindex(idx)
    vendor = dau_case["vendor"].astype(str) if "vendor" in dau_case else pd.Series("?", index=idx)
    d["log_quy_mo_ncc"] = np.log(vendor.map(vendor.value_counts()).astype(float))
    hm = dau_case["spend_area"].astype(str) if "spend_area" in dau_case else pd.Series("?", index=idx)
    top = hm.value_counts().index[:SO_HANG_MUC]
    d["hang_muc"] = hm.where(hm.isin(top), "Khác")
    d["co_huy"] = idx.isin(set(ev.loc[ev["act"].isin([CANCEL_IR, CANCEL_GR]), "case"]))
    d["ngay_nghi"] = (pd.Timestamp(moc) - ev.groupby("case")["ts"].max().reindex(idx)).dt.days
    tt_ev = ev.loc[loai.index[loai == "thanh_toan"]]
    thu = ts_dp.loc[tt_ev.index].dt.dayofweek
    return d, thu, top


def bien_kiem_soat(d, top):
    """Biến giả và biến liên tục (đã trừ trung bình) dùng chung cho mọi mô hình Cox."""
    x = pd.DataFrame(index=d.index)
    x["nhanh_b"] = (d["nhanh3"] == "(b)").astype(float)
    x["nhanh_khac"] = (d["nhanh3"] == "khác").astype(float)
    ten_hm = {}
    for i, h in enumerate(list(top[1:]) + ["Khác"], 1):  # hạng mục lớn nhất làm nhóm tham chiếu
        c = "hm_khac" if h == "Khác" else f"hm_{i}"
        x[c] = (d["hang_muc"] == h).astype(float)
        ten_hm[c] = h if str(h).strip() else "(không ghi hạng mục)"
    x["nhieu_gr_ir"] = d["nhieu_gr_ir"]
    x["log_gia_tri"] = d["log_gia_tri"] - d["log_gia_tri"].mean()
    x["gia_tri_0"] = d["gia_tri_0"]
    x["log_quy_mo_ncc"] = d["log_quy_mo_ncc"] - d["log_quy_mo_ncc"].mean()
    return x, ten_hm


def tap_phan_tich(d, cot_tt, moc):
    """Loại dòng hàng bị xóa; t = số ngày (thực), T = ceil(t) ≥ 1, E = 1 nếu hoàn tất. Trả về (bảng, số dòng bị loại)."""
    tt = d[cot_tt]
    ht, mo = tt == "Hoàn tất", tt == "Còn mở"
    t = pd.Series(np.nan, index=d.index)
    t[ht] = (d.loc[ht, "cuoi_clear_tt"] - d.loc[ht, "dau_po"]) / MOT_NGAY
    t[mo] = (pd.Timestamp(moc) - d.loc[mo, "dau_po"]) / MOT_NGAY
    loai = {"bị xóa": int((tt == "Bị xóa").sum()), "thời gian ≤ 0 hoặc thiếu": int(((ht | mo) & ~(t > 0)).sum())}
    giu = (ht | mo) & (t > 0)
    x = d.loc[giu].copy()
    x["t"] = t[giu]
    x["T"] = np.ceil(x["t"]).clip(lower=1)
    x["E"] = ht[giu].astype(int)
    for l in LUAT_TV + ["R6"]:
        x["S_" + l] = ngay_chia(x["s_" + l], x["t"])
    x["id"] = np.arange(len(x))
    return x, loai


# ---------------------------------------------------------------- 3. mô hình
def fit_ctv(dl, cot):
    from lifelines import CoxTimeVaryingFitter
    hang = [c for c in cot if dl[c].nunique() > 1]  # bỏ biến hằng (ví dụ làm lại khi đã loại dòng có hủy)
    du_lieu = dl[["id", "start", "stop", "event"] + hang]
    for pen in (0.0, 1e-5):
        try:
            m = CoxTimeVaryingFitter(penalizer=pen)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(du_lieu, id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=False)
            return m, [c for c in cot if c not in hang], pen
        except Exception as e:  # hội tụ kém: thử lại với phạt rất nhỏ
            loi = e
    raise loi


def fit_cph(df, cot, thoi_gian="T", su_kien="E"):
    from lifelines import CoxPHFitter
    hang = [c for c in cot if df[c].nunique() > 1]
    m = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df[[thoi_gian, su_kien] + hang], duration_col=thoi_gian, event_col=su_kien)
    return m, hang


def bang_hr(m, ten_mo_hinh):
    s = m.summary
    return pd.DataFrame({"mo_hinh": ten_mo_hinh, "bien": s.index, "HR": s["exp(coef)"].values,
                         "KTC_duoi": s["exp(coef) lower 95%"].values, "KTC_tren": s["exp(coef) upper 95%"].values,
                         "p": s["p"].values})


def in_hr(h, p, luat=LUAT_SQ3, dat_ten=None):
    h = h.set_index("bien")
    for l in luat:
        if l not in h.index:
            p(f"  {TEN.get(l, l):24s} không ước lượng được (biến hằng trong tập này)")
            continue
        r = h.loc[l]
        them = f"; p Holm {so_p(r['p_holm'])}" if "p_holm" in r and pd.notna(r["p_holm"]) else ""
        p(f"  {(dat_ten or TEN).get(l, l):24s} HR {so(r['HR'], 3)} (KTC 95%: {so(r['KTC_duoi'], 3)}–{so(r['KTC_tren'], 3)}); "
          f"p {so_p(r['p'])}{them}")


def km(t, e):
    from lifelines import KaplanMeierFitter
    return KaplanMeierFitter().fit(t, e)


def phan_vi_km(kmf, q):
    """Thời điểm đầu tiên tỷ lệ chưa thanh toán ≤ 1 − q (q = 0,5: trung vị; q = 0,9: P90)."""
    sf = kmf.survival_function_.iloc[:, 0]
    duoi = sf[sf <= 1 - q]
    return float(duoi.index[0]) if len(duoi) else float("nan")


def rmst(t, e, h):
    """RMST đến h ngày từ đường Kaplan-Meier và phương sai kiểu Greenwood (Klein & Moeschberger, 2003, mục 4.5):
    Var = Σ_{t_j ≤ h} A_j² · d_j / (n_j (n_j − d_j)), A_j = diện tích dưới S(t) từ t_j đến h."""
    t, e = np.asarray(t, dtype=float), np.asarray(e, dtype=int)
    if len(t) == 0:
        return float("nan"), float("nan")
    tj, nguoc = np.unique(t, return_inverse=True)
    d = np.bincount(nguoc, weights=e)
    c = np.bincount(nguoc)
    n = len(t) - np.concatenate([[0], np.cumsum(c)[:-1]])
    S = np.cumprod(1 - d / n)
    m = tj <= h
    tj, d, n, S = tj[m], d[m], n[m], S[m]
    moc = np.concatenate([[0.0], tj, [h]])
    S_doan = np.concatenate([[1.0], S])  # S trên [moc_k, moc_{k+1})
    dien_tich = S_doan * np.diff(moc)
    A = np.cumsum(dien_tich[::-1])[::-1][1:]  # diện tích từ t_j đến h
    with np.errstate(divide="ignore", invalid="ignore"):
        hang = np.where((d > 0) & (n > d), A ** 2 * d / (n * (n - d)), 0.0)
    return float(dien_tich.sum()), float(hang.sum())


def so_sanh_hai_nhom(t, e, nhom, h):
    """Log-rank và chênh lệch RMST (có dấu hiệu − không) đến h ngày."""
    from lifelines.statistics import logrank_test
    a, b = nhom.astype(bool), ~nhom.astype(bool)
    lr = logrank_test(t[a], t[b], e[a], e[b])
    r1, v1 = rmst(t[a], e[a], h)
    r0, v0 = rmst(t[b], e[b], h)
    se = np.sqrt(v1 + v0)
    k1, k0 = km(t[a], e[a]), km(t[b], e[b])
    return {"n_co": int(a.sum()), "n_khong": int(b.sum()), "trung_vi_co": phan_vi_km(k1, 0.5),
            "trung_vi_khong": phan_vi_km(k0, 0.5), "log_rank_chi2": lr.test_statistic, "log_rank_p": lr.p_value,
            "rmst_co": r1, "rmst_khong": r0, "rmst_chenh": r1 - r0, "rmst_duoi": r1 - r0 - 1.96 * se,
            "rmst_tren": r1 - r0 + 1.96 * se}


def schoenfeld(m, df, cot):
    from lifelines.statistics import proportional_hazard_test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = proportional_hazard_test(m, df, time_transform="rank").summary
    return r.reindex([c for c in cot if c in r.index])[["test_statistic", "p"]]


# ---------------------------------------------------------------- 4. hình
def nap_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def ve_duong(ax, t, e, nhan, mau, ls="-"):
    k = km(t, e)
    sf = k.survival_function_.iloc[:, 0]
    sf = sf[sf.index <= 365]
    ax.step(sf.index, 1 - sf.values, where="post", color=mau, lw=2, ls=ls, label=nhan)


def trang_tri(ax, tieu_de, xmax=365, nhan_x="số ngày từ khi tạo dòng đơn hàng"):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 30, 60, 90, 120, 180, 270, 365] if xmax == 365 else list(range(0, xmax + 1, 30)))
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(nhan_x)
    ax.set_ylabel("tỷ lệ đã thanh toán xong")
    ax.set_title(tieu_de, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")


def ve_hinh(x, lm, ra, p):
    """Hình Kaplan-Meier; lm = None thì bỏ hình mốc ngày 30."""
    plt = nap_plt()
    if plt is None:
        p("(Không có matplotlib, bỏ qua hình.)")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ve_duong(ax, x["T"], x["E"], f"toàn bộ (n = {vn(len(x))})", "#52514e")
    for nh, mau in zip(["(a)", "(b)", "khác"], MAU):
        m = x["nhanh3"] == nh
        ve_duong(ax, x.loc[m, "T"], x.loc[m, "E"], f"{'nhánh ' + nh if nh != 'khác' else 'nhóm khác'} (n = {vn(int(m.sum()))})", mau, "--")
    trang_tri(ax, "Kaplan-Meier: tỷ lệ dòng hàng đã thanh toán xong theo thời gian")
    fig.tight_layout()
    fig.savefig(ra / "km_toan_bo_theo_nhanh.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for ax, l in zip(axs, LUAT_SQ3):
        m = x["co_" + l]
        ve_duong(ax, x.loc[~m, "T"], x.loc[~m, "E"], f"không (n = {vn(int((~m).sum()))})", MAU[0])
        ve_duong(ax, x.loc[m, "T"], x.loc[m, "E"], f"có (n = {vn(int(m.sum()))})", MAU[1])
        trang_tri(ax, TEN[l] + (" (minh họa, nhóm cuối)" if l in LUAT_TV else ""))
    fig.tight_layout()
    fig.savefig(ra / "km_minh_hoa_theo_luat.png", dpi=150)
    plt.close(fig)

    if lm is None:
        p("Đã vẽ km_toan_bo_theo_nhanh.png, km_minh_hoa_theo_luat.png.")
        return
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, l in zip(axs, LUAT_TV):
        m = lm[l + "_30"].astype(bool)
        ve_duong(ax, lm.loc[~m, "T_lm"], lm.loc[~m, "E"], f"chưa có đến ngày 30 (n = {vn(int((~m).sum()))})", MAU[0])
        ve_duong(ax, lm.loc[m, "T_lm"], lm.loc[m, "E"], f"đã có đến ngày 30 (n = {vn(int(m.sum()))})", MAU[1])
        trang_tri(ax, f"{TEN[l]}: mốc ngày {MOC_LANDMARK}", xmax=RMST_NGAY - MOC_LANDMARK,
                  nhan_x=f"số ngày sau ngày {MOC_LANDMARK}")
    fig.tight_layout()
    fig.savefig(ra / "km_moc_ngay_30.png", dpi=150)
    plt.close(fig)
    p("Đã vẽ km_toan_bo_theo_nhanh.png, km_minh_hoa_theo_luat.png, km_moc_ngay_30.png.")


# ---------------------------------------------------------------- 5. phân tích
def mo_ta(d, x, thu, moc, ra, p):
    p("\n=== 1. Mô tả ===")
    g = g5.thoi_gian(d, d["tt_moi"], moc)
    p(f"Kiểm tra với Mục 3.2.2 (số ngày thực): hoàn tất {vn(g['n_ht'])}, còn mở {vn(g['n_mo'])}; trung vị {so(g['trung_vi'])}, "
      f"P90 {so(g['p90'])}, trung vị Kaplan-Meier {so(g['km'])} ngày (kỳ vọng 163.619; 39.988; 77,4; 127,2; 84,6).")
    k = km(x["T"], x["E"])
    p(f"Trên thang ngày nguyên: trung vị Kaplan-Meier {so(phan_vi_km(k, 0.5), 0)} ngày, P90 Kaplan-Meier "
      f"{so(phan_vi_km(k, 0.9), 0)} ngày.")
    sf = k.survival_function_.iloc[:, 0]
    rows = [{"ngay": n, "da_thanh_toan": 1 - float(sf[sf.index <= n].iloc[-1])} for n in MOC_KM]
    p("Tỷ lệ đã thanh toán xong (Kaplan-Meier): " + "; ".join(f"ngày {r['ngay']}: {vn(r['da_thanh_toan'], '.1%')}" for r in rows))
    pd.DataFrame(rows).to_csv(ra / "km_tong.csv", index=False, encoding="utf-8-sig")
    ht = x[x["E"] == 1]
    dau = (ht["dau_clear_tt"] - ht["dau_po"]) / MOT_NGAY
    p(f"Dòng hoàn tất: đến Clear thanh toán đầu tiên trung vị {so(dau.median())} ngày, đến Clear thanh toán cuối cùng "
      f"{so(ht['t'].median())} ngày; {pct(int((ht['dau_clear_tt'] < ht['cuoi_clear_tt']).sum()), len(ht))} dòng có hơn một "
      f"thời điểm thanh toán.")

    p("\nTheo nhánh:")
    rows = []
    for nh in ["(a)", "(b)", "khác"]:
        m = x["nhanh3"] == nh
        kk = km(x.loc[m, "T"], x.loc[m, "E"])
        r = {"nhanh": nh, "so_dong": int(m.sum()), "hoan_tat": int(x.loc[m, "E"].sum()),
             "trung_vi_km": phan_vi_km(kk, 0.5), "p90_km": phan_vi_km(kk, 0.9),
             "trung_vi_hoan_tat": x.loc[m & (x["E"] == 1), "t"].median()}
        rows.append(r)
        p(f"  nhánh {nh}: {vn(r['so_dong'])} dòng, hoàn tất {pct(r['hoan_tat'], r['so_dong'])}; trung vị KM "
          f"{so(r['trung_vi_km'], 0)}, P90 KM {so(r['p90_km'], 0)} ngày; trung vị dòng hoàn tất {so(r['trung_vi_hoan_tat'])} ngày")
    pd.DataFrame(rows).to_csv(ra / "km_theo_nhanh.csv", index=False, encoding="utf-8-sig")

    p("\nThời gian chờ theo giai đoạn, dòng hoàn tất (ngày; trung vị [P25–P75]):")
    gd = {"tạo đơn hàng → nhận hàng đầu tiên": ("dau_po", "gr1"),
          "ghi nhận hóa đơn đầu tiên → gỡ khóa đầu tiên": ("ir1", "rpb1"),
          "gỡ khóa đầu tiên → thanh toán cuối cùng": ("rpb1", "cuoi_clear_tt"),
          "ghi nhận hóa đơn đầu tiên → thanh toán cuối cùng": ("ir1", "cuoi_clear_tt")}
    rows = []
    for nh in ["(a)", "(b)"]:
        h = ht[ht["nhanh3"] == nh]
        for ten, (a, b) in gd.items():
            k_ = ((h[b] - h[a]) / MOT_NGAY).dropna()
            r = {"nhanh": nh, "giai_doan": ten, "n": len(k_), "trung_vi": k_.median(), "p25": k_.quantile(0.25),
                 "p75": k_.quantile(0.75), "so_am": int((k_ < 0).sum())}
            rows.append(r)
            p(f"  {nh} {ten}: n = {vn(r['n'])}; {so(r['trung_vi'])} [{so(r['p25'])}–{so(r['p75'])}]"
              + (f"; âm {vn(r['so_am'])}" if r["so_am"] else ""))
    pd.DataFrame(rows).to_csv(ra / "cho_theo_giai_doan.csv", index=False, encoding="utf-8-sig")

    ten_thu = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]
    dem = thu.value_counts().reindex(range(7)).fillna(0).astype(int)
    p("\nClear Invoice thanh toán theo thứ (giờ địa phương): "
      + "; ".join(f"{ten_thu[i]} {pct(dem[i], dem.sum(), 1)}" for i in range(7)))
    pd.DataFrame({"thu": ten_thu, "so_clear_thanh_toan": dem.values}).to_csv(ra / "clear_theo_thu.csv", index=False,
                                                                             encoding="utf-8-sig")


def dem_phoi_nhiem(x, p):
    p("\nThời điểm dấu hiệu so với thời gian theo dõi (tập phân tích):")
    rows = []
    for l in LUAT_TV + ["R6"]:
        co = x["co_" + l]
        S = x["S_" + l]
        r = {"luat": l, "co_dau_hieu": int(co.sum()), "phoi_nhiem_tu_dau": int((co & (S == 0)).sum()),
             "phoi_nhiem_giua_chung": int((co & (S > 0) & np.isfinite(S)).sum()),
             "sau_khi_ket_thuc": int((co & ~np.isfinite(S)).sum()), "truoc_tao_don": int((co & (x["s_" + l] < 0)).sum()),
             "trung_vi_ngay": x.loc[co, "s_" + l].median()}
        rows.append(r)
        p(f"  {TEN[l] if l != 'R6' else 'R6 (nhận hàng do người tạo đơn)'}: {vn(r['co_dau_hieu'])} dòng; dấu hiệu sau trung vị "
          f"{so(r['trung_vi_ngay'])} ngày; phơi nhiễm từ ngày đầu {vn(r['phoi_nhiem_tu_dau'])}, giữa chừng "
          f"{vn(r['phoi_nhiem_giua_chung'])}, sau thanh toán/kiểm duyệt {vn(r['sau_khi_ket_thuc'])}"
          + (f", trước khi tạo dòng đơn hàng {vn(r['truoc_tao_don'])}" if r["truoc_tao_don"] else ""))
    return pd.DataFrame(rows)


def mo_hinh_tv(x, kiem_soat, ten, luat_tv=LUAT_TV, luat_cd=LUAT_CO_DINH):
    xx = x[["id", "T", "E"] + ["S_" + l for l in luat_tv]].join(kiem_soat)
    for l in luat_cd:
        xx[l] = x["co_" + l].astype(float)
    dl = dang_dai(xx.reset_index(drop=True), luat_tv)
    cot = list(luat_tv) + list(luat_cd) + list(kiem_soat.columns)
    m, bo, pen = fit_ctv(dl, cot)
    h = bang_hr(m, ten)
    h["ten_luat"] = h["bien"].map(lambda b: "R6" if b == "R6_tv" else b)
    return m, h, len(dl), bo, pen


def buoc_cox(x, ks_all, ten_hm, ra, p):
    """Mô hình chính và kiểm tra độ vững R6. Trả về bảng HR (mọi biến) của hai mô hình."""
    p("\n=== 2. Cox có biến thay đổi theo thời gian (R2a, R3, làm lại) và biến ban đầu (R6, R7b), có biến kiểm soát ===")
    p("HR < 1: thanh toán chậm hơn; HR > 1: nhanh hơn.")
    ks = ks_all.loc[x.index].set_index(x["id"].values)
    xi = x.set_index("id", drop=False)
    t = time.time()
    _, h, n_dong, bo, pen = mo_hinh_tv(xi, ks, "chinh")
    h_luat = h[h["bien"].isin(LUAT_SQ3)].copy()
    h_luat["p_holm"] = holm(h_luat["p"].values)
    h = h.merge(h_luat[["bien", "p_holm"]], on="bien", how="left")
    p(f"{vn(n_dong)} đoạn thời gian; {vn(int(x['E'].sum()))} sự kiện; chạy {vn(round(time.time() - t))} giây"
      + (f"; phạt {pen}" if pen else "") + (f"; bỏ biến hằng {bo}" if bo else "") + ".")
    in_hr(h, p)
    p("Biến kiểm soát:")
    in_hr(h, p, [c for c in ks.columns], {c: ten_hm.get(c, c) for c in ks.columns})
    p("\nĐộ vững: R6 thay đổi theo thời gian (tại lần nhận hàng do người tạo dòng đơn hàng thực hiện):")
    xr = xi.copy()
    xr["S_R6_tv"] = xr["S_R6"]
    _, h6, _, _, _ = mo_hinh_tv(xr, ks, "R6_thay_doi_theo_thoi_gian", LUAT_TV + ["R6_tv"], ["R7b"])
    in_hr(h6, p, ["R6_tv"], {"R6_tv": "R6 (thay đổi theo t.g.)"})
    hr = pd.concat([h, h6], ignore_index=True)
    hr.to_csv(ra / "hr_mo_hinh_chinh.csv", index=False, encoding="utf-8-sig")
    return hr


def buoc_moc(x, ks_all, ra, p):
    """Mốc ngày 30 (R2a, R3, làm lại) và toàn bộ dòng hàng (R6, R7b): log-rank, Cox, Schoenfeld, RMST."""
    p(f"\n=== 3. Mốc ngày {MOC_LANDMARK}, log-rank, giả định nguy cơ tỷ lệ, RMST ===")
    lm = x[x["t"] > MOC_LANDMARK].copy()
    lm["T_lm"] = lm["T"] - MOC_LANDMARK
    for l in LUAT_TV:
        lm[l + "_30"] = (lm["co_" + l] & (lm["s_" + l] <= MOC_LANDMARK)).astype(float)
    p(f"Tập mốc: {pct(len(lm), len(x))} dòng hàng chưa thanh toán xong và chưa bị kiểm duyệt sau ngày {MOC_LANDMARK}.")
    so_sanh = []
    for l in LUAT_TV:
        r = so_sanh_hai_nhom(lm["T_lm"], lm["E"], lm[l + "_30"], RMST_NGAY - MOC_LANDMARK)
        so_sanh.append({"luat": l, "tap": f"mốc ngày {MOC_LANDMARK}", "chan_tren_rmst": RMST_NGAY - MOC_LANDMARK, **r})
    for l in LUAT_CO_DINH:
        r = so_sanh_hai_nhom(x["T"], x["E"], x["co_" + l], RMST_NGAY)
        so_sanh.append({"luat": l, "tap": "toàn bộ", "chan_tren_rmst": RMST_NGAY, **r})
    ss = pd.DataFrame(so_sanh)
    ss["log_rank_p_holm"] = holm(ss["log_rank_p"].values)

    ks_lm = ks_all.loc[lm.index]
    lm_df = lm[["T_lm", "E"] + [l + "_30" for l in LUAT_TV]].join(ks_lm)
    for l in LUAT_CO_DINH:
        lm_df[l] = lm["co_" + l].astype(float)
    m_lm, hang_lm = fit_cph(lm_df, [l + "_30" for l in LUAT_TV] + LUAT_CO_DINH + list(ks_lm.columns), "T_lm")
    h_lm = bang_hr(m_lm, f"moc_ngay_{MOC_LANDMARK}")
    sch_lm = schoenfeld(m_lm, lm_df[["T_lm", "E"] + hang_lm], [l + "_30" for l in LUAT_TV])
    cs_df = x[["T", "E"]].join(ks_all.loc[x.index])
    for l in LUAT_CO_DINH:
        cs_df[l] = x["co_" + l].astype(float)
    m_cs, hang_cs = fit_cph(cs_df, LUAT_CO_DINH + list(ks_all.columns))
    h_cs = bang_hr(m_cs, "ban_dau_toan_bo")
    sch_cs = schoenfeld(m_cs, cs_df[["T", "E"] + hang_cs], LUAT_CO_DINH)
    sch = pd.concat([sch_lm.rename(index=lambda c: c.replace("_30", "")), sch_cs])
    ss["schoenfeld_chi2"] = ss["luat"].map(sch["test_statistic"])
    ss["schoenfeld_p"] = ss["luat"].map(sch["p"])
    ss["vi_pham_nguy_co_ty_le"] = ss["schoenfeld_p"] < 0.05
    h_lm["bien"] = h_lm["bien"].str.replace("_30", "")
    hl = pd.concat([h_lm, h_cs], ignore_index=True)
    c = hl.set_index(["mo_hinh", "bien"])
    for k in ["HR", "KTC_duoi", "KTC_tren"]:
        ss["cox_" + k] = [c[k].get((f"moc_ngay_{MOC_LANDMARK}" if l in LUAT_TV else "ban_dau_toan_bo", l), np.nan)
                          for l in ss["luat"]]
    for _, r in ss.iterrows():
        p(f"{TEN[r['luat']]} ({r['tap']}): có {vn(r['n_co'])}, không {vn(r['n_khong'])}; trung vị KM {so(r['trung_vi_co'], 0)} so với "
          f"{so(r['trung_vi_khong'], 0)} ngày; log-rank χ² = {so(r['log_rank_chi2'])}, p Holm {so_p(r['log_rank_p_holm'])}; "
          f"HR Cox {so(r['cox_HR'], 3)} ({so(r['cox_KTC_duoi'], 3)}–{so(r['cox_KTC_tren'], 3)})")
        p(f"    Schoenfeld χ² = {so(r['schoenfeld_chi2'])}, p {so_p(r['schoenfeld_p'])}"
          f"{' → vi phạm, báo cáo RMST' if r['vi_pham_nguy_co_ty_le'] else ''}; RMST đến {r['chan_tren_rmst']} ngày: "
          f"{so(r['rmst_co'])} so với {so(r['rmst_khong'])}, chênh {so(r['rmst_chenh'])} ngày "
          f"(KTC 95%: {so(r['rmst_duoi'])}–{so(r['rmst_tren'])}); chênh > 0 nghĩa là chậm hơn")
    ss.to_csv(ra / "so_sanh_moc_va_rmst.csv", index=False, encoding="utf-8-sig")
    hl.to_csv(ra / "hr_moc_va_ban_dau.csv", index=False, encoding="utf-8-sig")
    return ss, hl, lm


def buoc_do_nhay(d, x, ks_all, moc, ra, p, luu=lambda: None):
    p("\n=== 4. Độ nhạy (mô hình Cox có biến thay đổi theo thời gian, cùng biến kiểm soát) ===")
    x_cu, loai_cu = tap_phan_tich(d, "tt_cu", moc)
    nghi = (x["E"] == 0) & (x["ngay_nghi"] > NGAY_NGHI)
    cac_tap = [("(1) trạng thái theo định nghĩa có debit memo", "do_nhay_1_debit_memo", x_cu),
               (f"(2) loại {vn(int(nghi.sum()))} dòng còn mở không hoạt động > {NGAY_NGHI} ngày", "do_nhay_2_khong_hoat_dong",
                x[~nghi]),
               (f"(3) loại {vn(int(x['co_huy'].sum()))} dòng có Cancel Invoice Receipt hoặc Cancel Goods Receipt",
                "do_nhay_3_khong_huy", x[~x["co_huy"]])]
    hr = []
    for mo_ta_tap, ten, xs in cac_tap:
        xs = xs.copy()
        xs["id"] = np.arange(len(xs))
        k2 = ks_all.loc[xs.index].set_index(xs["id"].values)
        xs = xs.set_index("id", drop=False)
        _, hs, _, bo, _ = mo_hinh_tv(xs, k2, ten)
        p(f"{mo_ta_tap}: {vn(len(xs))} dòng, hoàn tất {vn(int(xs['E'].sum()))}" + (f"; bỏ biến hằng {bo}" if bo else ""))
        in_hr(hs, p)
        hr.append(hs)
        luu()
    p("(Định nghĩa có debit memo loại: " + ", ".join(f"{k} {vn(v)}" for k, v in loai_cu.items()) + ".)")
    hr = pd.concat(hr, ignore_index=True)
    hr.to_csv(ra / "hr_do_nhay.csv", index=False, encoding="utf-8-sig")
    return hr


def phan_tich(ev, pv, moc, ra, p, luu):
    t0 = time.time()
    p(f"Phạm vi {vn(len(pv))} dòng hàng, {vn(len(ev))} sự kiện. Δ = {g4.DELTA} phút. Thời điểm trích xuất: {moc}")
    d, thu, top = chuan_bi(ev, pv, moc, p)
    x, loai = tap_phan_tich(d, "tt_moi", moc)
    p(f"Tập phân tích: {vn(len(x))} dòng hàng (hoàn tất {vn(int(x['E'].sum()))}, còn mở bị kiểm duyệt "
      f"{vn(int((x['E'] == 0).sum()))}); loại " + ", ".join(f"{k} {vn(v)}" for k, v in loai.items()) + ".")
    ks_all, ten_hm = bien_kiem_soat(d, top)
    p("Hạng mục chi tiêu: tham chiếu = " + str(top[0]) + "; " + "; ".join(f"{c} = {h}" for c, h in ten_hm.items()))
    mo_ta(d, x, thu, moc, ra, p)
    dem_phoi_nhiem(x, p).to_csv(ra / "thoi_diem_dau_hieu.csv", index=False, encoding="utf-8-sig")
    luu()
    buoc_cox(x, ks_all, ten_hm, ra, p)
    luu()
    _, _, lm = buoc_moc(x, ks_all, ra, p)
    ve_hinh(x, lm, ra, p)
    luu()
    buoc_do_nhay(d, x, ks_all, moc, ra, p, luu)
    p(f"\nXong phân tích sau {vn(round(time.time() - t0))} giây.")


# ---------------------------------------------------------------- tự kiểm
def tu_kiem():
    loi = 0

    def kt(ten, dat, ct=""):
        nonlocal loi
        loi += not dat
        print(f"{ten:66s} {'ĐẠT' if dat else 'SAI ' + str(ct)}")

    # ngày chia
    S = ngay_chia([np.nan, 5.0, 0.4, 2.3, -1.0, 2.9, 10.0], [10.0, 4.0, 3.0, 2.8, 5.0, 2.5, 10.0])
    kt("Ngày chia: không dấu hiệu / sau kết thúc / từ đầu / cùng ngày / trước tạo đơn",
       list(S) == [np.inf, np.inf, 0.0, 2.0, 0.0, np.inf, np.inf], S)
    # dạng dài
    x = pd.DataFrame({"id": [0, 1, 2, 3], "T": [10.0, 7.0, 6.0, 4.0], "E": [1, 0, 1, 1],
                      "S_a": [3.0, 0.0, 2.0, np.inf], "S_b": [np.inf, np.inf, 5.0, 4.0], "z": [1.0, 2.0, 3.0, 4.0]})
    dl = dang_dai(x, ["a", "b"])
    g = {i: dl[dl["id"] == i][["start", "stop", "event", "a", "b"]].values.tolist() for i in range(4)}
    kt("Dạng dài: chia tại ngày 3, sự kiện ở đoạn cuối", g[0] == [[0, 3, 0, 0, 0], [3, 10, 1, 1, 0]], g[0])
    kt("Dạng dài: phơi nhiễm từ đầu, bị kiểm duyệt", g[1] == [[0, 7, 0, 1, 0]], g[1])
    kt("Dạng dài: hai luật chia tại ngày 2 và 5", g[2] == [[0, 2, 0, 0, 0], [2, 5, 0, 1, 0], [5, 6, 1, 1, 1]], g[2])
    kt("Dạng dài: dấu hiệu tại T không tạo đoạn", g[3] == [[0, 4, 1, 0, 0]], g[3])
    kt("Dạng dài: giữ biến ban đầu", dl.loc[dl["id"] == 2, "z"].eq(3.0).all())

    # thời điểm dấu hiệu trên vết giả lập
    t0 = pd.Timestamp("2018-03-01 08:00")
    m, ng = pd.Timedelta(minutes=1), MOT_NGAY
    vet = {
        "A": [(PO, 0 * m, "u1"), (IR, ng, "u2"), (kc.CHANGE_PRICE, 3 * ng, "u1"), (kc.CHANGE_PRICE, 5 * ng, "u1"),
              (kc.RPB, 3 * ng + m, "batch_01"), (GR, 4 * ng, "u1"), ("Clear Invoice", 20 * ng, "batch_02")],
        "B": [(PO, 0 * m, "u1"), (kc.CHANGE_PRICE, ng, "u1"), (IR, 2 * ng, "u2"), (GR, 3 * ng, "u3"),
              (CANCEL_IR, 4 * ng, "batch_05"), (IR, 5 * ng, "u2"), (CANCEL_IR, 6 * ng, "u2"), (IR, 6 * ng, "u2"),
              (IR, 7 * ng, "u2"), ("Clear Invoice", 30 * ng, "batch_02")],
        "C": [(PO, 0 * m, "u1"), (GR, 2 * ng, "u4"), (GR, 3 * ng, "u1"), (IR, 4 * ng, "u2"), (kc.RPB, 5 * ng, "u2")],
    }
    ev = pd.DataFrame([{"case": c, "act": a, "ts": t0 + dt, "user": u} for c, ds in vet.items() for a, dt, u in ds])
    ev = sap_xep(ev)
    loai, _, _ = phan_loai_clear(ev, 60)
    kq, _ = kc.cac_luat(ev, set(ev["case"]), loai, ev["ts"])
    tg = thoi_diem_dau_hieu(ev, kq)
    kt("R2a: lần gỡ khóa đầu tiên trước nhận hàng", dict(tg["R2a"]) == {"A": t0 + 3 * ng + m}, dict(tg["R2a"]))
    kt("R3: Change Price đầu tiên sau hóa đơn đầu tiên", dict(tg["R3"]) == {"A": t0 + 3 * ng}, dict(tg["R3"]))
    kt("Làm lại: bỏ hủy do batch, hóa đơn cùng thời điểm hủy không tính", dict(tg["lam_lai"]) == {"B": t0 + 7 * ng},
       dict(tg["lam_lai"]))
    kt("R6 theo thời gian: lần nhận hàng của chính người tạo đơn", dict(tg["R6"]) == {"A": t0 + 4 * ng, "C": t0 + 3 * ng},
       dict(tg["R6"]))
    kt("R6 theo thời gian khớp tập R6 của kiem_chung_cuoi", set(tg["R6"].index) == set(kq["R6"][0]))

    # trạng thái và kiểm duyệt
    moc = pd.Timestamp("2019-01-18 13:34")
    dd = pd.DataFrame({"tt_moi": ["Hoàn tất", "Còn mở", "Bị xóa", "Hoàn tất"],
                       "dau_po": [t0, t0, t0, t0], "cuoi_clear_tt": [t0 + 2.2 * ng, pd.NaT, pd.NaT, t0 - ng]},
                      index=["h", "m", "x", "am"])
    for l in LUAT_TV + ["R6"]:
        dd["s_" + l] = np.nan
    xx, loai_tt = tap_phan_tich(dd, "tt_moi", moc)
    kt("Kiểm duyệt: bị xóa và thời gian âm bị loại, được đếm",
       loai_tt == {"bị xóa": 1, "thời gian ≤ 0 hoặc thiếu": 1} and list(xx.index) == ["h", "m"], loai_tt)
    kt("Kiểm duyệt: hoàn tất T = 3, E = 1; còn mở E = 0 tại thời điểm trích xuất",
       xx.loc["h", "T"] == 3 and xx.loc["h", "E"] == 1 and xx.loc["m", "E"] == 0
       and xx.loc["m", "T"] == np.ceil((moc - t0) / ng), xx[["T", "E"]])

    # mô hình: thiên lệch thời gian bất tử trên dữ liệu mô phỏng (dấu hiệu không ảnh hưởng thật đến thời gian)
    import importlib.util
    if importlib.util.find_spec("lifelines") is None:
        print("Chưa có lifelines: python -m pip install lifelines")
        return loi + 1
    rng = np.random.default_rng(2026)
    n = 6000
    tt = np.ceil(rng.exponential(60, n))
    s = rng.exponential(40, n)
    s[rng.random(n) < 0.7] = np.nan
    cd = rng.random(n) < 0.3
    tt_c = np.where(cd, np.ceil(rng.uniform(1, 200, n)), tt)
    e = ((~cd) | (tt <= tt_c)).astype(int)
    tt = np.minimum(tt, tt_c)
    x = pd.DataFrame({"id": np.arange(n), "T": tt, "E": e, "S_a": ngay_chia(s, tt), "z": rng.normal(size=n)})
    m1, _, _ = fit_ctv(dang_dai(x, ["a"]), ["a", "z"])
    hr_tv = float(m1.summary.loc["a", "exp(coef)"])
    nhom_cuoi = x.assign(a=(~np.isnan(s) & (s < tt)).astype(float))
    m2, _ = fit_cph(nhom_cuoi, ["a", "z"])
    hr_nc = float(m2.summary.loc["a", "exp(coef)"])
    kt(f"Mô phỏng HR thật = 1: biến thay đổi theo thời gian HR = {hr_tv:.2f} (0,85–1,15)", 0.85 < hr_tv < 1.15, hr_tv)
    kt(f"Mô phỏng HR thật = 1: chia theo nhóm cuối cho HR = {hr_nc:.2f} < 0,8 (thiên lệch)", hr_nc < 0.8, hr_nc)
    # HR thật = 0,5 sau dấu hiệu: thời gian còn lại sau dấu hiệu kéo dài gấp đôi
    t_goc = rng.exponential(60, n)
    s2 = rng.exponential(30, n)
    t2 = np.where(s2 < t_goc, s2 + (t_goc - s2) * 2, t_goc)
    x2 = pd.DataFrame({"id": np.arange(n), "T": np.ceil(t2).clip(1), "E": 1, "S_a": ngay_chia(s2, t2), "z": 0.0})
    m3, _, _ = fit_ctv(dang_dai(x2, ["a"]), ["a"])
    hr3 = float(m3.summary.loc["a", "exp(coef)"])
    kt(f"Mô phỏng HR thật = 0,5: ước lượng {hr3:.2f} (0,42–0,58)", 0.42 < hr3 < 0.58, hr3)
    # RMST của phân phối mũ: λ⁻¹(1 − e^{−λh})
    te = pd.Series(rng.exponential(50, 20000))
    r, v = rmst(te, pd.Series(1, index=te.index), 100)
    kt(f"RMST mũ(50) đến 100 ngày = {r:.1f} (lý thuyết {50 * (1 - np.exp(-2)):.1f})", abs(r - 50 * (1 - np.exp(-2))) < 1.0, r)
    from lifelines.utils import restricted_mean_survival_time
    tn = pd.Series(np.ceil(rng.exponential(30, 400)))
    en = pd.Series((rng.random(400) < 0.7).astype(int))
    r1, _ = rmst(tn, en, 60)
    r2 = restricted_mean_survival_time(km(tn, en), t=60)
    kt("RMST khớp lifelines", abs(r1 - r2) < 1e-6, (r1, r2))
    # không kiểm duyệt: phương sai ước lượng ≈ phương sai mẫu của min(T, h) / n
    # (lifelines trả về phương sai của chính min(T, h), không phải của ước lượng, nên không dùng cho KTC)
    _, v1 = rmst(tn, pd.Series(1, index=tn.index), 60)
    v_mau = np.var(np.minimum(tn, 60)) / len(tn)
    kt("Phương sai RMST khi không kiểm duyệt = var(min(T, h)) / n", abs(v1 - v_mau) / v_mau < 0.01, (v1, v_mau))
    print("\nTất cả đều đạt." if loi == 0 else f"\n{loi} kiểm tra SAI.")
    return loi


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(cau_hinh.LOG), help="đường dẫn log; mặc định trong cau_hinh.py")
    ap.add_argument("--cong-ty", default="companyID_0000")
    ap.add_argument("--ra", default="ket_qua_sq3")
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        sys.exit(tu_kiem())
    if not a.log:
        ap.error("cần đường dẫn log hoặc --tu-kiem")
    if tu_kiem():
        sys.exit("Kiểm thử giả lập không đạt, dừng.")
    print()
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)
    out = []

    def p(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        out.append(s)

    def luu():
        (ra / "tom_tat_sq3.txt").write_text("\n".join(out), encoding="utf-8")

    t = time.time()
    raw = sq2.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    p(f"Đọc log xong sau {vn(round(time.time() - t))} giây.")
    phan_tich(ev, pv, moc, ra, p, luu)
    luu()
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_sq3.txt.")


if __name__ == "__main__":
    main()
