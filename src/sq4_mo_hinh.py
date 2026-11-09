"""SQ4, huấn luyện và đánh giá mô hình dự báo sớm (thiết kế Mục 3.7.4, Bảng 3.10, ghi ngày 29/10/2026; ba điểm về cách
báo cáo ghi ngày 05/11/2026 theo góp ý G9). Nhãn, chia tập và tiền tố dùng đúng các hàm của sq4_mo_ta.py, nên số dòng hàng
và số dòng nhãn 1 khớp Bảng 4.24–4.27.

  Nhãn chính: R2a, R3 hoặc R6 trong H = 90 ngày kể từ khi tạo dòng đơn hàng. Mô hình riêng cho R3.
  Chia tập: T = 01/07/2018; huấn luyện tạo trong [01/01/2018; T − H], kiểm tra tạo trong [T; thời điểm trích xuất − H].
  Tiền tố 2, 3, 4, 5 sự kiện, kết thúc trước dấu hiệu đầu tiên của nhãn và trước ngày thứ H.
  Biến đầu vào: số lần mỗi hoạt động trong tiền tố (Clear Invoice tách "thanh toán" và "bù trừ do hủy"); số ngày từ khi tạo
    đến cuối tiền tố; hạng mục chi tiêu; log giá trị lúc tạo dòng đơn hàng và biến giá trị 0; quy mô nhà cung cấp (tập huấn
    luyện); target encoding nhà cung cấp (out-of-fold 5 phần theo đơn mua hàng); thông tin R7 (hóa đơn nhà cung cấp có ngày
    trước hoặc cùng ngày tạo dòng đơn hàng) khi hóa đơn đó nằm trong tiền tố.
  Mô hình: Logistic Regression và XGBoost, class weight; siêu tham số chọn bằng kiểm định chéo 5 phần theo đơn mua hàng
    trong tập huấn luyện (PR-AUC); seed cố định; tập kiểm tra chỉ dùng một lần.
  Mốc so sánh: (1) chỉ thuộc tính biết lúc tạo dòng đơn hàng (XGBoost, cùng cách chọn siêu tham số); (2) tỷ lệ sai lệch
    lịch sử của nhà cung cấp tính trên tập huấn luyện (chính là giá trị target encoding).
  Đánh giá: PR-AUC (chính) kèm mốc ngẫu nhiên = tỷ lệ nhãn 1; AUC-ROC; Precision@5%, @10% và lift; KTC bootstrap 95% và
    so sánh bootstrap ghép cặp 1.000 lần. Theo G9: PR-AUC theo tháng tạo của tập kiểm tra kèm tỷ số PR-AUC / tỷ lệ nhãn 1;
    không báo Brier hay đường hiệu chỉnh, không dùng ngưỡng 0,5 (xác suất chỉ để xếp hạng); ngưỡng 100 dòng nhãn 1 trong tập
    kiểm tra áp cho cả từng luật — dưới ngưỡng thì đánh dấu (*): chỉ báo cáo, không kết luận.
  SHAP: giá trị SHAP của XGBoost (TreeSHAP, tính sẵn trong xgboost), trung bình |SHAP| theo biến; hình beeswarm nếu có
    thư viện shap.
  Độ nhạy (Bảng 3.10): (1) loại vendorID_0741 và vendorID_0572; (2) nhãn không có R6; (3) loại dòng có Cancel Invoice Receipt
    hoặc Cancel Goods Receipt; (4) H = 120; (5) bỏ Vendor creates invoice khỏi tiền tố, thông tin R7 chỉ dùng từ khi tiền tố
    có Record Invoice Receipt. Nhãn độ nhạy: nhãn chính + R5 v2.2; nhãn chính + R1 (thời điểm = lần Clear "thanh toán" đầu
    tiên bị luật đánh dấu).

Cách chạy (cùng thư mục với sq4_mo_ta.py; cần pandas, numpy, scikit-learn, xgboost; shap tùy chọn):
    pip install scikit-learn xgboost shap
    python src/sq4_mo_hinh.py --tu-kiem
    python src/sq4_mo_hinh.py --chi-chinh   (8 cấu hình chính)
    python src/sq4_mo_hinh.py               (thêm 28 cấu hình độ nhạy)
Kết quả ghi vào ket_qua_sq4_mo_hinh/; gửi lại tom_tat_sq4_mo_hinh.txt.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import g4_kiem_tra as g4
import kiem_chung_cuoi as kc
import sq2_tuan_thu as sq2
import sq3_thoi_gian as sq3
import sq4_mo_ta as mt
from r5_v2_phan_loai_clear import (CANCEL_GR, CANCEL_IR, CLEAR, GR, IR, PO, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear,
                                   sap_xep)
import cau_hinh

SEED = 2026
SO_PHAN = 5
SO_BOOT = 1000
SO_LUONG = -1  # số luồng CPU cho XGBoost. CẢNH BÁO: đổi số luồng làm XGBoost cho kết quả hơi khác (thứ tự cộng số thực), có thể đổi bộ tham số được chọn; muốn lặp lại đúng kết quả cũ thì giữ mặc định
BOOT_THEO_DON = True  # G10: bootstrap rút lại theo đơn mua hàng; --bootstrap-dong để lặp lại cách của lần chạy đầu
CAC_K = mt.CAC_K
HL, KT = mt.HL, mt.KT
NGUONG = mt.NGUONG
VENDOR_INV = mt.VENDOR_INV
SO_HANG_MUC = 8
LAM_TRON_TE = 20  # target encoding: làm trơn về tỷ lệ chung với trọng số 20 dòng hàng
TOP = [0.05, 0.10]
NHAN = dict(mt.NHAN, chinh_R5=["R2a", "R3", "R6", "R5"], chinh_R1=["R2a", "R3", "R6", "R1"])
TEN_NHAN = dict(mt.TEN_NHAN, chinh_R5="Nhãn chính + R5 v2.2", chinh_R1="Nhãn chính + R1")
LUAT_DH = ["R2a", "R3", "R6", "R5", "R1"]

LUOI_LR = [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]]
LUOI_XGB = [dict(max_depth=d, n_estimators=n, min_child_weight=w, learning_rate=0.05)
            for d, n, w in itertools.product([3, 5], [300, 600], [1, 5])]

vn, pct = sq3.vn, sq3.pct


def so4(x):
    return "–" if x is None or not np.isfinite(x) else vn(float(x), ".4f")


# ---------------------------------------------------------------- cấu hình
def cac_cau_hinh(chi_chinh):
    """(mã, nhãn, H, tập dữ liệu, bỏ Vendor creates invoice khỏi tiền tố). Phân tích chính trước."""
    ch = [("chinh", "chinh", 90, "toàn bộ", False), ("R3", "R3", 90, "toàn bộ", False)]
    dn = [("dn1_loai_hai_ncc", "chinh", 90, "loại hai nhà cung cấp R6", False),
          ("dn2_khong_R6", "khong_R6", 90, "toàn bộ", False),
          ("dn3_loai_huy", "chinh", 90, "loại dòng có hủy", False),
          ("dn4_H120", "chinh", 120, "toàn bộ", False),
          ("dn5_bo_vendor_inv", "chinh", 90, "toàn bộ", True),
          ("nhan_cong_R5", "chinh_R5", 90, "toàn bộ", False),
          ("nhan_cong_R1", "chinh_R1", 90, "toàn bộ", False)]
    return ch if chi_chinh else ch + dn


# ---------------------------------------------------------------- bảng dòng hàng
def bo_sung_dong_hang(d, ev, pv, ts_dp, loai):
    """Thêm vào bảng dòng hàng của sq4_mo_ta: đơn mua hàng, hạng mục, giá trị lúc tạo, ngày hóa đơn nhà cung cấp và ngày tạo
    (giờ địa phương, theo ngày, như R7b), thời điểm dấu hiệu R1 và R5 v2.2 (cho nhãn độ nhạy)."""
    idx = d.index
    d["don"] = pd.Series(idx.astype(str), index=idx).str.rsplit("_", n=1).str[0]
    dau = ev.drop_duplicates("case").set_index("case").reindex(idx)
    d["hang_muc_goc"] = dau["spend_area"].astype(str) if "spend_area" in dau else "?"
    # giá trị tại Create Purchase Order Item đầu tiên; không dùng giá trị của sự kiện sau (biết sau thời điểm tạo)
    if "networth" in ev:
        po = ev[(ev["act"] == PO) & ev["networth"].notna()].groupby("case")["networth"].first().reindex(idx)
    else:
        po = pd.Series(np.nan, index=idx)
    d["gia_tri_0"] = (~(po > 0)).astype(float)
    d["log_gia_tri"] = np.log(po.where(po > 0)).fillna(0.0)
    vi = ts_dp[ev["act"] == VENDOR_INV].groupby(ev.loc[ev["act"] == VENDOR_INV, "case"]).min().dt.normalize()
    po_d = ts_dp[ev["act"] == PO].groupby(ev.loc[ev["act"] == PO, "case"]).min().dt.normalize()
    d["ngay_vi"], d["ngay_po"] = vi.reindex(idx), po_d.reindex(idx)
    # R1: lần Clear "thanh toán" đầu tiên mà lúc đó chưa có nhận hàng còn hiệu lực (cùng logic r5.r1_theo_clear)
    e = ev[ev["act"].isin([GR, CANCEL_GR, CLEAR])].copy()
    e["cum"] = e["act"].map({GR: 1, CANCEL_GR: -1, CLEAR: 0}).groupby(e["case"]).cumsum()
    tt = set(loai.index[loai == "thanh_toan"])
    m = (e["act"] == CLEAR) & (e["cum"] <= 0) & e.index.isin(tt)
    d["sig_R1"] = e.loc[m].groupby("case")["ts"].min().reindex(idx)
    # R5 v2.2: lần vi phạm đầu tiên (cùng logic g4.r5_v22)
    bd = g4.dem_r5(ev, loai)
    vp = bd[bd["vuot_tg"] & ~bd["ir_sau"]]
    d["sig_R5"] = vp.groupby("case")["ts"].min().reindex(idx)
    return d


def ev_tien_to(ev, loai, bo_vendor_inv):
    """Nhật ký dùng để cắt tiền tố: tên hoạt động có tách Clear "thanh toán" / "bù trừ do hủy"; tùy chọn bỏ Vendor creates
    invoice."""
    e = ev[["case", "act", "ts"]].copy()
    la_clear = e["act"] == CLEAR
    e.loc[la_clear, "act"] = np.where(loai.reindex(e.index[la_clear]).eq("thanh_toan"), "Clear Invoice (thanh toán)",
                                      "Clear Invoice (bù trừ do hủy)")
    if bo_vendor_inv:
        e = e[e["act"] != VENDOR_INV]
    return e.reset_index(drop=True)


def dem_tien_to(e, k, hoat_dong):
    """Số lần mỗi hoạt động trong k sự kiện đầu của mỗi dòng hàng, và tiền tố có Record Invoice Receipt hay chưa."""
    vt = e.groupby("case").cumcount()
    s = e[vt < k]
    x = pd.crosstab(s["case"], s["act"]).reindex(columns=hoat_dong, fill_value=0)
    x.columns = ["n_" + c for c in x.columns]
    return x


# ---------------------------------------------------------------- biến đầu vào
def gan_phan(d_hl, y_hl, rng_seed=SEED):
    """Chia dòng hàng huấn luyện thành 5 phần, mọi dòng cùng đơn mua hàng nằm trong cùng phần, cân bằng nhãn 1."""
    from sklearn.model_selection import StratifiedGroupKFold
    phan = pd.Series(-1, index=d_hl.index)
    n_phan = min(SO_PHAN, int(y_hl.sum()), d_hl["don"].nunique())
    if n_phan < 2:
        return phan.replace(-1, 0)
    cv = StratifiedGroupKFold(n_splits=n_phan, shuffle=True, random_state=rng_seed)
    for i, (_, va) in enumerate(cv.split(np.zeros(len(d_hl)), y_hl.values, d_hl["don"].values)):
        phan.iloc[va] = i
    return phan


def target_encoding(vendor_hl, y_hl, phan, vendor_kt):
    """Out-of-fold cho dòng huấn luyện; toàn bộ tập huấn luyện cho dòng kiểm tra. Làm trơn về tỷ lệ chung."""
    def bang(v, y):
        chung = y.mean()  # tỷ lệ chung cũng tính ngoài phần, để giá trị out-of-fold không chứa nhãn của chính phần đó
        g = pd.DataFrame({"v": v.values, "y": y.values}).groupby("v")["y"].agg(["sum", "count"])
        return (g["sum"] + LAM_TRON_TE * chung) / (g["count"] + LAM_TRON_TE), chung

    te_hl = pd.Series(y_hl.mean(), index=vendor_hl.index)
    for f in sorted(phan.unique()):
        trong = phan == f
        if (~trong).sum() == 0:
            continue
        b, chung = bang(vendor_hl[~trong], y_hl[~trong])
        te_hl[trong] = vendor_hl[trong].map(b).fillna(chung).values
    b, chung = bang(vendor_hl, y_hl)
    te_kt = vendor_kt.map(b).fillna(chung)
    return te_hl, te_kt


def bien_tinh(d, hl, kt, te_hl, te_kt):
    """Biến biết lúc tạo dòng đơn hàng: hạng mục (8 hạng mục lớn nhất của tập huấn luyện), giá trị, quy mô nhà cung cấp
    (số dòng hàng trong tập huấn luyện), target encoding nhà cung cấp."""
    top = d.loc[hl, "hang_muc_goc"].value_counts().index[:SO_HANG_MUC]
    quy_mo = d.loc[hl, "vendor"].value_counts()
    x = pd.DataFrame(index=d.index[(hl | kt).values])
    hm = d.loc[x.index, "hang_muc_goc"]
    for i, h in enumerate(top):
        x[f"hm_{i}"] = (hm == h).astype(float)
    x["log_gia_tri"] = d.loc[x.index, "log_gia_tri"]
    x["gia_tri_0"] = d.loc[x.index, "gia_tri_0"]
    x["log_quy_mo_ncc"] = np.log1p(d.loc[x.index, "vendor"].map(quy_mo).fillna(0).astype(float))
    x["te_ncc"] = pd.concat([te_hl, te_kt]).reindex(x.index)
    return x, {f"hm_{i}": h for i, h in enumerate(top)}


def bien_tien_to(d, e, tk_k, k, hoat_dong, chi_khi_co_ir):
    """Biến của tiền tố k: số lần mỗi hoạt động, số ngày từ khi tạo đến cuối tiền tố, thông tin R7 khi hóa đơn nhà cung cấp
    nằm trong tiền tố (độ nhạy (5): khi tiền tố đã có Record Invoice Receipt)."""
    x = dem_tien_to(e, k, hoat_dong).reindex(d.index, fill_value=0)
    x["ngay_den_cuoi_tien_to"] = (tk_k - d["tao"]) / sq3.MOT_NGAY
    co = x["n_" + IR] > 0 if chi_khi_co_ir else x.get("n_" + VENDOR_INV, pd.Series(0, index=d.index)) > 0
    x["r7_vi_truoc_po"] = (co & (d["ngay_vi"] < d["ngay_po"])).astype(float)
    x["r7_vi_cung_ngay_po"] = (co & (d["ngay_vi"] == d["ngay_po"])).astype(float)
    return x


# ---------------------------------------------------------------- chỉ số
def chuan_bi_ap(score, y):
    o = np.argsort(-score, kind="mergesort")
    s = score[o]
    cuoi = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # vị trí cuối của mỗi nhóm điểm bằng nhau
    return o, y[o].astype(float), cuoi


def ap_trong_so(cb, w=None):
    """Average precision (như sklearn.average_precision_score), có trọng số để tính bootstrap nhanh."""
    o, yy, cuoi = cb
    ww = np.ones_like(yy) if w is None else w[o].astype(float)
    tp = np.cumsum(ww * yy)[cuoi]
    fp = np.cumsum(ww * (1 - yy))[cuoi]
    if tp[-1] <= 0:
        return np.nan
    tong = tp + fp
    prec = np.divide(tp, tong, out=np.zeros_like(tp), where=tong > 0)  # nhóm có trọng số 0: không đóng góp
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def p_top(score, y, frac):
    n = max(1, int(np.ceil(frac * len(y))))
    o = np.argsort(-score, kind="mergesort")[:n]
    return float(y[o].mean())


def chi_so(score, y):
    from sklearn.metrics import roc_auc_score
    ty_le = float(y.mean())
    r = {"n": len(y), "n1": int(y.sum()), "ty_le_nhan_1": ty_le, "pr_auc": ap_trong_so(chuan_bi_ap(score, y)),
         "auc_roc": roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan}
    for f in TOP:
        pk = p_top(score, y, f)
        r[f"p_at_{int(f * 100)}"] = pk
        r[f"lift_{int(f * 100)}"] = pk / ty_le if ty_le > 0 else np.nan
    return r


def bootstrap(diem, y, so_boot=SO_BOOT, seed=SEED, nhom=None):
    """KTC 95% của PR-AUC từng mô hình và của chênh lệch ghép cặp (cùng mẫu bootstrap cho mọi mô hình).
    nhom: mã đơn mua hàng của từng dòng; khi có, mỗi lần bootstrap rút lại các đơn (có hoàn lại) và lấy mọi dòng của đơn
    được rút (G10, ghi ngày 06/11/2026). nhom = None: rút lại theo dòng hàng như lần chạy đầu ngày 05/11/2026."""
    rng = np.random.default_rng(seed)
    cb = {m: chuan_bi_ap(s, y) for m, s in diem.items()}
    n = len(y)
    if nhom is not None:
        ma_nhom, chi_so_nhom = np.unique(np.asarray(nhom), return_inverse=True)
        g = len(ma_nhom)
    kq = {m: np.empty(so_boot) for m in diem}
    for b in range(so_boot):
        if nhom is None:
            w = np.bincount(rng.integers(0, n, n), minlength=n)
        else:
            w = np.bincount(rng.integers(0, g, g), minlength=g)[chi_so_nhom]
        for m in diem:
            kq[m][b] = ap_trong_so(cb[m], w)
    return kq


def tom_tat_boot(kq, cap, diem_quan_sat=None):
    """KTC percentile; chênh lệch báo giá trị quan sát trên tập kiểm tra (nếu có), bootstrap chỉ cho KTC và p."""
    ci = {m: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for m, v in kq.items()}
    ss = []
    for a, b in cap:
        dlt = kq[a] - kq[b]
        dlt = dlt[np.isfinite(dlt)]
        p = min(1.0, 2 * min((dlt <= 0).mean(), (dlt >= 0).mean())) if len(dlt) else np.nan
        qs = diem_quan_sat[a] - diem_quan_sat[b] if diem_quan_sat else (float(np.mean(dlt)) if len(dlt) else np.nan)
        ss.append({"mo_hinh": a, "so_voi": b, "chenh_pr_auc": qs,
                   "ktc_duoi": np.percentile(dlt, 2.5) if len(dlt) else np.nan,
                   "ktc_tren": np.percentile(dlt, 97.5) if len(dlt) else np.nan, "p_hai_phia": p})
    return ci, ss


# ---------------------------------------------------------------- mô hình
def tao_lr(tham_so):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=tham_so["C"], class_weight="balanced", max_iter=5000,
                                                              random_state=SEED))


def tao_xgb(tham_so, ty_le_am_duong):
    from xgboost import XGBClassifier
    return XGBClassifier(**tham_so, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
                         scale_pos_weight=ty_le_am_duong, random_state=SEED, n_jobs=SO_LUONG, verbosity=0)


def chon_tham_so(tao, luoi, X, y, phan):
    """Kiểm định chéo theo phần đã gán (theo đơn mua hàng); chọn bộ tham số có PR-AUC trung bình cao nhất."""
    tot, diem_tot, bang = None, -np.inf, []
    for ts in luoi:
        ds = []
        for f in sorted(phan.unique()):
            va = (phan == f).values
            if y[va].sum() == 0 or y[~va].sum() == 0:
                continue
            m = tao(ts)
            m.fit(X[~va], y[~va])
            ds.append(ap_trong_so(chuan_bi_ap(m.predict_proba(X[va])[:, 1], y[va])))
        tb = float(np.nanmean(ds)) if ds else np.nan
        bang.append({**ts, "pr_auc_cv": tb})
        if np.isfinite(tb) and tb > diem_tot:
            tot, diem_tot = ts, tb
    return (tot or luoi[0]), bang


def shap_xgb(m, X, ten_bien, toi_da=20000, seed=SEED):
    """TreeSHAP tính sẵn trong xgboost (pred_contribs); bỏ cột hằng số cuối."""
    import xgboost
    if len(X) > toi_da:
        X = X[np.random.default_rng(seed).choice(len(X), toi_da, replace=False)]
    # không truyền tên biến: mô hình được huấn luyện trên mảng numpy, và tên có dấu ngoặc có thể bị xgboost từ chối
    c = m.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)[:, :-1]
    return c, X


# ---------------------------------------------------------------- một cấu hình
def chay_cau_hinh(ma, ten_nhan, h, ten_tap_dl, bo_vi, k, d, chon, ev_tt, tk, hoat_dong, moc, ra, p, so_boot, luoi_lr,
                  luoi_xgb, ghi_shap):
    luat = NHAN[ten_nhan]
    sig, y_all = mt.nhan(d, luat, h)
    tp = mt.tap(d["tao"], moc, h)
    hl0, kt0 = (tp == HL) & chon, (tp == KT) & chon
    hop = mt.hop_le(tk[k], sig, d["tao"], h)
    hl, kt = hl0 & hop, kt0 & hop
    n1_kt = int((kt & y_all).sum())
    if (hl & y_all).sum() < 2 or n1_kt < 1:
        p(f"  {ma}, tiền tố {k}: quá ít dòng nhãn 1 (huấn luyện {int((hl & y_all).sum())}, kiểm tra {n1_kt}), bỏ qua.")
        return None
    # phần kiểm định chéo và target encoding gán ở cấp dòng hàng huấn luyện (trước khi cắt tiền tố)
    phan_dh = gan_phan(d[hl0], y_all[hl0])
    te_hl, te_kt = target_encoding(d.loc[hl0, "vendor"], y_all[hl0].astype(float), phan_dh, d.loc[kt0, "vendor"])
    xs, ten_hm = bien_tinh(d, hl0, kt0, te_hl, te_kt)
    xt = bien_tien_to(d, ev_tt, tk[k], k, hoat_dong, chi_khi_co_ir=bo_vi)
    X_df = xs.join(xt, how="left")
    cot_tinh = list(xs.columns)
    cot_all = list(X_df.columns)
    Xh, Xk = X_df.loc[hl[hl].index], X_df.loc[kt[kt].index]
    yh, yk = y_all[hl].values.astype(int), y_all[kt].values.astype(int)
    ph = phan_dh.reindex(Xh.index)
    am_duong = (len(yh) - yh.sum()) / max(yh.sum(), 1)

    t0 = time.time()
    ts_lr, cv_lr = chon_tham_so(tao_lr, luoi_lr, Xh[cot_all].values, yh, ph)
    ts_x, cv_x = chon_tham_so(lambda t: tao_xgb(t, am_duong), luoi_xgb, Xh[cot_all].values, yh, ph)
    ts_m1, cv_m1 = chon_tham_so(lambda t: tao_xgb(t, am_duong), luoi_xgb, Xh[cot_tinh].values, yh, ph)
    lr = tao_lr(ts_lr).fit(Xh[cot_all].values, yh)
    xg = tao_xgb(ts_x, am_duong).fit(Xh[cot_all].values, yh)
    m1 = tao_xgb(ts_m1, am_duong).fit(Xh[cot_tinh].values, yh)
    diem = {"XGBoost": xg.predict_proba(Xk[cot_all].values)[:, 1],
            "Logistic Regression": lr.predict_proba(Xk[cot_all].values)[:, 1],
            "Mốc 1: biến lúc tạo": m1.predict_proba(Xk[cot_tinh].values)[:, 1],
            "Mốc 2: tỷ lệ của nhà cung cấp": Xk["te_ncc"].values}

    nen = {"ma": ma, "nhan": ten_nhan, "H": h, "tap_du_lieu": ten_tap_dl, "bo_vendor_inv": bo_vi, "tien_to": k,
           "n_huan_luyen": len(yh), "n1_huan_luyen": int(yh.sum()), "chi_bao_cao": n1_kt < NGUONG}
    kq_boot = bootstrap(diem, yk, so_boot, nhom=d.loc[Xk.index, "don"].values if BOOT_THEO_DON else None)
    co_tt = (Xk["n_Clear Invoice (thanh toán)"].values > 0) if "n_Clear Invoice (thanh toán)" in Xk else np.zeros(len(yk), bool)
    pd.DataFrame({"case": Xk.index, "don": d.loc[Xk.index, "don"].values, "y": yk, "tien_to_co_clear_tt": co_tt,
                  **diem}).to_csv(
        ra / f"diem_kiem_tra_{ma}_k{k}.csv", index=False, encoding="utf-8-sig")
    ci, ss = tom_tat_boot(kq_boot, [("XGBoost", "Logistic Regression"), ("XGBoost", "Mốc 1: biến lúc tạo"),
                                    ("XGBoost", "Mốc 2: tỷ lệ của nhà cung cấp"),
                                    ("Logistic Regression", "Mốc 1: biến lúc tạo"),
                                    ("Logistic Regression", "Mốc 2: tỷ lệ của nhà cung cấp")],
                          {m: ap_trong_so(chuan_bi_ap(s, yk)) for m, s in diem.items()})
    dong = []
    for m, s in diem.items():
        r = {**nen, "mo_hinh": m, **chi_so(s, yk), "pr_auc_ktc_duoi": ci[m][0], "pr_auc_ktc_tren": ci[m][1]}
        dong.append(r)
    ss = [{**nen, **x} for x in ss]

    # theo tháng tạo của tập kiểm tra (G9, điểm a)
    thang = d.loc[Xk.index, "tao"].dt.to_period("M").astype(str).values
    tt_rows = []
    for th in sorted(set(thang)):
        mm = thang == th
        for m in ["XGBoost", "Logistic Regression", "Mốc 1: biến lúc tạo", "Mốc 2: tỷ lệ của nhà cung cấp"]:
            yy = yk[mm]
            ap = ap_trong_so(chuan_bi_ap(diem[m][mm], yy)) if yy.sum() > 0 else np.nan
            tt_rows.append({**nen, "thang": th, "mo_hinh": m, "n": int(mm.sum()), "n1": int(yy.sum()),
                            "ty_le_nhan_1": yy.mean(), "pr_auc": ap, "pr_auc_chia_ty_le": ap / yy.mean() if yy.sum() else np.nan,
                            "chi_bao_cao_thang": int(yy.sum()) < NGUONG})

    # theo luật: dòng nhãn 1 có dấu hiệu của luật đó trong H ngày, so với mọi dòng nhãn 0 (G9, điểm c)
    luat_rows = []
    for l in luat:
        co_l = ((d.loc[Xk.index, "sig_" + l] - d.loc[Xk.index, "tao"]) <= pd.Timedelta(days=h)).values & (yk == 1)
        mm = co_l | (yk == 0)
        for m in ["XGBoost", "Logistic Regression", "Mốc 1: biến lúc tạo", "Mốc 2: tỷ lệ của nhà cung cấp"]:
            ap = ap_trong_so(chuan_bi_ap(diem[m][mm], co_l[mm].astype(int))) if co_l.sum() else np.nan
            luat_rows.append({**nen, "luat": l, "mo_hinh": m, "n1_luat": int(co_l.sum()),
                              "ty_le_nhan_1": co_l.sum() / mm.sum(), "pr_auc": ap,
                              "chi_bao_cao_luat": int(co_l.sum()) < NGUONG})

    # SHAP
    shap_rows = []
    if ghi_shap:
        try:
            c, Xs = shap_xgb(xg, Xk[cot_all].values, cot_all)
        except Exception as ex:  # noqa: BLE001  SHAP lỗi thì vẫn giữ các kết quả khác
            p(f"    !!! Không tính được SHAP: {ex}")
            ghi_shap = False
    if ghi_shap:
        tb = np.abs(c).mean(axis=0)
        for b, v in zip(cot_all, tb):
            shap_rows.append({**nen, "bien": ten_hm.get(b, b), "ma_bien": b, "tb_abs_shap": float(v)})
        np.save(ra / f"shap_{ma}_k{k}.npy", c)
        ve_shap(c, Xs, [ten_hm.get(b, b) for b in cot_all], ra / f"shap_{ma}_k{k}.png", f"{TEN_NHAN[ten_nhan]}, tiền tố {k}")

    tham = {"ma": ma, "tien_to": k, "lr": ts_lr, "xgb": ts_x, "moc1": ts_m1,
            "cv_lr": cv_lr, "cv_xgb": cv_x, "cv_moc1": cv_m1, "giay": round(time.time() - t0, 1)}
    return dong, ss, tt_rows, luat_rows, shap_rows, tham


def ve_shap(c, X, ten, duong_dan, tieu_de):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        import shap
        plt.figure()
        shap.summary_plot(c, X, feature_names=ten, show=False, max_display=15)
        plt.title(tieu_de, fontsize=9)
    except Exception:  # không có thư viện shap: vẽ cột trung bình |SHAP|
        tb = np.abs(c).mean(axis=0)
        o = np.argsort(tb)[-15:]
        plt.figure(figsize=(6, 4.5))
        plt.barh(np.array(ten)[o], tb[o], color="0.3")
        plt.xlabel("Trung bình |SHAP| (log-odds)")
        plt.title(tieu_de, fontsize=9)
    plt.tight_layout()
    plt.savefig(duong_dan, dpi=200)
    plt.close("all")


# ---------------------------------------------------------------- toàn bộ
def phan_tich(ev, pv, moc, ra, p, luu, chi_chinh=False, so_boot=SO_BOOT, luoi_lr=LUOI_LR, luoi_xgb=LUOI_XGB,
              ky_vong=mt.KY_VONG, chi_ma=None, chi_k=None):
    t_bat_dau = time.time()
    p(f"Phạm vi {vn(len(pv))} dòng hàng, {vn(len(ev))} sự kiện. Thời điểm trích xuất: {moc}")
    d = mt.bang_dong_hang(ev, pv, p, ky_vong)
    ts_dp = g4.gio_dia_phuong(ev, lambda *a: None)
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    d = bo_sung_dong_hang(d, ev, pv, ts_dp, loai)
    p(f"Dòng hàng thiếu giá trị tại Create Purchase Order Item: {vn(int((d['gia_tri_0'] == 1).sum()))}; "
      f"R1 có thời điểm: {vn(int(d['sig_R1'].notna().sum()))} (kỳ vọng 654); R5 v2.2 có thời điểm: "
      f"{vn(int(d['sig_R5'].notna().sum()))} (kỳ vọng 681).")
    ev_tt = {False: ev_tien_to(ev, loai, False), True: ev_tien_to(ev, loai, True)}
    tk = {bv: mt.cuoi_tien_to(ev_tt[bv])[0].reindex(d.index) for bv in ev_tt}
    hoat_dong = sorted(ev_tt[False]["act"].unique())
    tat_ca = pd.Series(True, index=d.index)
    cac_tap_dl = {"toàn bộ": tat_ca, "loại hai nhà cung cấp R6": ~d["vendor"].isin(mt.HAI_NCC_R6),
                  "loại dòng có hủy": ~d["co_huy"]}
    p(f"{len(hoat_dong)} hoạt động làm biến đếm (Clear Invoice tách hai loại). Seed {SEED}; {SO_PHAN} phần theo đơn mua "
      f"hàng; lưới: LR {len(luoi_lr)} bộ, XGBoost {len(luoi_xgb)} bộ; bootstrap {vn(so_boot)} lần.")
    p("Mỗi độ dài tiền tố một mô hình: huấn luyện chỉ trên tiền tố cùng độ dài, siêu tham số chọn riêng (G10, ghi 06/11/2026). "
      + ("Bootstrap rút lại theo đơn mua hàng (G10)." if BOOT_THEO_DON else "Bootstrap rút lại theo dòng hàng (như lần chạy đầu 05/11/2026)."))
    p("Xác suất dự báo chỉ dùng để xếp hạng (class weight; hai tập có tỷ lệ nhãn 1 khác nhau): không báo Brier, đường hiệu "
      "chỉnh hay ngưỡng 0,5; Precision@k và lift theo top k%.")
    p(f"(*) tập kiểm tra có dưới {NGUONG} dòng nhãn 1 (toàn nhãn, hoặc của riêng luật / tháng): chỉ báo cáo, không kết luận.")

    out = {"chi_so": [], "so_sanh": [], "theo_thang": [], "theo_luat": [], "shap": [], "tham_so": []}
    cau_hinh = cac_cau_hinh(chi_chinh and not chi_ma)
    if chi_ma:  # chạy lại một số cấu hình (ví dụ khi lần chạy trước bị ngắt)
        cau_hinh = [c for c in cau_hinh if c[0] in chi_ma]
        p(f"Chỉ chạy cấu hình: {', '.join(c[0] for c in cau_hinh)}" + (f"; tiền tố {chi_k}" if chi_k else ""))
    for ma, ten_nhan, h, ten_tap_dl, bo_vi in cau_hinh:
        p(f"\n=== {ma}: {TEN_NHAN[ten_nhan]}, H = {h}, {ten_tap_dl}"
          + (", bỏ Vendor creates invoice khỏi tiền tố" if bo_vi else "") + " ===")
        for k in (chi_k or CAC_K):
            r = chay_cau_hinh(ma, ten_nhan, h, ten_tap_dl, bo_vi, k, d, cac_tap_dl[ten_tap_dl], ev_tt[bo_vi], tk[bo_vi],
                              hoat_dong, moc, ra, p, so_boot, luoi_lr, luoi_xgb, ghi_shap=ma in ("chinh", "R3"))
            if r is None:
                continue
            dong, ss, tt_rows, luat_rows, shap_rows, tham = r
            for key, v in zip(["chi_so", "so_sanh", "theo_thang", "theo_luat", "shap"], [dong, ss, tt_rows, luat_rows, shap_rows]):
                out[key] += v
            out["tham_so"].append(tham)
            in_ket_qua(dong, ss, luat_rows, tham, p)
            ghi(out, ra)
            luu()
    in_holm(out, ra, p)
    in_theo_thang(pd.DataFrame(out["theo_thang"]), p)
    in_shap(pd.DataFrame(out["shap"]), p)
    ghi(out, ra)
    p(f"\nXong sau {vn(round((time.time() - t_bat_dau) / 60, 1), '.1f')} phút.")


def ghi(out, ra):
    for key in ["chi_so", "so_sanh", "theo_thang", "theo_luat", "shap"]:
        if out[key]:
            pd.DataFrame(out[key]).to_csv(ra / f"{key}.csv", index=False, encoding="utf-8-sig")
    (ra / "tham_so.json").write_text(json.dumps(out["tham_so"], ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def in_ket_qua(dong, ss, luat_rows, tham, p):
    r0 = dong[0]
    sao = " (*)" if r0["chi_bao_cao"] else ""
    p(f"  Tiền tố {r0['tien_to']}: huấn luyện {vn(r0['n_huan_luyen'])} dòng ({vn(r0['n1_huan_luyen'])} nhãn 1); kiểm tra "
      f"{pct(r0['n1'], r0['n'])} nhãn 1{sao}; mốc ngẫu nhiên PR-AUC = {so4(r0['ty_le_nhan_1'])}; chạy {tham['giay']} giây")
    p(f"    {'Mô hình':32s} {'PR-AUC (KTC 95%)':>26s} {'AUC-ROC':>8s} {'P@5%':>7s} {'Lift@5%':>8s} {'P@10%':>7s} {'Lift@10%':>9s}")
    for r in dong:
        p(f"    {r['mo_hinh']:32s} {so4(r['pr_auc']):>8s} ({so4(r['pr_auc_ktc_duoi'])}–{so4(r['pr_auc_ktc_tren'])}) "
          f"{so4(r['auc_roc']):>8s} {so4(r['p_at_5']):>7s} {vn(r['lift_5'], '.2f'):>8s} {so4(r['p_at_10']):>7s} "
          f"{vn(r['lift_10'], '.2f'):>9s}")
    for s in ss:
        p(f"    Chênh PR-AUC {s['mo_hinh']} − {s['so_voi']}: {so4(s['chenh_pr_auc'])} ({so4(s['ktc_duoi'])}–"
          f"{so4(s['ktc_tren'])}), p bootstrap = {sq3.so_p(s['p_hai_phia'])}")
    for r in luat_rows:
        if r["mo_hinh"] == "XGBoost":
            p(f"    Theo luật {r['luat']}: {vn(r['n1_luat'])} dòng nhãn 1 trong tập kiểm tra; PR-AUC XGBoost "
              f"{so4(r['pr_auc'])} (mốc ngẫu nhiên {so4(r['ty_le_nhan_1'])})" + (" (*)" if r["chi_bao_cao_luat"] else ""))


def holm(pv):
    """p hiệu chỉnh Holm (Holm, 1979) cho một họ kiểm định."""
    pv = np.asarray(pv, float)
    o = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * pv[j]))
        adj[j] = run
    return adj


def in_holm(out, ra, p):
    """G10: hiệu chỉnh Holm trong từng nhãn (5 cặp so sánh × 4 độ dài tiền tố = 20 kiểm định)."""
    if not out["so_sanh"]:
        return
    s = pd.DataFrame(out["so_sanh"])
    s["p_holm"] = np.nan
    for ma, g in s.groupby("ma"):
        s.loc[g.index, "p_holm"] = holm(g["p_hai_phia"].fillna(1.0).values)
    out["so_sanh"] = s.to_dict("records")
    p("\nSo sánh có ý nghĩa sau hiệu chỉnh Holm trong từng nhãn (p Holm < 0,05; 20 kiểm định mỗi nhãn):")
    for ma, g in s.groupby("ma", sort=False):
        co = g[g["p_holm"] < 0.05]
        p(f"  {ma}: " + ("; ".join(f"tiền tố {int(r['tien_to'])}: {r['mo_hinh']} − {r['so_voi']} {so4(r['chenh_pr_auc'])} "
                                  f"(p Holm {sq3.so_p(r['p_holm'])})" for _, r in co.iterrows()) or "không có"))


def in_theo_thang(t, p):
    if t.empty:
        return
    p("\nPR-AUC theo tháng tạo của tập kiểm tra, XGBoost (tỷ số = PR-AUC / tỷ lệ nhãn 1 của tháng):")
    x = t[(t["mo_hinh"] == "XGBoost") & t["ma"].isin(["chinh", "R3"])]
    for _, r in x.iterrows():
        p(f"  {r['ma']:6s} tiền tố {r['tien_to']} {r['thang']}: {vn(r['n']):>6s} dòng, {vn(r['n1']):>4s} nhãn 1 "
          f"({vn(r['ty_le_nhan_1'], '.2%')}); PR-AUC {so4(r['pr_auc'])}; tỷ số {vn(r['pr_auc_chia_ty_le'], '.2f')}"
          + (" (*)" if r["chi_bao_cao_thang"] else ""))


def in_shap(s, p):
    if s.empty:
        return
    p("\nSHAP (XGBoost, tập kiểm tra): 8 biến có trung bình |SHAP| lớn nhất")
    for (ma, k), g in s.groupby(["ma", "tien_to"]):
        g = g.sort_values("tb_abs_shap", ascending=False)
        p(f"  {ma}, tiền tố {k}: " + "; ".join(f"{r['bien']} {vn(r['tb_abs_shap'], '.3f')}" for _, r in g.head(8).iterrows()))
        r7 = g[g["ma_bien"].str.startswith("r7_")]
        p("    Thông tin R7: " + "; ".join(f"{r['ma_bien']} {vn(r['tb_abs_shap'], '.3f')} (hạng "
                                           f"{int((g['tb_abs_shap'] > r['tb_abs_shap']).sum()) + 1}/{len(g)})"
                                           for _, r in r7.iterrows()))


# ---------------------------------------------------------------- tự kiểm
def log_gia_lap(n_case=1600, seed=7):
    """Log giả lập cho kiểm thử toàn luồng: dòng hàng tạo rải năm 2018, vài nhà cung cấp, R3/R2a/R6 cài có chủ đích
    (R3 hay gặp hơn khi có Change Quantity sớm, để mô hình có tín hiệu học)."""
    rng = np.random.default_rng(seed)
    t_dau, ng, ph = pd.Timestamp("2018-01-01 08:00"), sq3.MOT_NGAY, pd.Timedelta(minutes=1)
    rows = []
    for i in range(n_case):
        c = f"{4500000000 + i // 3}_{(i % 3 + 1) * 10:05d}"
        v = f"vendorID_{rng.integers(0, 12):04d}"
        tao = t_dau + pd.Timedelta(days=float(rng.uniform(0, 380)))
        u_tao = f"user_{rng.integers(1, 6):03d}"
        tin_hieu = rng.random() < 0.4  # Change Quantity sớm → R3 dễ xảy ra hơn
        ev = [(PO, tao, u_tao, 100.0 + i)]
        if rng.random() < 0.2:
            ev.append((VENDOR_INV, (tao - 3 * ng).normalize() + pd.Timedelta(hours=23, minutes=59), "NONE", np.nan))
        if tin_hieu:
            ev.append(("Change Quantity", tao + 2 * ng, u_tao, np.nan))
        gr_u = u_tao if rng.random() < 0.05 else "user_009"
        ev.append((GR, tao + float(rng.uniform(3, 20)) * ng, gr_u, np.nan))
        ir_t = tao + float(rng.uniform(10, 40)) * ng
        ev.append((IR, ir_t, "user_010", np.nan))
        if rng.random() < 0.05:  # làm lại: hủy rồi ghi lại hóa đơn
            ev.append((CANCEL_IR, ir_t + 2 * ng, "user_010", np.nan))
            ev.append((IR, ir_t + 2 * ng + 5 * ph, "user_010", np.nan))
        if rng.random() < (0.35 if tin_hieu else 0.03):
            ev.append((kc.CHANGE_PRICE, ir_t + float(rng.uniform(1, 30)) * ng, "user_011", np.nan))
        ev.append((CLEAR, ir_t + float(rng.uniform(30, 60)) * ng, "user_002", np.nan))
        for a, t, u, nw in ev:
            rows.append({"case": c, "act": a, "ts": t, "user": u, "vendor": v, "spend_area": f"hm{hash(v) % 3}",
                         "networth": nw})
    return sap_xep(pd.DataFrame(rows))


def tu_kiem():
    loi = mt.tu_kiem()
    print()

    def kt(ten, dat, ct=""):
        nonlocal loi
        loi += not dat
        print(f"{ten:72s} {'ĐẠT' if dat else 'SAI ' + str(ct)}")

    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.05).astype(int)
    s = np.round(rng.random(3000) + y * 0.3, 2)  # có điểm bằng nhau
    kt("PR-AUC tự viết bằng sklearn.average_precision_score (có điểm bằng nhau)",
       abs(ap_trong_so(chuan_bi_ap(s, y)) - average_precision_score(y, s)) < 1e-12)
    w = np.bincount(rng.integers(0, 3000, 3000), minlength=3000)
    kt("PR-AUC có trọng số bằng PR-AUC trên mẫu lặp lại", abs(ap_trong_so(chuan_bi_ap(s, y), w)
                                                          - average_precision_score(np.repeat(y, w), np.repeat(s, w))) < 1e-12)
    kt("Precision@k: điểm hoàn hảo cho 1,0", p_top(y.astype(float), y, 0.01) == 1.0)
    kq = bootstrap({"a": s, "b": s.copy()}, y, 50)
    _, ss = tom_tat_boot(kq, [("a", "b")])
    kt("Bootstrap ghép cặp: hai mô hình giống nhau có chênh lệch 0", ss[0]["chenh_pr_auc"] == 0 and ss[0]["p_hai_phia"] == 1)

    nh = np.repeat(np.arange(600), 5)
    kq2 = bootstrap({"a": s, "b": s.copy()}, y, 50, nhom=nh)
    _, ss2 = tom_tat_boot(kq2, [("a", "b")])
    kq3 = bootstrap({"a": s}, y, 200, nhom=nh)
    kt("Bootstrap theo đơn: chạy được, hai mô hình giống nhau chênh 0, PR-AUC trong [0; 1]",
       ss2[0]["chenh_pr_auc"] == 0 and np.nanmin(kq3["a"]) >= 0 and np.nanmax(kq3["a"]) <= 1)
    w1 = np.bincount(np.random.default_rng(3).integers(0, 600, 600), minlength=600)[nh]
    kt("Bootstrap theo đơn: mọi dòng cùng đơn có cùng trọng số", all(len(set(w1[nh == i])) == 1 for i in range(600)))

    hh = holm([0.01, 0.04, 0.03, 0.005])
    kt("Holm: ví dụ tính tay (0,005; 0,01; 0,03; 0,04) → (0,02; 0,03; 0,06; 0,06)",
       np.allclose(hh, [0.03, 0.06, 0.06, 0.02]), hh)

    # target encoding và chia phần
    d = pd.DataFrame({"don": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4", "P5", "P5"]},
                     index=[f"c{i}" for i in range(10)])
    yy = pd.Series([1, 0, 1, 0, 0, 0, 1, 0, 0, 1], index=d.index)
    ph = gan_phan(d, yy)
    kt("Chia phần: hai dòng cùng đơn mua hàng nằm cùng phần", all(ph.groupby(d["don"]).nunique() == 1), ph)
    vend = pd.Series(["v1"] * 2 + ["v2"] * 8, index=d.index)
    te_hl, _ = target_encoding(vend, yy.astype(float), ph, pd.Series(["v1", "v9"]))
    # dòng c0 (v1, nhãn 1): giá trị out-of-fold chỉ dùng dòng v1 khác phần
    khac = (vend == "v1") & (ph != ph["c0"])
    chung = yy[ph != ph["c0"]].mean()
    mong = (yy[khac].sum() + LAM_TRON_TE * chung) / (khac.sum() + LAM_TRON_TE)
    kt("Target encoding out-of-fold không dùng nhãn của chính phần đó", abs(te_hl["c0"] - mong) < 1e-12, (te_hl["c0"], mong))

    # biến tiền tố: không lấy sự kiện sau cuối tiền tố; R7 theo Vendor creates invoice / theo IR (độ nhạy 5)
    t0, ng = pd.Timestamp("2018-03-01 08:00"), sq3.MOT_NGAY
    ev = sap_xep(pd.DataFrame([
        {"case": "D", "act": VENDOR_INV, "ts": t0 - 2 * ng, "user": "NONE", "vendor": "v1", "spend_area": "x", "networth": np.nan},
        {"case": "D", "act": PO, "ts": t0, "user": "u1", "vendor": "v1", "spend_area": "x", "networth": 50.0},
        {"case": "D", "act": GR, "ts": t0 + 2 * ng, "user": "u2", "vendor": "v1", "spend_area": "x", "networth": 60.0},
        {"case": "D", "act": IR, "ts": t0 + 3 * ng, "user": "u3", "vendor": "v1", "spend_area": "x", "networth": 60.0},
        {"case": "D", "act": CLEAR, "ts": t0 + 30 * ng, "user": "user_002", "vendor": "v1", "spend_area": "x", "networth": 60.0}]))
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    dd = pd.DataFrame(index=pd.Index(["D"], name="case"))
    dd["tao"] = [t0]
    dd = bo_sung_dong_hang(dd, ev, {"D"}, ev["ts"], loai)
    hd = sorted(ev_tien_to(ev, loai, False)["act"].unique())
    e1 = ev_tien_to(ev, loai, False)
    x2 = bien_tien_to(dd, e1, mt.cuoi_tien_to(e1)[0].reindex(dd.index)[2], 2, hd, False)
    kt("Tiền tố 2 (có Vendor creates invoice): đếm VI và PO, không đếm GR; R7 trước ngày tạo = 1",
       x2.loc["D", "n_" + VENDOR_INV] == 1 and x2.loc["D", "n_" + PO] == 1 and x2.loc["D", "n_" + GR] == 0
       and x2.loc["D", "r7_vi_truoc_po"] == 1 and x2.loc["D", "ngay_den_cuoi_tien_to"] == 0, x2)
    kt("Giá trị lấy tại Create Purchase Order Item (50), không lấy sự kiện sau", abs(dd.loc["D", "log_gia_tri"] - np.log(50)) < 1e-12)
    kt("Clear Invoice không ghép với hủy được đếm là thanh toán", (e1["act"] == "Clear Invoice (thanh toán)").sum() == 1)
    e5 = ev_tien_to(ev, loai, True)
    tk5 = mt.cuoi_tien_to(e5)[0].reindex(dd.index)
    x5_2 = bien_tien_to(dd, e5, tk5[2], 2, hd, True)
    x5_3 = bien_tien_to(dd, e5, tk5[3], 3, hd, True)
    kt("Độ nhạy (5): bỏ VI; R7 chỉ có từ khi tiền tố có IR (tiền tố 2: 0; tiền tố 3: 1)",
       x5_2.loc["D", "n_" + VENDOR_INV] == 0 and x5_2.loc["D", "r7_vi_truoc_po"] == 0 and x5_3.loc["D", "r7_vi_truoc_po"] == 1,
       (x5_2, x5_3))

    # toàn luồng trên log giả lập, lưới nhỏ và 50 lần bootstrap
    import tempfile
    evg = log_gia_lap()
    pv = set(evg["case"])
    with tempfile.TemporaryDirectory() as tm:
        ra = Path(tm)
        dong = []
        try:
            phan_tich(evg, pv, pd.Timestamp("2019-01-18 13:34"), ra, dong.append, lambda: None, chi_chinh=True, so_boot=50,
                      luoi_lr=LUOI_LR[:2], luoi_xgb=LUOI_XGB[:2], ky_vong={})
            cs = pd.read_csv(ra / "chi_so.csv")
            ok = (len(cs) == 2 * len(CAC_K) * 4 or len(cs) > 0) and cs["pr_auc"].between(0, 1).all()
            x = cs[(cs["ma"] == "R3") & (cs["tien_to"] == 2)].set_index("mo_hinh")
            kt("Toàn luồng chạy được; mọi PR-AUC trong [0; 1]", ok, cs.head())
            kt("Log giả lập: tiền tố 2 của R3 có tín hiệu, XGBoost vượt mốc ngẫu nhiên",
               x.loc["XGBoost", "pr_auc"] > x.loc["XGBoost", "ty_le_nhan_1"], x[["pr_auc", "ty_le_nhan_1"]])
            kt("Có file theo tháng, theo luật, SHAP",
               all((ra / f).exists() for f in ["theo_thang.csv", "theo_luat.csv", "shap.csv", "tham_so.json"]))
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            kt("Toàn luồng trên log giả lập", False, ex)
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
    ap.add_argument("--ra", default="ket_qua_sq4_mo_hinh")
    ap.add_argument("--chi-chinh", action="store_true", help="chỉ nhãn chính và nhãn R3 (H = 90), bỏ các độ nhạy")
    ap.add_argument("--ma", nargs="+", help="chỉ chạy các cấu hình này, ví dụ: --ma nhan_cong_R1")
    ap.add_argument("--tien-to", nargs="+", type=int, help="chỉ chạy các độ dài tiền tố này, ví dụ: --tien-to 4 5")
    ap.add_argument("--bo-tu-kiem", action="store_true", help="không chạy lại kiểm thử giả lập trước khi chạy")
    ap.add_argument("--so-luong", type=int, default=-1, help="số luồng CPU cho XGBoost (mặc định: tất cả). Đổi số luồng làm kết quả XGBoost không lặp lại đúng lần chạy trước")
    ap.add_argument("--bootstrap-dong", action="store_true", help="bootstrap theo dòng hàng như lần chạy đầu (mặc định: theo đơn)")
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    global BOOT_THEO_DON, SO_LUONG
    BOOT_THEO_DON = not a.bootstrap_dong
    SO_LUONG = a.so_luong
    if SO_LUONG != -1:
        print("CẢNH BÁO: --so-luong khác mặc định; kết quả XGBoost có thể không trùng các lần chạy với đủ luồng.")
    if a.tu_kiem:
        sys.exit(tu_kiem())
    if not a.log:
        ap.error("cần đường dẫn log hoặc --tu-kiem")
    if not a.bo_tu_kiem and tu_kiem():
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
        (ra / "tom_tat_sq4_mo_hinh.txt").write_text("\n".join(out), encoding="utf-8")

    import datetime
    p(f"Chạy lúc {datetime.datetime.now():%d/%m/%Y %H:%M} (giờ máy). Lệnh: python {' '.join(sys.argv)}")
    t = time.time()
    raw = sq2.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    p(f"Đọc log xong sau {vn(round(time.time() - t))} giây.")
    phan_tich(ev, pv, moc, ra, p, luu, chi_chinh=a.chi_chinh, chi_ma=a.ma, chi_k=a.tien_to)
    luu()
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_sq4_mo_hinh.txt.")


if __name__ == "__main__":
    main()
