"""Các kiểm tra bổ sung theo góp ý G3 (13/10/2026) của thầy.

Cách chạy (đặt cùng thư mục với r5_v2_phan_loai_clear.py và kiem_tra_trang_thai_case.py):
    python src/g3_kiem_tra_bo_sung.py
    python src/g3_kiem_tra_bo_sung.py --tu-kiem        # chạy thử trên vết giả lập

Kết quả ghi vào ket_qua_g3/ (đổi bằng --ra); tom_tat_g3.txt là bản tóm tắt, gửi lại file này.
Mục trong G3 được ghi ở đầu mỗi phần: 2.3(a)–(d), 2.4, 2.5, 3, 5.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import r5_v2_phan_loai_clear as r5
from r5_v2_phan_loai_clear import (CANCEL_IR, CANCEL_SUB, CLEAR, IR, MEMO, SUB_IR, buoc2_moc_trich_xuat,
                                   loc_pham_vi, phan_loai_clear, sap_xep)
import cau_hinh

SEED = 2026
DELTA = 60
# 40 case v2/v2.1 đã đọc tay ngày 13/10, loại khỏi mẫu mới (G3 2.3c)
DA_DOC = set("""
4507001660_00010 4507004571_00010 4507005916_00010 4507009485_00140 4507009485_00150 4507014665_00120
4507017329_00010 4507017721_00020 4507017976_00010 4507020530_00010 4507027027_00020 4507027294_00010
4507027836_00010 4507030709_00010 4507031904_00390 4507036502_00010 4507041097_00010 4508043759_00020
4508045757_00010 4508049042_00020 4507000238_00100 4507002106_00010 4507002842_00030 4507004352_00130
4507009606_00010 4507013684_00100 4507015858_00090 4507016308_00010 4507018347_00010 4507018678_00010
4507022027_00020 4507024971_00020 4507025803_00001 4507035174_00040 4507036321_00010 4507037362_00020
4508042476_00050 4508047125_00030 4508051498_00010 4508065058_00100""".split()) | set(r5.CASE_DA_DOC)


def doc_log(duong_dan):
    """Đọc log như r5 và giữ thêm Cumulative net worth (EUR) của từng sự kiện (G3 2.3c, 3)."""
    p = Path(duong_dan)
    raw = pd.read_csv(p, encoding="latin-1", low_memory=False) if p.suffix.lower() == ".csv" else r5.doc_xes_gio_dia_phuong(p)
    df = r5.chuan_hoa_cot(raw)
    for c in ["Cumulative net worth (EUR)", "event Cumulative net worth (EUR)"]:
        if c in raw.columns:
            df["networth"] = pd.to_numeric(raw.loc[df.index, c], errors="coerce").values
            break
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- bộ đếm theo thời điểm (mã giả P.1 của thầy)
def bo_dem(ev, loai, tinh_hd_bo_sung=False):
    """Trả về bảng các Clear "thanh toán" với: hop_le, n_paid, n_paid_tg (đếm theo thời điểm khác nhau),
    nhom (A/B), vuot (lần trả vượt), vuot_tg. Sự kiện cùng timestamp: IR/Cancel/memo xử lý trước Clear (sap_xep)."""
    acts = [IR, CANCEL_IR, MEMO, CLEAR] + ([SUB_IR, CANCEL_SUB] if tinh_hd_bo_sung else [])
    e = ev[ev["act"].isin(acts)].copy()
    tt = set(loai.index[loai == "thanh_toan"])
    e = e[(e["act"] != CLEAR) | e.index.isin(tt)]
    g = e.groupby("case")
    lk = {k: (e["act"] == a).astype(int).groupby(e["case"]).cumsum()
          for k, a in {"ir": IR, "huy": CANCEL_IR, "memo": MEMO, "tra": CLEAR, "sub": SUB_IR, "huy_sub": CANCEL_SUB}.items()}
    hop_le = (lk["ir"] - lk["huy"]).clip(lower=0) + lk["memo"]
    if tinh_hd_bo_sung:
        hop_le = hop_le + (lk["sub"] - lk["huy_sub"]).clip(lower=0)
    la_tra = e["act"] == CLEAR
    # đếm theo số thời điểm thanh toán khác nhau (G3 2.3a)
    moi_tg = la_tra & ~(e["case"].eq(e["case"].shift()) & e["ts"].eq(e["ts"].where(la_tra).groupby(e["case"]).ffill().shift()))
    tra_tg = moi_tg.astype(int).groupby(e["case"]).cumsum()
    out = e.loc[la_tra, ["case", "ts"] + [c for c in ("user", "networth") if c in e]].copy()
    out["hop_le"] = hop_le[la_tra]
    out["n_paid"] = lk["tra"][la_tra]
    out["n_paid_tg"] = tra_tg[la_tra]
    out["nhom"] = np.where(out["hop_le"] == 0, "B", "A")
    out["vuot"] = out["n_paid"] > out["hop_le"]
    out["vuot_tg"] = out["n_paid_tg"] > out["hop_le"]
    del g
    return out


def la_batch(u):
    return u.astype(str).str.lower().str.startswith("batch")


def lay_mau(ev, cases, n, rng, ten, ra, them=None):
    ds = sorted(set(cases) - DA_DOC)
    chon = list(rng.choice(ds, size=min(n, len(ds)), replace=False)) if ds else []
    cot = ["case", "ts", "act"] + [c for c in ("loai_clear", "user", "networth") if c in ev]
    vet = ev[ev["case"].isin(chon)][cot].copy()
    if them is not None:
        vet = vet.merge(them, how="left", left_index=True, right_index=True)
    vet["nhan_xet"] = ""
    vet.to_csv(ra / "mau_kiem_chung" / f"{ten}.csv", index=False, encoding="utf-8-sig")
    return len(ds), len(chon)


def ve_histogram(phut, duong_dan, tieu_de):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = np.log10(phut[phut > 0].dropna())
    if not len(x):
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(x, bins=np.arange(0, 6.01, 0.1))
    for v, nhan in [(60, "1 giờ"), (1440, "1 ngày"), (10080, "1 tuần"), (43200, "30 ngày")]:
        ax.axvline(np.log10(v), ls="--", c="grey")
        ax.text(np.log10(v), ax.get_ylim()[1] * 0.95, nhan, rotation=90, va="top", ha="right")
    ax.set_xlabel("khoảng cách (phút, thang log10)")
    ax.set_ylabel("số Clear")
    ax.set_title(tieu_de)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150)
    plt.close(fig)


def hinh_chon_delta(ev, ra, p):
    """G3 Mục 5: hình histogram khoảng cách Cancel IR → Clear kế tiếp (thang log), đánh dấu Δ = 60 phút
    và điểm trũng 1.000 phút, kèm đường tỷ lệ Clear ghép được do chính người hủy thực hiện theo Δ."""
    kc = r5.khoang_cach_cancel_clear(ev)["phut"]
    luoi = [1, 5, 10, 30, 60, 120, 240, 480, 1000, 1440, 2880, 10080, 43200]
    dong = []
    for d in luoi:
        loai, _, ng = phan_loai_clear(ev, d)
        dong.append({"delta_phut": d, "so_clear_ghep": int((loai == "bu_tru_huy").sum()),
                     "ty_le_cung_nguoi": float(np.mean(ng)) if ng else np.nan})
    bang = pd.DataFrame(dong)
    bang.to_csv(ra / "g3_hinh_delta_cung_nguoi.csv", index=False, encoding="utf-8-sig")
    p("Tỷ lệ Clear ghép được do chính người hủy thực hiện, theo Δ: "
      + "; ".join(f"{r.delta_phut:,} phút {r.ty_le_cung_nguoi:.1%} ({r.so_clear_ghep:,} Clear)".replace(",", ".") for r in bang.itertuples()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        p("(không có matplotlib, bỏ qua hình)")
        return
    x = np.log10(kc[kc > 0].dropna())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(x, bins=np.arange(0, 6.01, 0.1), color="#8fb3d9")
    ax.set_xlabel("Khoảng cách từ Cancel Invoice Receipt đến Clear Invoice kế tiếp (phút, thang log)")
    ax.set_ylabel("Số lần hủy")
    ve = [1, 10, 60, 1000, 10080, 43200, 525600]
    ax.set_xticks(np.log10(ve))
    ax.set_xticklabels(["1", "10", "60", "1.000", "1 tuần", "30 ngày", "1 năm"])
    ax.axvline(np.log10(60), c="#c0392b", lw=2)
    ax.text(np.log10(60), ax.get_ylim()[1] * 0.97, " Δ = 60 phút", color="#c0392b", va="top")
    ax.axvline(np.log10(1000), c="grey", ls="--")
    ax.text(np.log10(1000), ax.get_ylim()[1] * 0.88, " điểm trũng 1.000 phút", color="grey", va="top")
    ax2 = ax.twinx()
    b = bang.dropna()
    ax2.plot(np.log10(b["delta_phut"]), b["ty_le_cung_nguoi"] * 100, "o-", c="#27ae60")
    ax2.set_ylabel("% Clear ghép được do chính người hủy (theo Δ)", color="#27ae60")
    ax2.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(ra / "hinh_3_chon_delta.png", dpi=200)
    plt.close(fig)


def do_chi_tiet_ts(ev, ra, p):
    """Phụ lục C: độ chi tiết timestamp của từng hoạt động. Hoạt động 'ghi đến ngày' có giờ:phút gần như cố định."""
    hm = ev["ts"].dt.strftime("%H:%M")
    dong = []
    for a, x in hm.groupby(ev["act"]):
        vc = x.value_counts()
        dong.append({"act": a, "so_su_kien": len(x), "gio_pho_bien": vc.index[0], "ty_le_gio_pho_bien": round(vc.iloc[0] / len(x), 4),
                     "do_chi_tiet": "đến ngày" if vc.iloc[0] / len(x) >= 0.9 else "đến phút"})
    bang = pd.DataFrame(dong).sort_values("so_su_kien", ascending=False)
    bang.to_csv(ra / "g3_phu_luc_c_do_chi_tiet_ts.csv", index=False, encoding="utf-8-sig")
    p(bang.to_string(index=False))


# ---------------------------------------------------------------- phân tích
def phan_tich(ev, pv, moc, ra, p, rng):
    def pct(n, N):
        return f"{n:,} ({n / N:.2%})".replace(",", "X").replace(".", ",").replace("X", ".") if N else f"{n} (–)"

    N = len(pv)
    loai, kg, _ = phan_loai_clear(ev, DELTA)
    ev = ev.copy()
    ev["loai_clear"] = loai.reindex(ev.index)
    bd = bo_dem(ev, loai)
    r5v21 = set(bd.loc[bd["vuot"], "case"])
    d = r5.dem_theo_case(ev, pv, loai)
    r5v2 = set(d.index[d["n_clear_tt"] > d["n_ir"] - d["n_cancel"] + d["n_memo"]])
    p(f"Phạm vi {N:,} case. Δ = {DELTA} phút. R5 v2 = {len(r5v2):,}; R5 v2.1 = {len(r5v21):,}")

    # ---- 2.3(a) thanh toán cùng thời điểm
    p("\n=== G3 2.3(a). Thanh toán cùng thời điểm trong các case R5 v2.1 ===")
    v = bd[bd["vuot"]]
    tra = bd[["case", "ts"]]
    so_cung_ts = tra.groupby(["case", "ts"]).size().rename("so_clear_cung_ts")
    v = v.join(so_cung_ts, on=["case", "ts"])
    case_vuot_cung_ts = set(v.loc[v["so_clear_cung_ts"] > 1, "case"])
    case_vuot_khac_ts = set(v.loc[v["so_clear_cung_ts"] == 1, "case"])
    r5v21_tg = set(bd.loc[bd["vuot_tg"], "case"])
    p(f"Case có lần trả vượt nằm cùng timestamp với Clear thanh toán khác: {pct(len(case_vuot_cung_ts), len(r5v21))}")
    p(f"  trong đó CHỈ có lần vượt cùng timestamp (không có lần vượt ở thời điểm riêng): "
      f"{pct(len(case_vuot_cung_ts - case_vuot_khac_ts), len(r5v21))}")
    p(f"R5 v2.1 đếm theo số thời điểm thanh toán khác nhau (độ nhạy): {pct(len(r5v21_tg), N)}")

    # ---- 2.3(b) 62 case chỉ v2.1 bắt
    p("\n=== G3 2.3(b). Case chỉ v2.1 bắt (không bị v2 bắt) ===")
    chi21 = r5v21 - r5v2
    ir_ts = ev.loc[ev["act"] == IR, ["case", "ts"]]
    vuot_dau = v.groupby("case")["ts"].min()
    co_ir_sau = ir_ts.join(vuot_dau.rename("ts_vuot"), on="case").query("ts > ts_vuot")["case"].unique()
    tra_truoc_hd = chi21 & set(co_ir_sau)
    p(f"Số case: {len(chi21):,}. Có Record Invoice Receipt xảy ra SAU lần trả vượt đầu tiên "
      f"(trả trước khi ghi nhận hóa đơn): {pct(len(tra_truoc_hd), len(chi21))}")
    p(f"Trên toàn bộ R5 v2.1, kiểu 'trả trước khi ghi nhận hóa đơn': {pct(len(r5v21 & set(co_ir_sau)), len(r5v21))}")

    # ---- 2.3(c) networth
    p("\n=== G3 2.3(c). Cumulative net worth ===")
    if "networth" in ev and ev["networth"].notna().any():
        nw = ev.groupby("case")["networth"].nunique()
        p(f"Case có Cumulative net worth thay đổi trong dòng hàng: {pct(int((nw > 1).sum()), len(nw))}")
        nw21 = nw.reindex(sorted(r5v21))
        p(f"  riêng các case R5 v2.1: {pct(int((nw21 > 1).sum()), len(nw21))}"
          + (" → giá trị gần như không đổi trong dòng hàng, so sánh 'bằng nhau' giữa hai lần trả KHÔNG có ý nghĩa"
             if (nw21 > 1).mean() < 0.05 else ""))
        hai = v.merge(bd[["case", "ts", "networth"]].rename(columns={"ts": "ts_truoc", "networth": "nw_truoc"}), on="case")
        hai = hai[hai["ts_truoc"] < hai["ts"]]
        bang_nhau = hai.groupby("case").apply(lambda x: bool((x["nw_truoc"] == x["networth"]).any()))
        p(f"Case R5 v2.1 có lần trả vượt cùng net worth với một lần trả trước đó: "
          f"{pct(int(bang_nhau.sum()), len(r5v21))}")
    else:
        p("Không đọc được cột Cumulative net worth.")

    # ---- 2.3(d) phân loại tự động 879 case
    p("\n=== G3 2.3(d). Phân loại tự động các case R5 v2.1 theo mẫu vết ===")
    dd = d.loc[sorted(r5v21)]
    co_huy, co_memo = dd["n_cancel"] > 0, dd["n_memo"] > 0
    cung = dd.index.isin(case_vuot_cung_ts - case_vuot_khac_ts)
    lop = np.select([cung, co_huy & co_memo, co_huy, co_memo, ~co_huy & ~co_memo],
                    ["chỉ vượt ở Clear cùng timestamp", "có hủy và debit memo", "có hủy, không memo",
                     "có debit memo, không hủy", "không hủy, không memo"], "khác")
    lop = pd.Series(lop, index=dd.index, name="loai_mau_vet")
    lop = lop.where(~lop.index.isin(tra_truoc_hd) | (lop == "chỉ vượt ở Clear cùng timestamp"),
                    "trả trước khi ghi nhận hóa đơn (chỉ v2.1)")
    bang_d = lop.value_counts().rename("so_case").to_frame()
    bang_d["ty_le"] = (bang_d["so_case"] / len(r5v21)).round(4)
    bang_d.to_csv(ra / "g3_2_3d_phan_loai_r5v21.csv", encoding="utf-8-sig")
    lop.to_csv(ra / "g3_2_3d_tung_case.csv", encoding="utf-8-sig")
    p(bang_d.to_string())

    # ---- 2.4 debit memo và hóa đơn bổ sung
    p("\n=== G3 2.4. Debit memo và Record/Cancel Subsequent Invoice ===")
    c_memo = set(ev.loc[ev["act"] == MEMO, "case"])
    c_sub = set(ev.loc[ev["act"] == SUB_IR, "case"])
    p(f"Case có debit memo: {len(c_memo):,}; có Record Subsequent Invoice: {len(c_sub):,}; có cả hai: {len(c_memo & c_sub):,} "
      f"({len(c_memo & c_sub) / max(1, len(c_sub)):.1%} số case có hóa đơn bổ sung)")
    m_ts = ev.loc[ev["act"] == MEMO, ["case", "ts"]].sort_values("ts")
    s_ts = ev.loc[ev["act"] == SUB_IR, ["case", "ts"]].rename(columns={"ts": "ts_sub"}).sort_values("ts_sub")
    if len(s_ts) and len(m_ts):
        gh = pd.merge_asof(s_ts, m_ts.rename(columns={"ts": "ts_memo"}), left_on="ts_sub", right_on="ts_memo",
                           by="case", direction="backward")
        ngay = (gh["ts_sub"] - gh["ts_memo"]).dt.total_seconds() / 86400
        p(f"Record Subsequent Invoice có debit memo trước đó trong cùng dòng hàng: {pct(int(ngay.notna().sum()), len(gh))}; "
          f"khoảng cách (ngày) P25/P50/P75 = {ngay.quantile(.25):.1f}/{ngay.median():.1f}/{ngay.quantile(.75):.1f}")
    cs = ev.loc[ev["act"] == CANCEL_SUB, ["case", "ts"]].sort_values("ts")
    kcl = ev.loc[ev["act"] == CLEAR, ["case", "ts"]].rename(columns={"ts": "ts_clear"}).sort_values("ts_clear")
    if len(cs):
        m = pd.merge_asof(cs, kcl, left_on="ts", right_on="ts_clear", by="case", direction="forward")
        ph = (m["ts_clear"] - m["ts"]).dt.total_seconds() / 60
        p(f"Cancel Subsequent Invoice → Clear kế tiếp ({len(m):,} lần): "
          f"≤ 60 phút {pct(int((ph <= 60).sum()), len(m))}; 1 giờ–7 ngày {pct(int(((ph > 60) & (ph <= 10080)).sum()), len(m))}; "
          f"> 7 ngày {pct(int((ph > 10080).sum()), len(m))}; không có Clear sau {pct(int(ph.isna().sum()), len(m))}")
        m.assign(phut=ph).to_csv(ra / "g3_2_4_khoang_cach_cancel_sub_clear.csv", index=False, encoding="utf-8-sig")
    bd_sub = bo_dem(ev, loai, tinh_hd_bo_sung=True)
    r5_sub = set(bd_sub.loc[bd_sub["vuot"], "case"])
    p(f"Độ nhạy R5 v2.1 + max(0, #Record Subsequent − #Cancel Subsequent): {pct(len(r5_sub), N)} "
      f"(bớt {len(r5v21 - r5_sub):,}, thêm {len(r5_sub - r5v21):,})")

    # ---- 2.5 hai chỉ báo phụ
    p("\n=== G3 2.5. Hai chỉ báo phụ (không vào nhãn) ===")
    # (1) hủy hóa đơn sau khi đã thanh toán: tại lần hủy, số lần đã trả >= số chứng từ hợp lệ trước lần hủy
    e = ev[ev["act"].isin([IR, CANCEL_IR, MEMO, CLEAR])].copy()
    e = e[(e["act"] != CLEAR) | (e["loai_clear"] == "thanh_toan")]
    lk = {k: (e["act"] == a).astype(int).groupby(e["case"]).cumsum() for k, a in
          {"ir": IR, "huy": CANCEL_IR, "memo": MEMO, "tra": CLEAR}.items()}
    hop_le_truoc = (lk["ir"] - (lk["huy"] - 1)).clip(lower=0) + lk["memo"]
    la_huy = e["act"] == CANCEL_IR
    huy_sau_tra = la_huy & (lk["tra"] >= hop_le_truoc) & (lk["tra"] > 0)
    hs = e.loc[huy_sau_tra, ["case", "ts"] + (["user"] if "user" in e else [])].copy()
    lan_tra_cuoi = e.loc[e["act"] == CLEAR, ["case", "ts"]].rename(columns={"ts": "ts_tra"}).sort_values("ts_tra")
    hs = pd.merge_asof(hs.sort_values("ts"), lan_tra_cuoi, left_on="ts", right_on="ts_tra", by="case", direction="backward")
    tre = (hs["ts"] - hs["ts_tra"]).dt.total_seconds() / 86400
    p(f"(1) Hủy hóa đơn sau khi đã thanh toán: {pct(hs['case'].nunique(), N)} dòng hàng, {len(hs):,} lần hủy; "
      f"độ trễ từ lần trả gần nhất (ngày) P25/P50/P75 = {tre.quantile(.25):.0f}/{tre.median():.0f}/{tre.quantile(.75):.0f}")
    if "user" in hs:
        p(f"    do tài khoản batch: {pct(int(la_batch(hs['user']).sum()), len(hs))}; 5 người hủy nhiều nhất: "
          + ", ".join(f"{k} {v:,}" for k, v in hs["user"].value_counts().head(5).items()))
        hs["thu"] = hs["ts"].dt.day_name()
        p("    theo thứ trong tuần: " + ", ".join(f"{k} {v}" for k, v in hs["thu"].value_counts().items()))
    hs.to_csv(ra / "g3_2_5_huy_sau_khi_tra.csv", index=False, encoding="utf-8-sig")
    # (2) hủy không có ghi nhận tương ứng
    huy_hon = d.index[d["n_cancel"] > d["n_ir"]]
    huy_sub_hon = d.index[d["n_cancel_sub"] > d["n_sub"]]
    p(f"(2) #Cancel Invoice Receipt > #Record Invoice Receipt: {pct(len(huy_hon), N)} dòng hàng; "
      f"#Cancel Subsequent > #Record Subsequent: {pct(len(huy_sub_hon), N)} dòng hàng "
      f"(tổng {int(d['n_cancel_sub'].sum()):,} lần hủy so với {int(d['n_sub'].sum()):,} lần ghi nhận)")

    # ---- 3 nhóm A/B
    p("\n=== G3 3. Clear thanh toán khi không còn chứng từ phải trả (nhóm B) ===")
    B = bd[bd["nhom"] == "B"].copy()
    p(f"Clear thanh toán: {len(bd):,}; nhóm A (hợp lệ > 0): {pct(int((bd['nhom'] == 'A').sum()), len(bd))}; "
      f"nhóm B (hợp lệ = 0): {pct(len(B), len(bd))} ở {B['case'].nunique():,} dòng hàng")
    if len(B):
        if "user" in B:
            p(f"Nhóm B do tài khoản batch: {pct(int(la_batch(B['user']).sum()), len(B))}; "
              f"nhóm A do batch: {pct(int(la_batch(bd.loc[bd['nhom'] == 'A', 'user']).sum()), int((bd['nhom'] == 'A').sum()))}")
            p("  người thực hiện nhiều nhất (nhóm B): " + ", ".join(f"{k} {v:,}" for k, v in B["user"].value_counts().head(5).items()))
        huy = ev.loc[ev["act"] == CANCEL_IR, ["case", "ts"]].rename(columns={"ts": "ts_huy"}).sort_values("ts_huy")
        B = pd.merge_asof(B.sort_values("ts"), huy, left_on="ts", right_on="ts_huy", by="case", direction="backward")
        kc = (B["ts"] - B["ts_huy"]).dt.total_seconds() / 86400
        khoang = [("≤ 1 ngày", kc <= 1), ("1–7 ngày", (kc > 1) & (kc <= 7)), ("7–30 ngày", (kc > 7) & (kc <= 30)),
               ("30–90 ngày", (kc > 30) & (kc <= 90)), ("> 90 ngày", kc > 90), ("không có lần hủy trước", kc.isna())]
        p("Khoảng cách từ lần hủy gần nhất: " + "; ".join(f"{t} {pct(int(m.sum()), len(B))}" for t, m in khoang))
        p("Theo thứ trong tuần: " + ", ".join(f"{k} {v:,}" for k, v in B["ts"].dt.day_name().value_counts().items()))
        p("Theo ngày trong tháng (5 ngày nhiều nhất): " + ", ".join(f"ngày {k} {v:,}" for k, v in B["ts"].dt.day.value_counts().head(5).items()))
        ve_histogram(kc * 1440, ra / "g3_3_hinh_khoang_cach_huy_clear_nhom_b.png",
                     "Nhóm B: khoảng cách từ lần hủy gần nhất đến Clear")
        p("Theo giờ (5 giờ nhiều nhất): " + ", ".join(f"{k}h {v:,}" for k, v in B["ts"].dt.hour.value_counts().head(5).items()))
        ir_sau = ir_ts.merge(B[["case", "ts"]].rename(columns={"ts": "ts_b"}), on="case").query("ts > ts_b")["case"].unique()
        p(f"Dòng hàng nhóm B có Record Invoice Receipt xảy ra sau đó: {pct(len(set(ir_sau)), B['case'].nunique())}")
        r4v2 = r5.r4_theo_clear(ev, loai.index[loai == "thanh_toan"])
        p(f"Dòng hàng R4 v2 ({len(r4v2):,}) có Clear nhóm B: {pct(len(r4v2 & set(B['case'])), len(r4v2))}; "
          f"dòng hàng R5 v2.1 có Clear nhóm B: {pct(len(r5v21 & set(B['case'])), len(r5v21))}")
        if "networth" in B and B["networth"].notna().any():
            B = B.merge(ev.loc[ev["act"] == CANCEL_IR, ["case", "ts", "networth"]].rename(
                columns={"ts": "ts_huy", "networth": "nw_huy"}), on=["case", "ts_huy"], how="left")
            p(f"Net worth tại Clear nhóm B bằng net worth tại lần hủy gần nhất: "
              f"{pct(int((B['networth'] == B['nw_huy']).sum()), len(B))}")
        B.to_csv(ra / "g3_3_clear_nhom_b.csv", index=False, encoding="utf-8-sig")
        n_du, n_lay = lay_mau(ev, set(B["case"]), 20, rng, "nhom_b_clear_khi_khong_con_chung_tu", ra)
        p(f"Mẫu nhóm B: {n_du:,} case đủ điều kiện, lấy {n_lay}")

    # ---- mẫu 2.3(b), 2.3(c)
    p("\n=== Mẫu kiểm chứng mới ===")
    for ten, ds in [("chi_v21_bat", chi21), ("r5v21_mau_moi", r5v21)]:
        n_du, n_lay = lay_mau(ev, ds, 20, rng, ten, ra)
        p(f"{ten}: {n_du:,} case đủ điều kiện (đã loại 40 case đọc ngày 13/10), lấy {n_lay}")

    # ---- 5. hai con số 1.170 và 2.096 case đổi trạng thái
    p("\n=== G3 5. Kiểm tra số liệu ===")
    de_xuat = d["n_clear"] > d["n_ir"]
    giai_thich = set(d.index[de_xuat & (d["n_memo"] > 0) & (d["n_clear"] <= d["n_ir"] + d["n_memo"])])
    bo = r5v2 - r5v21
    p(f"Tập 'dư được giải thích bằng debit memo' = {len(giai_thich):,}; tập 'v2 bắt nhưng v2.1 bỏ' = {len(bo):,}; "
      f"phần giao = {len(giai_thich & bo):,}. "
      + ("HAI TẬP GIỐNG NHAU, cần xem lại code." if giai_thich == bo else "Hai tập khác nhau, trùng số là ngẫu nhiên."))
    cu = r5.trang_thai(d, d["n_clear"], d["n_ir"] - d["n_cancel"], d["cuoi_clear"])
    moi = r5.trang_thai(d, d["n_clear_tt"], d["n_ir"] - d["n_cancel"] + d["n_memo"], d["cuoi_clear_tt"])
    doi = d.index[(cu == "Hoàn tất") & (moi == "Còn mở")]
    hop_le_cuoi = (d["n_ir"] - d["n_cancel"]).clip(lower=0) + d["n_memo"]
    cuoi_sk = ev.groupby("case")["ts"].max()
    im_lang = (pd.Timestamp(moc) - cuoi_sk).dt.days
    khong_con = d.index[hop_le_cuoi == 0]
    p(f"Case đổi từ Hoàn tất sang Còn mở: {len(doi):,}; trong đó số chứng từ hợp lệ cuối = 0: "
      f"{pct(len(set(doi) & set(khong_con)), len(doi))}")
    for nguong in (90, 180):
        dong = set(khong_con) & set(im_lang.index[im_lang > nguong])
        p(f"  ... và không có sự kiện nào trong {nguong} ngày trước thời điểm trích xuất: "
          f"{pct(len(set(doi) & dong), len(doi))}; trên toàn bộ 'Còn mở' mới: "
          f"{pct(len(set(moi.index[moi == 'Còn mở']) & dong), int((moi == 'Còn mở').sum()))}")
    pd.DataFrame({"trang_thai_moi": moi, "hop_le_cuoi": hop_le_cuoi, "so_ngay_khong_hoat_dong": im_lang.reindex(d.index)}) \
        .loc[doi].to_csv(ra / "g3_5_case_hoan_tat_sang_con_mo.csv", encoding="utf-8-sig")

    # ---- 5. sự kiện chỉ ghi đến ngày
    p("\n=== G3 5. Sự kiện chỉ ghi đến ngày ===")
    hm = ev["ts"].dt.strftime("%H:%M")
    for a in ["Vendor creates invoice", MEMO]:
        x = hm[ev["act"] == a]
        if len(x):
            p(f"{a}: {len(x):,} sự kiện; giờ phổ biến: " + ", ".join(f"{k} {v / len(x):.1%}" for k, v in x.value_counts().head(3).items()))

    # ---- 5. hình chọn Δ và độ chi tiết timestamp (Phụ lục C)
    p("\n=== G3 5. Hình chọn Δ ===")
    hinh_chon_delta(ev, ra, p)
    p("\n=== Phụ lục C. Độ chi tiết timestamp theo hoạt động (trong phạm vi) ===")
    do_chi_tiet_ts(ev, ra, p)

    # ---- Wilson cho các tỷ lệ đọc tay đã có
    p("\n=== Khoảng tin cậy Wilson 95% cho kết quả đọc tay ngày 13/10 ===")
    for nhan, k, n in [("R5 v2: báo nhầm", 15, 20), ("R5 v2.1: đáng nghi cao", 8, 20), ("R5 v2.1: đáng nghi cao + cần số tiền", 14, 20),
                       ("Nhóm bị loại v1→v2: loại đúng", 20, 20), ("Lần hủy không ghép: đúng", 20, 20)]:
        lo, hi = wilson(k, n)
        p(f"  {nhan}: {k}/{n} = {k / n:.0%} (KTC 95%: {lo:.0%}–{hi:.0%})")
    return dict(r5v21=len(r5v21), B=len(B) if len(B) else 0)


# ---------------------------------------------------------------- tự kiểm
def tu_kiem():
    t = pd.Timestamp("2018-03-01 08:00")
    m, ngay = pd.Timedelta(minutes=1), pd.Timedelta(days=1)
    PO, GR = r5.PO, r5.GR
    vet = {
        "A_cung_ts": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay), (CLEAR, 30 * ngay)],   # vượt cùng timestamp
        "B_tra_trung": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 30 * ngay), (CLEAR, 40 * ngay)],
        "C_tra_truoc_hd": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 10 * ngay), (CLEAR, 20 * ngay), (IR, 25 * ngay)],
        "D_nhom_b": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CANCEL_IR, 5 * ngay), (CLEAR, 12 * ngay)],
        "E_huy_sau_tra": [(PO, 0 * m), (GR, ngay), (IR, 2 * ngay), (CLEAR, 10 * ngay), (CANCEL_IR, 100 * ngay)],
    }
    df = pd.DataFrame([{"case": c, "act": a, "ts": t + dt, "user": "batch_01" if c == "D_nhom_b" and a == CLEAR else "user_1"}
                       for c, ds in vet.items() for a, dt in ds])
    ev = sap_xep(df)
    loai, _, _ = phan_loai_clear(ev, DELTA)
    bd = bo_dem(ev, loai)
    kq = {
        "R5 v2.1": sorted(set(bd.loc[bd["vuot"], "case"])),
        "R5 v2.1 theo thời điểm khác nhau": sorted(set(bd.loc[bd["vuot_tg"], "case"])),
        "nhóm B": sorted(set(bd.loc[bd["nhom"] == "B", "case"])),
    }
    ky_vong = {"R5 v2.1": ["A_cung_ts", "B_tra_trung", "C_tra_truoc_hd", "D_nhom_b"],
               "R5 v2.1 theo thời điểm khác nhau": ["B_tra_trung", "C_tra_truoc_hd", "D_nhom_b"],
               "nhóm B": ["D_nhom_b"]}
    loi = 0
    for k in kq:
        dat = kq[k] == ky_vong[k]
        loi += not dat
        print(f"{k:34s} {kq[k]}  {'ĐẠT' if dat else 'SAI, kỳ vọng ' + str(ky_vong[k])}")
    print("\nTất cả đều đạt." if loi == 0 else f"\n{loi} kiểm tra SAI.")
    return loi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(cau_hinh.LOG), help="đường dẫn log; mặc định trong cau_hinh.py")
    ap.add_argument("--cong-ty", default="companyID_0000")
    ap.add_argument("--ra", default="ket_qua_g3")
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

    raw_df = doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw_df, None)
    pv = loc_pham_vi(raw_df, a.cong_ty, moc)
    ev = sap_xep(raw_df[raw_df["case"].isin(pv)])
    p(f"Thời điểm trích xuất: {moc}")
    phan_tich(ev, pv, moc, ra, p, np.random.default_rng(SEED))
    (ra / "tom_tat_g3.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Đã chạy xong.")


if __name__ == "__main__":
    main()
