"""Kiểm tra tuân thủ cho SQ2 (chạy sau khi đóng băng bộ luật).

  1. Bộ luật trên 210.887 dòng hàng: số và tỷ lệ dòng hàng có dấu hiệu theo từng luật (R1, R2a, R3, R5 v2.2, R6, R7,
     biến thể phụ R7b), R5 dạng khoảng (tính / không tính debit memo), nhánh R5a/R5b, chỉ báo R2b; nhãn chính
     (R2a, R3, R6, R7b) và nhãn độ nhạy (thêm R1, R5); số dòng hàng theo số luật vi phạm và bảng chồng lấn giữa các luật.
     Định nghĩa luật lấy nguyên từ kiem_chung_cuoi.py (R1–R7b) và g4_kiem_tra.py (R5 v2.2), không viết lại.
  2. Phân bố theo hạng mục chi tiêu (Spend area text) và nhà cung cấp (Vendor): tỷ lệ theo nhóm, hệ số so với tỷ lệ chung,
     mức tập trung (10 nhà cung cấp có nhiều dòng hàng bị đánh dấu nhất chiếm bao nhiêu phần dấu hiệu).
  3. Tỷ trọng giá trị: giá trị dòng hàng = Cumulative net worth (EUR) tại Create Purchase Order Item đầu tiên; giá trị đã ẩn
     danh nên chỉ so sánh tương đối (tỷ trọng giá trị của dòng hàng bị đánh dấu so với tỷ trọng số dòng hàng).
  4. Alignment với mô hình chuẩn hai nhánh (QuyTrinhChuan_P2P_2nhanh.pnml của SQ1, cần pm4py): chỉ giữ 7 hoạt động của mô hình,
     tính theo biến thể (mỗi biến thể một lần, nhân với số dòng hàng) trên toàn bộ biến thể (PHU = 1,0; bản 21/10/2026 dùng
     các biến thể phủ 95% số dòng hàng). Báo cáo fitness, precision (Align-ETConformance), các bước lệch phổ biến, và độ nhạy trên tập dòng hàng chỉ có
     một lần nhận hàng và một lần ghi nhận hóa đơn.
Thứ tự sự kiện cùng timestamp theo sap_xep của r5 (GR trước IR trước Clear), như ở các script trước.

Cách chạy (cùng thư mục với sq1_kham_pha.py, kiem_chung_cuoi.py, g5_kiem_tra.py, g4_kiem_tra.py, g3_kiem_tra_bo_sung.py,
r5_v2_phan_loai_clear.py, kiem_tra_trang_thai_case.py; cần ket_qua_sq1/QuyTrinhChuan_P2P_2nhanh.pnml):
    python src/sq2_tuan_thu.py --tu-kiem
    python src/sq2_tuan_thu.py
Mục 1–3 chỉ cần pandas/numpy; mục 4 cần pm4py. Kết quả ghi vào ket_qua_sq2/ (đổi bằng --ra); gửi lại tom_tat_sq2.txt.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import g3_kiem_tra_bo_sung as g3
import g4_kiem_tra as g4
import g5_kiem_tra as g5
import kiem_chung_cuoi as kc
import r5_v2_phan_loai_clear as r5
import sq1_kham_pha as sq1
from r5_v2_phan_loai_clear import GR, IR, PO, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear, sap_xep
import cau_hinh

PHU = 1.0  # phần số dòng hàng được tính alignment; 23/10/2026 đổi từ 0,95 sang toàn bộ biến thể
MIN_NCC = 50  # nhà cung cấp cần ít nhất ngần này dòng hàng mới xếp hạng theo tỷ lệ
LUAT = ["R1", "R2a", "R3", "R5", "R6", "R7", "R7b"]
NHAN_CHINH = ["R2a", "R3", "R6", "R7b"]
NHAN_DO_NHAY = NHAN_CHINH + ["R1", "R5"]
TEN_LUAT = {
    "R1": "Thanh toán khi chưa nhận hàng (không vào nhãn chính)",
    "R2a": "Gỡ khóa trước khi nhận hàng",
    "R3": "Sửa giá sau hóa đơn",
    "R5": "Thanh toán vượt số chứng từ phải trả, v2.2 (không vào nhãn chính)",
    "R6": "Cùng người tạo đơn hàng và nhận hàng",
    "R7": "Ghi nhận hóa đơn trước khi tạo đơn hàng",
    "R7b": "Biến thể phụ R7: ngày hóa đơn nhà cung cấp trước ngày tạo đơn hàng",
}
KET_QUA_KIEM_CHUNG = {"R1": 654, "R2a": 508, "R3": 2008, "R5": 681, "R6": 650, "R7": 0, "R7b": 2430}  # 18/10/2026


def doc_log(duong_dan):
    """Như g3.doc_log (giữ Cumulative net worth) và giữ thêm Vendor, Spend area text."""
    p = Path(duong_dan)
    raw = pd.read_csv(p, encoding="latin-1", low_memory=False) if p.suffix.lower() == ".csv" else r5.doc_xes_gio_dia_phuong(p)
    df = r5.chuan_hoa_cot(raw)
    for chuan, ds in {"networth": ["Cumulative net worth (EUR)", "event Cumulative net worth (EUR)"],
                      "vendor": ["case:Vendor", "case Vendor"],
                      "spend_area": ["case:Spend area text", "case Spend area text"]}.items():
        for c in ds:
            if c in raw.columns:
                v = raw.loc[df.index, c]
                df[chuan] = pd.to_numeric(v, errors="coerce").values if chuan == "networth" else v.astype(str).values
                break
    return df


# ---------------------------------------------------------------- 1. bộ luật
def dau_hieu(ev, pv):
    """Bảng dòng hàng × luật (True/False), kèm nhánh R5a/R5b, R5 không tính memo, R2b, nhãn chính, nhãn độ nhạy."""
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    ts_dp = g4.gio_dia_phuong(ev, lambda *a: None)
    kq, r2a_khong_gr = kc.cac_luat(ev, pv, loai, ts_dp)
    bd = g4.dem_r5(ev, loai)
    s22, nhanh_r5, tra_truoc = g4.r5_v22(bd)
    s22_km, _, _ = g4.r5_v22(g4.dem_r5(ev, loai, tinh_memo=False))
    idx = pd.Index(sorted(pv), name="case")
    d = pd.DataFrame(index=idx)
    for k in ["R1", "R2a", "R3", "R6", "R7", "R7b"]:
        d[k] = idx.isin(kq[k][0])
    d["R5"] = idx.isin(s22)
    d = d[LUAT]
    d["R2a_chua_nhan_hang"] = idx.isin(r2a_khong_gr)
    d["R5_nhanh"] = nhanh_r5.reindex(idx).fillna("")
    d["R5_khong_tinh_memo"] = idx.isin(s22_km)
    d["tra_truoc_ghi_nhan_hd"] = idx.isin(tra_truoc)
    rpb = ev[ev["act"] == kc.RPB]
    d["co_rpb"] = idx.isin(set(rpb["case"]))
    d["R2b_rpb_nguoi_that"] = idx.isin(set(rpb.loc[kc.la_nguoi_that(rpb["user"]), "case"]))
    d["nhan_chinh"] = d[NHAN_CHINH].any(axis=1)
    d["nhan_do_nhay"] = d[NHAN_DO_NHAY].any(axis=1)
    d["so_luat_nhan_chinh"] = d[NHAN_CHINH].sum(axis=1)
    d["trang_thai"] = g5.trang_thai_cac_dinh_nghia(g5.so_lieu_case(ev, pv, loai))["moi"].reindex(idx)
    return d


def bang_ty_le(d, p):
    N = len(d)
    rows = []
    for k in LUAT + ["nhan_chinh", "nhan_do_nhay"]:
        n = int(d[k].sum())
        lo, hi = g3.wilson(n, N)
        rows.append({"luat": k, "noi_dung": TEN_LUAT.get(k, "Nhãn chính: R2a, R3, R6, R7b" if k == "nhan_chinh"
                                                         else "Nhãn độ nhạy: nhãn chính + R1, R5"),
                     "so_dong_hang": n, "ty_le": n / N, "ktc95_duoi": lo, "ktc95_tren": hi,
                     "kiem_chung_cuoi": KET_QUA_KIEM_CHUNG.get(k, np.nan)})
    bang = pd.DataFrame(rows)
    for r in bang.itertuples():
        dong = f"  {r.luat:13s} {g4.pct(r.so_dong_hang, N):>18s}  {r.noi_dung}"
        if not np.isnan(r.kiem_chung_cuoi) and r.kiem_chung_cuoi != r.so_dong_hang:
            dong += f"  !!! khác kiểm chứng cuối ({int(r.kiem_chung_cuoi):,})"
        p(dong)
    khop = all(bang.set_index("luat").loc[k, "so_dong_hang"] == v for k, v in KET_QUA_KIEM_CHUNG.items())
    p("  Số liệu khớp kiểm chứng cuối 18/10." if khop else "  !!! Có luật lệch so với kiểm chứng cuối, cần xem lại trước khi dùng.")
    return bang


def chong_lan(d):
    m = d[LUAT].astype(int)
    return m.T.dot(m)


# ---------------------------------------------------------------- 2. phân bố theo nhóm
def theo_nhom(d, nhom, cot):
    """Mỗi nhóm một dòng: số dòng hàng, số và tỷ lệ dấu hiệu theo từng luật, hệ số = tỷ lệ nhóm / tỷ lệ chung."""
    x = d[cot].groupby(nhom)
    bang = x.sum().astype(int)
    bang.insert(0, "so_dong_hang", x.size())
    chung = d[cot].mean()
    for k in cot:
        bang["ty_le_" + k] = bang[k] / bang["so_dong_hang"]
        bang["he_so_" + k] = bang["ty_le_" + k] / chung[k] if chung[k] > 0 else np.nan
    return bang.sort_values("so_dong_hang", ascending=False, kind="stable")


def tap_trung(d, nhom, k, top=10):
    """Phần dấu hiệu k nằm ở `top` nhóm có nhiều dòng hàng bị đánh dấu nhất, so với phần số dòng hàng của chính các nhóm đó;
    và số nhóm ít nhất chứa một nửa số dòng hàng bị đánh dấu."""
    dem = d.loc[d[k], :].groupby(nhom[d[k]]).size().sort_values(ascending=False, kind="stable")
    if dem.empty:
        return {"luat": k, "so_nhom_co_dau_hieu": 0}
    dau = dem.head(top)
    n_nua = int((dem.cumsum() / dem.sum() < 0.5 - 1e-12).sum()) + 1
    return {"luat": k, "so_nhom_co_dau_hieu": len(dem), f"phan_dau_hieu_top{top}": dau.sum() / dem.sum(),
            f"phan_dong_hang_top{top}": nhom.isin(dau.index).mean(), "so_nhom_chua_nua_dau_hieu": n_nua}


# ---------------------------------------------------------------- 3. giá trị
def gia_tri_dong_hang(ev, idx):
    """Cumulative net worth tại Create Purchase Order Item đầu tiên; thiếu thì lấy giá trị đầu tiên có trong dòng hàng."""
    if "networth" not in ev:
        return pd.Series(np.nan, index=idx)
    e = ev[ev["networth"].notna()]
    po = e[e["act"] == PO].groupby("case")["networth"].first()
    bat_ky = e.groupby("case")["networth"].first()
    return po.reindex(idx).fillna(bat_ky.reindex(idx))


def ty_trong_gia_tri(d, v):
    tong = v.sum()
    rows = []
    for k in LUAT + ["nhan_chinh", "nhan_do_nhay"]:
        m = d[k]
        rows.append({"luat": k, "so_dong_hang": int(m.sum()), "ty_trong_so_dong_hang": m.mean(),
                     "ty_trong_gia_tri": v[m].sum() / tong if tong else np.nan,
                     "trung_vi_gia_tri_co_dau_hieu": v[m].median(), "trung_vi_gia_tri_khong_dau_hieu": v[~m].median()})
    b = pd.DataFrame(rows)
    b["ti_so_gia_tri_tren_so_dong"] = b["ty_trong_gia_tri"] / b["ty_trong_so_dong_hang"]
    return b


# ---------------------------------------------------------------- 4. alignment
def bien_the_mo_hinh(ev, idx):
    """Chuỗi 7 hoạt động mô hình của mỗi dòng hàng, nối bằng MUI_TEN (chuỗi rỗng nếu không có hoạt động nào)."""
    e = ev[ev["act"].isin(sq1.HD_MO_HINH)]
    return e.groupby("case", sort=False)["act"].agg(sq1.MUI_TEN.join).reindex(idx).fillna("")


def chon_bien_the(vt, phu=PHU):
    """Các biến thể phổ biến nhất (giảm dần theo số dòng hàng) đủ phủ `phu`."""
    dem = vt.value_counts()
    dem = dem.iloc[np.lexsort((np.array(dem.index, dtype=object), -dem.values))]
    tl = dem.cumsum() / dem.sum()
    k = int((tl < phu - 1e-12).sum()) + 1
    return dem.head(min(k, len(dem)))


def tao_trace(vet, ten):
    from pm4py.objects.log.obj import Event, Trace
    t0 = pd.Timestamp("2018-01-01")
    tr = Trace(attributes={"concept:name": ten})
    for i, a in enumerate(vet.split(sq1.MUI_TEN) if vet else []):
        tr.append(Event({"concept:name": a, "time:timestamp": (t0 + pd.Timedelta(minutes=i)).to_pydatetime()}))
    return tr


def can_bien_the(ds_vet, net, im, fm):
    """Alignment mỗi biến thể một lần. Trả về DataFrame: vet, fitness, cost, lech (danh sách (loại bước, hoạt động))."""
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
    from pm4py.objects.log.obj import EventLog
    log = EventLog([tao_trace(v, f"bt{i}") for i, v in enumerate(ds_vet)])
    kq = alignments.apply(log, net, im, fm, parameters={"show_progress_bar": False})
    rows = []
    for v, a in zip(ds_vet, kq):
        lech = []
        for x in a["alignment"]:
            if x[0] == x[1] or (x[0] == ">>" and x[1] is None):
                continue
            lech.append(("thừa trong log" if x[1] == ">>" else "thiếu so với mô hình", x[0] if x[1] == ">>" else x[1]))
        rows.append({"vet": v, "fitness": float(a["fitness"]), "cost": a["cost"], "lech": lech})
    return pd.DataFrame(rows)


def precision_theo_tan_suat(dem, net, im, fm):
    """Align-ETConformance trên log gồm mỗi biến thể lặp lại đúng số dòng hàng (dùng chung một đối tượng Trace cho đỡ bộ nhớ)."""
    from pm4py.algo.evaluation.precision import algorithm as precision
    from pm4py.objects.log.obj import EventLog
    log = EventLog()
    for i, (v, n) in enumerate(dem.items()):
        tr = tao_trace(v, f"bt{i}")
        for _ in range(int(n)):
            log.append(tr)
    return precision.apply(log, net, im, fm, variant=precision.Variants.ALIGN_ETCONFORMANCE,
                           parameters={"show_progress_bar": False})


def tong_hop_alignment(dem, al):
    """Fitness trung bình có trọng số, tỷ lệ dòng hàng khớp hoàn toàn, bảng các bước lệch (theo số dòng hàng)."""
    x = al.set_index("vet").loc[list(dem.index)]
    w = dem.values
    fit = float(np.average(x["fitness"], weights=w))
    khop = float(w[(x["fitness"] >= 1 - 1e-12).values].sum() / w.sum())
    rows = []
    for (v, n), lech in zip(dem.items(), x["lech"]):
        for (loai, hd), so_lan in pd.Series(lech, dtype=object).value_counts().items() if lech else []:
            rows.append({"loai_buoc_lech": loai, "hoat_dong": hd, "so_dong_hang": int(n), "so_lan": int(n) * int(so_lan)})
    lech = (pd.DataFrame(rows, columns=["loai_buoc_lech", "hoat_dong", "so_dong_hang", "so_lan"])
            .groupby(["loai_buoc_lech", "hoat_dong"]).sum().reset_index()
            .sort_values(["so_dong_hang", "loai_buoc_lech", "hoat_dong"], ascending=[False, True, True], kind="stable"))
    lech["ty_le_dong_hang"] = lech["so_dong_hang"] / w.sum()
    return fit, khop, lech


def alignment(ev, d, ra, p, pm4py):
    duong_dan = Path("ket_qua_sq1") / sq1.TEN_PNML
    if duong_dan.exists():
        net, im, fm = pm4py.read_pnml(str(duong_dan))
        p(f"  Đọc mô hình chuẩn từ {duong_dan}.")
    else:
        net, im, fm = sq1.mo_hinh_chuan(pm4py)
        p(f"  Không thấy {duong_dan}, dựng lại mô hình chuẩn như SQ1.")
    kt = sq1.kiem_thu_mo_hinh(net, im, fm)
    if (kt["ket_qua"] != "ĐẠT").any():
        p("  !!! Mô hình không đạt 8 vết giả lập, dừng mục alignment.")
        return
    p("  Mô hình đạt 8 vết giả lập của SQ1.")
    N = len(d)
    vt = bien_the_mo_hinh(ev, d.index)
    n_gr = ev[ev["act"] == GR].groupby("case").size().reindex(d.index).fillna(0)
    n_ir = ev[ev["act"] == IR].groupby("case").size().reindex(d.index).fillna(0)
    mot_mot = (n_gr == 1) & (n_ir == 1)
    xong = d["trang_thai"] == "Hoàn tất"
    tap = [("toàn bộ phạm vi", pd.Series(True, index=d.index), ""),
           ("chỉ 1 lần nhận hàng và 1 lần ghi nhận hóa đơn", mot_mot, "_1gr_1ir"),
           ("dòng hàng hoàn tất", xong, "_hoan_tat"),
           ("hoàn tất, 1 lần nhận hàng và 1 lần ghi nhận hóa đơn", xong & mot_mot, "_hoan_tat_1gr_1ir")]
    dem_tap = [chon_bien_the(vt[m]) for _, m, _ in tap]
    dem = dem_tap[0]
    ds = list(dict.fromkeys(v for dm in dem_tap for v in dm.index))
    t = time.time()
    al = can_bien_the(ds, net, im, fm)
    p(f"  Tính alignment cho {len(ds)} biến thể ({time.time() - t:,.0f} giây).")
    al_ghi = al.assign(bien_the=al["vet"],
                       so_dong_hang=al["vet"].map(vt.value_counts()).astype(int),
                       lech=al["lech"].map(lambda ls: "; ".join(f"{a}: {b}" for a, b in ls)))
    al_ghi.drop(columns="vet").sort_values("so_dong_hang", ascending=False).to_csv(
        ra / "alignment_bien_the.csv", index=False, encoding="utf-8-sig")

    tong = []
    p(f"  Dòng hàng hoàn tất (định nghĩa G5): {g4.pct(int(xong.sum()), N)}. Dòng hàng còn mở thiếu các bước cuối "
      f"so với mô hình chỉ vì chưa đi hết quy trình tại thời điểm trích xuất.")
    for (ten, m, hau_to), dm in zip(tap, dem_tap):
        n_tap = int(m.sum())
        fit, khop, lech = tong_hop_alignment(dm, al)
        t = time.time()
        prec = precision_theo_tan_suat(dm, net, im, fm)
        p(f"\n  [{ten}] {n_tap:,} dòng hàng; {len(dm)} biến thể phủ {g4.pct(int(dm.sum()), n_tap)}.")
        p(f"    Fitness trung bình (theo dòng hàng): {fit:.4f}; dòng hàng khớp hoàn toàn mô hình: {khop:.2%}; "
          f"precision: {prec:.4f} ({time.time() - t:,.0f} giây)")
        p("    Các bước lệch phổ biến (tỷ lệ trên số dòng hàng được tính alignment):")
        p(lech.head(12).assign(ty_le_dong_hang=lambda x: (x["ty_le_dong_hang"] * 100).round(2)).to_string(index=False))
        lech.to_csv(ra / f"alignment_buoc_lech{hau_to}.csv", index=False, encoding="utf-8-sig")
        tong.append({"tap": ten, "so_dong_hang": n_tap, "so_bien_the": len(dm), "so_dong_hang_duoc_tinh": int(dm.sum()),
                     "fitness": fit, "ty_le_khop_hoan_toan": khop, "precision": prec})
    pd.DataFrame(tong).to_csv(ra / "alignment_tong_hop.csv", index=False, encoding="utf-8-sig")

    f = al.set_index("vet")["fitness"]
    fc = vt.map(lambda v: f.get(v, np.nan))
    fc[~vt.isin(set(dem.index))] = np.nan  # chỉ giữ dòng hàng thuộc tập biến thể được tính alignment của toàn phạm vi
    pd.DataFrame({"fitness": fc, "bien_the_mo_hinh": vt, "trang_thai": d["trang_thai"]}).to_csv(
        ra / "fitness_tung_dong_hang.csv", encoding="utf-8-sig")
    p("\n  Dấu hiệu theo luật và khớp mô hình (chỉ dòng hàng có alignment; cột _hoan_tat: chỉ dòng hàng hoàn tất):")
    rows = []
    co = fc.notna()
    nhom = [(k, d[k]) for k in LUAT + ["nhan_chinh"]] + [("không dấu hiệu nào", ~d["nhan_do_nhay"])]
    for k, mk in nhom:
        r = {"luat": k}
        for hau_to, mx in [("", co), ("_hoan_tat", co & xong)]:
            m = mk & mx
            r["so_dong_hang_co_alignment" + hau_to] = int(m.sum())
            r["ty_le_khong_khop_mo_hinh" + hau_to] = float((fc[m] < 1 - 1e-12).mean()) if m.any() else np.nan
        rows.append(r)
    b = pd.DataFrame(rows)
    b.to_csv(ra / "luat_va_alignment.csv", index=False, encoding="utf-8-sig")
    p(b.assign(**{c: b[c] * 100 for c in b.columns if c.startswith("ty_le")}).round(2).to_string(index=False))
    p("  (ty_le tính bằng %)")


# ---------------------------------------------------------------- phân tích
def phan_tich(ev, pv, moc, ra, p):
    N = len(pv)
    p(f"Phạm vi {N:,} dòng hàng, {len(ev):,} sự kiện. Δ = {g4.DELTA} phút. Thời điểm trích xuất: {moc}")

    p("\n=== 1. Tỷ lệ dấu hiệu theo luật ===")
    d = dau_hieu(ev, pv)
    bang = bang_ty_le(d, p)
    bang.to_csv(ra / "ty_le_theo_luat.csv", index=False, encoding="utf-8-sig")
    p(f"  R2a: trong đó chưa từng nhận hàng {g4.pct(int(d['R2a_chua_nhan_hang'].sum()), int(d['R2a'].sum()))}.")
    p(f"  R5 dạng khoảng: {g4.pct(int(d['R5'].sum()), N)} (tính debit memo, số chính) đến "
      f"{g4.pct(int(d['R5_khong_tinh_memo'].sum()), N)} (không tính debit memo). "
      f"Nhánh R5a {int((d['R5_nhanh'] == 'R5a').sum()):,}, R5b {int((d['R5_nhanh'] == 'R5b').sum()):,}. "
      f"Chỉ báo trả trước khi ghi nhận hóa đơn: {int(d['tra_truoc_ghi_nhan_hd'].sum()):,}.")
    p(f"  R2b (chỉ báo): Remove Payment Block do người dùng thật {g4.pct(int(d['R2b_rpb_nguoi_that'].sum()), int(d['co_rpb'].sum()))} "
      f"số dòng hàng có gỡ khóa.")
    p("  Số luật của nhãn chính trên một dòng hàng:\n" + d["so_luat_nhan_chinh"].value_counts().sort_index().to_string())
    cl = chong_lan(d)
    cl.to_csv(ra / "chong_lan_luat.csv", encoding="utf-8-sig")
    p("  Chồng lấn giữa các luật (số dòng hàng có cả hai dấu hiệu; đường chéo = số dòng hàng của luật):\n" + cl.to_string())
    d.to_csv(ra / "dau_hieu_tung_dong_hang.csv", encoding="utf-8-sig")

    cot = LUAT + ["nhan_chinh", "nhan_do_nhay"]
    p("\n=== 2. Phân bố theo hạng mục chi tiêu và nhà cung cấp ===")
    thuoc_tinh = ev.drop_duplicates("case").set_index("case").reindex(d.index)
    for ten_cot, ten, tep in [("spend_area", "hạng mục chi tiêu", "theo_hang_muc.csv"), ("vendor", "nhà cung cấp", "theo_nha_cung_cap.csv")]:
        if ten_cot not in thuoc_tinh:
            p(f"  Log không có cột {ten}, bỏ qua.")
            continue
        nhom = thuoc_tinh[ten_cot].fillna("(trống)").astype(str)
        b = theo_nhom(d, nhom, cot)
        b.to_csv(ra / tep, encoding="utf-8-sig")
        p(f"\n  {ten.capitalize()}: {len(b):,} nhóm.")
        tt = pd.DataFrame([tap_trung(d, nhom, k) for k in cot])
        tt.to_csv(ra / tep.replace(".csv", "_tap_trung.csv"), index=False, encoding="utf-8-sig")
        p("  Mức tập trung (top10 = 10 nhóm có nhiều dòng hàng bị đánh dấu nhất):\n" + tt.round(4).to_string(index=False))
        if ten_cot == "spend_area":
            hien = b.head(15)[["so_dong_hang"] + [f"ty_le_{k}" for k in NHAN_CHINH + ["nhan_chinh"]] + ["he_so_nhan_chinh"]]
            p("  15 hạng mục nhiều dòng hàng nhất (tỷ lệ tính bằng %):\n" + hien.assign(
                **{c: hien[c] * 100 for c in hien.columns if c.startswith("ty_le_")}).round(2).to_string())
        else:
            du = b[b["so_dong_hang"] >= MIN_NCC].copy()
            ci = du.apply(lambda r: g3.wilson(int(r["nhan_chinh"]), int(r["so_dong_hang"])), axis=1)
            du["ktc95_duoi_nhan_chinh"] = [c[0] for c in ci]
            du = du.sort_values(["ktc95_duoi_nhan_chinh", "so_dong_hang"], ascending=False, kind="stable")
            du.to_csv(ra / "nha_cung_cap_xep_hang.csv", encoding="utf-8-sig")
            p(f"  {len(du):,} nhà cung cấp có ≥ {MIN_NCC} dòng hàng. 15 nhà cung cấp có cận dưới KTC 95% của tỷ lệ nhãn chính "
              f"cao nhất (tỷ lệ tính bằng %):")
            hien = du.head(15)[["so_dong_hang", "nhan_chinh", "ty_le_nhan_chinh", "ktc95_duoi_nhan_chinh"] + NHAN_CHINH]
            p(hien.assign(ty_le_nhan_chinh=hien["ty_le_nhan_chinh"] * 100, ktc95_duoi_nhan_chinh=hien["ktc95_duoi_nhan_chinh"] * 100)
              .round(2).to_string())

    p("\n=== 3. Tỷ trọng giá trị (giá trị đã ẩn danh, chỉ so sánh tương đối) ===")
    v = gia_tri_dong_hang(ev, d.index)
    p(f"  Dòng hàng có giá trị: {g4.pct(int(v.notna().sum()), N)}; giá trị = 0: {int((v == 0).sum()):,}.")
    if v.notna().any():
        gt = ty_trong_gia_tri(d, v.fillna(0))
        gt.to_csv(ra / "ty_trong_gia_tri.csv", index=False, encoding="utf-8-sig")
        p(gt.round(4).to_string(index=False))
        p("  ti_so_gia_tri_tren_so_dong > 1: dòng hàng bị đánh dấu có giá trị lớn hơn trung bình.")

    p("\n=== 4. Alignment với mô hình chuẩn ===")
    pm4py = sq1.nap_pm4py()
    if pm4py is None:
        p("  Chưa có pm4py nên bỏ qua mục này. Cài bằng:  pip install pm4py  rồi chạy lại.")
        return
    alignment(ev, d, ra, p, pm4py)


# ---------------------------------------------------------------- tự kiểm
def tu_kiem():
    loi = 0

    def kt(ten, dat, chi_tiet=""):
        nonlocal loi
        loi += not dat
        print(f"{ten:52s} {'ĐẠT' if dat else 'SAI ' + str(chi_tiet)}")

    print("--- Luật (kiểm thử của kiem_chung_cuoi.py) ---")
    loi += kc.tu_kiem()
    print("--- SQ2 ---")
    # dau_hieu phải cho đúng các tập của kiem_chung_cuoi trên cùng log giả lập
    t = pd.Timestamp("2018-03-01 08:00")
    m, ngay = pd.Timedelta(minutes=1), pd.Timedelta(days=1)
    vet = {
        "A": [(PO, 0 * m, "u1"), (GR, ngay, "u2"), (IR, 2 * ngay, "u3"), ("Clear Invoice", 30 * ngay, "u4")],
        "B_r1": [(PO, 0 * m, "u1"), (IR, ngay, "u3"), ("Clear Invoice", 10 * ngay, "u4"), (GR, 20 * ngay, "u2")],
        "C_r2a": [(PO, 0 * m, "u1"), (IR, ngay, "u3"), (kc.RPB, 2 * ngay, "u3"), (GR, 5 * ngay, "u2")],
        "E_r3": [(PO, 0 * m, "u1"), (GR, ngay, "u2"), (IR, 2 * ngay, "u3"), ("Change Price", 3 * ngay, "u1")],
        "G_r6": [(PO, 0 * m, "u1"), (GR, ngay, "u1"), (IR, 2 * ngay, "u3")],
        # trả hai lần cho một hóa đơn → R5
        "R_r5": [(PO, 0 * m, "u1"), (GR, ngay, "u2"), (IR, 2 * ngay, "u3"), ("Clear Invoice", 10 * ngay, "u4"),
                 ("Clear Invoice", 17 * ngay, "u4")],
        "J_r7b": [(kc.VENDOR_INV, -9 * pd.Timedelta(hours=1) - m, "NONE"), (PO, 0 * m, "u1"), (GR, ngay, "u2"), (IR, 2 * ngay, "u3")],
    }
    gia = {"A": 100.0, "B_r1": 300.0, "C_r2a": 50.0, "E_r3": 50.0, "G_r6": 200.0, "R_r5": 100.0, "J_r7b": 200.0}
    ncc = {"A": "V1", "B_r1": "V1", "C_r2a": "V1", "E_r3": "V2", "G_r6": "V2", "R_r5": "V3", "J_r7b": "V3"}
    hm = {"A": "S1", "B_r1": "S1", "C_r2a": "S2", "E_r3": "S2", "G_r6": "S2", "R_r5": "S1", "J_r7b": "S1"}
    df = pd.DataFrame([{"case": c, "act": a, "ts": t + dt, "user": u, "networth": gia[c], "vendor": ncc[c], "spend_area": hm[c]}
                       for c, ds in vet.items() for a, dt, u in ds])
    ev = sap_xep(df)
    d = dau_hieu(ev, set(ev["case"]))
    ky_vong = {"R1": ["B_r1"], "R2a": ["C_r2a"], "R3": ["E_r3"], "R5": ["R_r5"], "R6": ["G_r6"], "R7": [], "R7b": ["J_r7b"]}
    sai = {k: list(d.index[d[k]]) for k, v in ky_vong.items() if list(d.index[d[k]]) != v}
    kt("dau_hieu: từng luật", not sai, sai)
    kt("Nhãn chính = R2a, R3, R6, R7b", list(d.index[d["nhan_chinh"]]) == ["C_r2a", "E_r3", "G_r6", "J_r7b"], list(d.index[d["nhan_chinh"]]))
    kt("Nhãn độ nhạy thêm R1, R5", set(d.index[d["nhan_do_nhay"]]) == {"B_r1", "C_r2a", "E_r3", "G_r6", "J_r7b", "R_r5"})
    kt("R2b: RPB do người thật", list(d.index[d["R2b_rpb_nguoi_that"]]) == ["C_r2a"])
    kt("Trạng thái: A, B_r1, R_r5 hoàn tất; G_r6 còn mở", list(d.index[d["trang_thai"] == "Hoàn tất"]) == ["A", "B_r1", "R_r5"]
       and d.loc["G_r6", "trang_thai"] == "Còn mở", d["trang_thai"].to_dict())
    cl = chong_lan(d)
    kt("Chồng lấn: đường chéo = số dòng hàng của luật", all(cl.loc[k, k] == d[k].sum() for k in LUAT))

    nhom = pd.Series(hm).reindex(d.index)
    b = theo_nhom(d, nhom, ["nhan_chinh"])
    kt("Theo nhóm: S2 3/3 nhãn chính, S1 1/4", (b.loc["S2", "nhan_chinh"], b.loc["S2", "so_dong_hang"], b.loc["S1", "nhan_chinh"],
                                                   b.loc["S1", "so_dong_hang"]) == (3, 3, 1, 4), b.to_dict())
    kt("Theo nhóm: hệ số S2 = 1 / (4/7)", np.isclose(b.loc["S2", "he_so_nhan_chinh"], 7 / 4))
    tt = tap_trung(d, pd.Series(ncc).reindex(d.index), "nhan_chinh", top=1)
    # nhãn chính: V2 có 2 (E, G), V1 1, V3 1 → top1 = V2: 2/4 dấu hiệu, 2/7 dòng hàng, 1 nhóm chứa một nửa
    kt("Tập trung: top1 = 50% dấu hiệu, 2/7 dòng hàng", np.isclose(tt["phan_dau_hieu_top1"], 0.5)
       and np.isclose(tt["phan_dong_hang_top1"], 2 / 7) and tt["so_nhom_chua_nua_dau_hieu"] == 1, tt)
    v = gia_tri_dong_hang(ev, d.index)
    kt("Giá trị lấy tại PO đầu tiên", v.to_dict() == gia, v.to_dict())
    gt = ty_trong_gia_tri(d, v).set_index("luat")
    kt("Tỷ trọng giá trị nhãn chính = 500/1000", np.isclose(gt.loc["nhan_chinh", "ty_trong_gia_tri"], 0.5))
    kt("Tỉ số giá trị / số dòng R1 = 0,3 / (1/7)", np.isclose(gt.loc["R1", "ti_so_gia_tri_tren_so_dong"], 0.3 * 7))

    vt = pd.Series(["a"] * 6 + ["b"] * 3 + ["c"], index=list("0123456789"))
    dm = chon_bien_the(vt, 0.9)
    kt("Chọn biến thể phủ 90%: a, b", list(dm.index) == ["a", "b"] and list(dm.values) == [6, 3], dm.to_dict())
    vt = bien_the_mo_hinh(ev, d.index)
    kt("Biến thể mô hình bỏ Change Price", vt["E_r3"] == sq1.MUI_TEN.join([PO, GR, IR]), vt["E_r3"])

    pm4py = sq1.nap_pm4py()
    if pm4py is None:
        print("Alignment: chưa có pm4py, bỏ qua (pip install pm4py).")
    else:
        from pm4py.algo.evaluation.precision import algorithm as precision
        from pm4py.objects.log.obj import EventLog
        net, im, fm = sq1.mo_hinh_chuan(pm4py)
        V = sq1.VET_MO_HINH
        dem = pd.Series({sq1.MUI_TEN.join(V[k][0]): n for k, n in [("a_hop_le", 5), ("b_hop_le", 3), ("b_thieu_rpb", 2)]})
        al = can_bien_the(list(dem.index), net, im, fm)
        fit, khop, lech = tong_hop_alignment(dem, al)
        f_thieu = al.set_index("vet").loc[dem.index[2], "fitness"]
        kt("Alignment: fitness có trọng số", np.isclose(fit, (8 + 2 * f_thieu) / 10) and f_thieu < 1, (fit, f_thieu))
        kt("Alignment: 80% khớp hoàn toàn", np.isclose(khop, 0.8), khop)
        kt("Alignment: b_thieu_rpb thiếu Remove Payment Block ở 2 dòng hàng",
           lech.to_dict("records") == [{"loai_buoc_lech": "thiếu so với mô hình", "hoat_dong": kc.RPB, "so_dong_hang": 2,
                                        "so_lan": 2, "ty_le_dong_hang": 0.2}], lech.to_dict("records"))
        # precision với Trace dùng chung phải bằng precision trên log mở rộng từng vết riêng
        log_day_du = EventLog([tao_trace(v, f"c{i}_{j}") for i, (v, n) in enumerate(dem.items()) for j in range(n)])
        p_ref = precision.apply(log_day_du, net, im, fm, variant=precision.Variants.ALIGN_ETCONFORMANCE,
                                parameters={"show_progress_bar": False})
        p_moi = precision_theo_tan_suat(dem, net, im, fm)
        kt("Precision theo tần suất = precision trên log đầy đủ", np.isclose(p_ref, p_moi), (p_ref, p_moi))
        kt("Precision trong khoảng (0, 1]", 0 < p_moi <= 1, p_moi)
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
    ap.add_argument("--ra", default="ket_qua_sq2")
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
        print(s)
        out.append(s)

    raw = doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    phan_tich(ev, pv, moc, ra, p)
    (ra / "tom_tat_sq2.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_sq2.txt.")


if __name__ == "__main__":
    main()
