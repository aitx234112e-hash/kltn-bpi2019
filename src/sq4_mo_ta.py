"""SQ4, phần mô tả trước khi huấn luyện (thiết kế Mục 3.7.4, ghi ngày 29/10/2026). Script này không huấn luyện mô hình nào.

  1. Thời điểm dấu hiệu: tỷ lệ dòng hàng R2a, R3, R6 có dấu hiệu đầu tiên trong H = 90 và 120 ngày kể từ khi tạo dòng đơn
     hàng; tính trên mọi dòng bị đánh dấu và trên dòng tạo trước thời điểm trích xuất ít nhất 120 ngày.
  2. Chia theo thời gian, T = 01/07/2018: huấn luyện trên dòng tạo trong [01/01/2018; T − H], kiểm tra trên dòng tạo trong
     [T; thời điểm trích xuất − H]. Lấy mọi dòng hàng trong phạm vi, kể cả dòng về sau còn mở hoặc bị xóa.
  3. Tiền tố 2–5 sự kiện, kết thúc trước dấu hiệu đầu tiên của nhãn và trước ngày thứ H. Số dòng hàng và số dòng nhãn 1
     theo tập, độ dài tiền tố và luật; ba nhãn (chính, R3 riêng, không có R6); hai tập độ nhạy (loại vendorID_0741 và
     vendorID_0572; loại dòng có Cancel Invoice Receipt hoặc Cancel Goods Receipt).
  4. Tỷ lệ nhãn 1 theo tháng tạo dòng đơn hàng.
  5. Kiểm tra rò rỉ: số tiền tố có Vendor creates invoice nhưng chưa có Record Invoice Receipt.

Cách chạy (cùng thư mục với sq3_thoi_gian.py):
    python src/sq4_mo_ta.py --tu-kiem
    python src/sq4_mo_ta.py
Kết quả ghi vào ket_qua_sq4_mo_ta/; gửi lại tom_tat_sq4_mo_ta.txt.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import g4_kiem_tra as g4
import g5_kiem_tra as g5
import kiem_chung_cuoi as kc
import sq2_tuan_thu as sq2
import sq3_thoi_gian as sq3
from r5_v2_phan_loai_clear import (CANCEL_GR, CANCEL_IR, GR, IR, PO, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear,
                                   sap_xep)
import cau_hinh

T_CAT = pd.Timestamp("2018-07-01")
BAT_DAU = pd.Timestamp("2018-01-01")
CAC_H = [90, 120]
CAC_K = [2, 3, 4, 5]
LUAT = ["R2a", "R3", "R6"]
NHAN = {"chinh": ["R2a", "R3", "R6"], "R3": ["R3"], "khong_R6": ["R2a", "R3"]}
TEN_NHAN = {"chinh": "Nhãn chính (R2a, R3, R6)", "R3": "Nhãn R3 riêng", "khong_R6": "Nhãn không có R6 (R2a, R3)"}
HAI_NCC_R6 = ["vendorID_0741", "vendorID_0572"]
KY_VONG = {"R2a": 508, "R3": 2008, "R6": 650}  # trên toàn phạm vi 210.887 dòng hàng
NGUONG = 100  # độ dài tiền tố có ít hơn 100 dòng nhãn 1 trong tập kiểm tra thì chỉ báo cáo, không kết luận
HL, KT = "huấn luyện", "kiểm tra"
VENDOR_INV = kc.VENDOR_INV
MOT_NGAY = sq3.MOT_NGAY
vn, pct = sq3.vn, sq3.pct


# ---------------------------------------------------------------- bảng dòng hàng, nhãn, tập, tiền tố
def bang_dong_hang(ev, pv, p, ky_vong=KY_VONG):
    """Mỗi dòng hàng trong phạm vi: thời điểm tạo, trạng thái tại thời điểm trích xuất, thời điểm dấu hiệu đầu tiên của
    từng luật (như SQ3), nhà cung cấp, có hủy hay không."""
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    ts_dp = g4.gio_dia_phuong(ev, p)
    kq, _ = kc.cac_luat(ev, pv, loai, ts_dp)
    tg = sq3.thoi_diem_dau_hieu(ev, kq)
    so_lieu = g5.so_lieu_case(ev, pv, loai)
    d = pd.DataFrame(index=so_lieu.index)
    d["tao"] = so_lieu["dau_po"]
    d["trang_thai"] = g5.trang_thai_cac_dinh_nghia(so_lieu)["moi"]
    for l in LUAT:
        d["sig_" + l] = tg[l].reindex(d.index)
    dau = ev.drop_duplicates("case").set_index("case").reindex(d.index)
    d["vendor"] = dau["vendor"].astype(str) if "vendor" in dau else "?"
    d["co_huy"] = d.index.isin(set(ev.loc[ev["act"].isin([CANCEL_IR, CANCEL_GR]), "case"]))
    dem = {l: int(d["sig_" + l].notna().sum()) for l in LUAT}
    lech = {l: (dem[l], ky_vong[l]) for l in ky_vong if dem[l] != ky_vong[l]}
    p("Số dòng hàng theo luật khớp kiểm chứng (R2a 508, R3 2.008, R6 650)." if not lech
      else f"!!! Lệch số dòng hàng (tính được, kỳ vọng): {lech}")
    lech_tg = [l for l in LUAT if set(tg[l].index) != set(kq[l][0])]
    p("Mọi dòng hàng bị đánh dấu đều có thời điểm dấu hiệu." if not lech_tg else f"!!! Thiếu thời điểm dấu hiệu: {lech_tg}")
    return d


def nhan(d, luat, h):
    """Thời điểm dấu hiệu đầu tiên trong các luật của nhãn; nhãn = 1 nếu dấu hiệu đó xảy ra trong h ngày kể từ khi tạo."""
    sig = d[["sig_" + l for l in luat]].min(axis=1)
    y = sig.notna() & (sig - d["tao"] <= pd.Timedelta(days=h))
    return sig, y


def tap(tao, moc, h, t_cat=T_CAT):
    """Huấn luyện: tạo trong [01/01/2018; T − h]; kiểm tra: tạo trong [T; moc − h]; còn lại: chuỗi rỗng."""
    H = pd.Timedelta(days=h)
    return pd.Series(np.select([(tao >= BAT_DAU) & (tao <= t_cat - H), (tao >= t_cat) & (tao <= moc - H)], [HL, KT], ""),
                     index=tao.index)


def cuoi_tien_to(ev, cac_k=CAC_K):
    """Với mỗi dòng hàng và mỗi k: thời điểm kết thúc tiền tố k sự kiện (muộn nhất trong k sự kiện đầu; NaT nếu vết ngắn
    hơn k), và tiền tố có Vendor creates invoice mà chưa có Record Invoice Receipt hay không."""
    g = ev.groupby("case")
    vt = g.cumcount()
    cm = g["ts"].cummax()
    vi = (ev["act"] == VENDOR_INV).groupby(ev["case"]).cummax()
    ir = (ev["act"] == IR).groupby(ev["case"]).cummax()
    idx = pd.Index(ev["case"].unique(), name="case")
    tk, chua_ir = pd.DataFrame(index=idx), pd.DataFrame(index=idx)
    for k in cac_k:
        m = vt == k - 1
        c = ev.loc[m, "case"].values
        tk[k] = pd.Series(cm[m].values, index=c).reindex(idx)
        chua_ir[k] = pd.Series((vi[m] & ~ir[m]).values, index=c).reindex(idx).fillna(False).astype(bool)
    return tk, chua_ir


def hop_le(tk, sig, tao, h):
    """Tiền tố dùng được: đủ k sự kiện và kết thúc trước min(dấu hiệu đầu tiên của nhãn, thời điểm tạo + h)."""
    gioi_han = tao + pd.Timedelta(days=h)
    gioi_han = gioi_han.where(sig.isna() | (sig > gioi_han), sig)
    return tk.notna() & (tk < gioi_han)


def bang_tien_to(d, tk, moc, h, ten_nhan, ten_tap_dl, chon):
    """Số dòng hàng và số dòng nhãn 1 theo tập và độ dài tiền tố ("0" = mọi dòng hàng của tập, chưa cắt tiền tố)."""
    luat = NHAN[ten_nhan]
    sig, y = nhan(d, luat, h)
    tp = tap(d["tao"], moc, h)
    trong_h = {l: (d["sig_" + l] - d["tao"]) <= pd.Timedelta(days=h) for l in LUAT}
    rows = []
    for s in [HL, KT]:
        m0 = (tp == s) & chon
        for k in [0] + CAC_K:
            m = m0 if k == 0 else m0 & hop_le(tk[k], sig, d["tao"], h)
            r = {"H": h, "nhan": ten_nhan, "tap_du_lieu": ten_tap_dl, "tap": s, "tien_to": k,
                 "n": int(m.sum()), "n1": int((m & y).sum()),
                 "con_mo": int((m & (d["trang_thai"] == "Còn mở")).sum()),
                 "bi_xoa": int((m & (d["trang_thai"] == "Bị xóa")).sum())}
            for l in LUAT:
                r["n1_" + l] = int((m & y & trong_h[l]).sum()) if l in luat else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def in_bang(b, p):
    luat = [l for l in LUAT if b["n1_" + l].notna().any()]
    nhieu_luat = len(luat) > 1
    p("    Tập          Tiền tố   Dòng hàng   Nhãn 1 (%)" + ("".join(f"   {l:>5s}" for l in luat) if nhieu_luat else ""))
    for _, r in b.iterrows():
        tt = "dòng hàng" if r["tien_to"] == 0 else f"{int(r['tien_to'])} sự kiện"
        it = r["tap"] == KT and r["tien_to"] > 0 and r["n1"] < NGUONG
        o = f"    {r['tap']:12s} {tt:10s} {vn(int(r['n'])):>9s}   {pct(r['n1'], r['n']):>15s}"
        if nhieu_luat:
            o += "".join(f"   {vn(int(r['n1_' + l])):>5s}" for l in luat)
        p(o + ("  (*)" if it else ""))


# ---------------------------------------------------------------- các phần mô tả
def ty_le_trong_h(d, moc, p):
    p("\n1. Dấu hiệu đầu tiên xảy ra trong H ngày kể từ khi tạo dòng đơn hàng")
    du = d["tao"] <= moc - pd.Timedelta(days=max(CAC_H))
    rows = []
    for ten, luat in [(l, [l]) for l in LUAT] + [("Nhãn chính", NHAN["chinh"])]:
        sig, _ = nhan(d, luat, 0)
        s = ((sig - d["tao"]) / MOT_NGAY)[sig.notna()]
        r = {"luat": ten, "n": len(s), "trung_vi_ngay": s.median(), "am": int((s < 0).sum())}
        for h in CAC_H:
            r[f"trong_{h}"] = int((s <= h).sum())
        s2 = s[du.reindex(s.index)]
        r["n_du_120"] = len(s2)
        for h in CAC_H:
            r[f"trong_{h}_du_120"] = int((s2 <= h).sum())
        rows.append(r)
        p(f"  {ten:11s} {vn(len(s)):>6s} dòng, trung vị ngày {sq3.so(s.median())}; trong 90 ngày {pct(r['trong_90'], len(s), 1)}, "
          f"trong 120 ngày {pct(r['trong_120'], len(s), 1)}. Chỉ dòng tạo trước thời điểm trích xuất ≥ 120 ngày "
          f"({vn(len(s2))} dòng): {pct(r['trong_90_du_120'], len(s2), 1)} và {pct(r['trong_120_du_120'], len(s2), 1)}"
          + (f"; {vn(r['am'])} dòng có dấu hiệu trước thời điểm tạo" if r["am"] else "") + ".")
    return pd.DataFrame(rows)


def co_tap(d, moc, p):
    p("\n2. Chia theo thời gian (T = 01/07/2018), mọi dòng hàng trong phạm vi")
    tao = d["tao"]
    p(f"  Dòng hàng không có Create Purchase Order Item: {vn(int(tao.isna().sum()))}; tạo trước 01/01/2018: "
      f"{vn(int((tao < BAT_DAU).sum()))}.")
    for h in CAC_H:
        tp = tap(tao, moc, h)
        H = pd.Timedelta(days=h)
        p(f"  H = {h}: huấn luyện tạo từ 01/01/2018 đến {(T_CAT - H):%d/%m/%Y %H:%M}, kiểm tra tạo từ {T_CAT:%d/%m/%Y} "
          f"đến {(moc - H):%d/%m/%Y %H:%M}.")
        for s in [HL, KT]:
            m = tp == s
            tt = d.loc[m, "trang_thai"].value_counts()
            p(f"    {s:12s} {vn(int(m.sum())):>7s} dòng hàng (hoàn tất {vn(int(tt.get('Hoàn tất', 0)))}, còn mở "
              f"{vn(int(tt.get('Còn mở', 0)))}, bị xóa {vn(int(tt.get('Bị xóa', 0)))})")
        p(f"    Không thuộc tập nào: {vn(int((tp == '').sum()))} (khoảng giữa T − H và T, sau thời điểm trích xuất − H, "
          f"hoặc trước 01/01/2018).")


def theo_thang(d, tk, moc, p):
    p("\n4. Tỷ lệ nhãn 1 theo tháng tạo dòng đơn hàng (nhãn chính, H = 90)")
    sig, y = nhan(d, NHAN["chinh"], 90)
    tp = tap(d["tao"], moc, 90)
    m2 = hop_le(tk[2], sig, d["tao"], 90)
    thang = d["tao"].dt.to_period("M").astype(str)
    rows = []
    for s in [HL, KT]:
        for th in sorted(thang[tp == s].unique()):
            m = (tp == s) & (thang == th)
            rows.append({"tap": s, "thang": th, "n": int(m.sum()), "n1": int((m & y).sum()),
                         "n_tien_to_2": int((m & m2).sum()), "n1_tien_to_2": int((m & m2 & y).sum())})
            r = rows[-1]
            p(f"  {s:12s} {th}: {vn(r['n']):>6s} dòng hàng, nhãn 1 {pct(r['n1'], r['n']):>13s}; "
              f"tiền tố 2: {vn(r['n_tien_to_2']):>6s}, nhãn 1 {pct(r['n1_tien_to_2'], r['n_tien_to_2'])}")
    return pd.DataFrame(rows)


def ro_ri_vendor_inv(d, tk, chua_ir, moc, p):
    p("\n5. Tiền tố có Vendor creates invoice nhưng chưa có Record Invoice Receipt (nhãn chính, H = 90)")
    sig, y = nhan(d, NHAN["chinh"], 90)
    tp = tap(d["tao"], moc, 90)
    for k in CAC_K:
        m = (tp != "") & hop_le(tk[k], sig, d["tao"], 90)
        c = m & chua_ir[k].reindex(d.index).fillna(False).astype(bool)
        p(f"  Tiền tố {k}: {pct(int(c.sum()), int(m.sum()), 1)} tiền tố; trong các tiền tố nhãn 1: "
          f"{pct(int((c & y).sum()), int((m & y).sum()), 1)}")


def phan_tich(ev, pv, moc, ra, p, luu, ky_vong=KY_VONG):
    t0 = time.time()
    p(f"Phạm vi {vn(len(pv))} dòng hàng, {vn(len(ev))} sự kiện. Thời điểm trích xuất: {moc}")
    d = bang_dong_hang(ev, pv, p, ky_vong)
    tk, chua_ir = cuoi_tien_to(ev)
    tk = tk.reindex(d.index)
    ty_le_trong_h(d, moc, p).to_csv(ra / "dau_hieu_trong_H.csv", index=False, encoding="utf-8-sig")
    co_tap(d, moc, p)
    luu()
    p("\n3. Số dòng hàng và số dòng nhãn 1 theo độ dài tiền tố. Tiền tố kết thúc trước dấu hiệu đầu tiên của nhãn và "
      "trước ngày thứ H.\n   Cột R2a, R3, R6: số dòng nhãn 1 có dấu hiệu của luật đó trong H ngày (một dòng có thể "
      f"có nhiều luật). (*): tập kiểm tra có dưới {NGUONG} dòng nhãn 1, chỉ báo cáo, không kết luận.")
    tat_ca = pd.Series(True, index=d.index)
    cac_tap_dl = {"toàn bộ": tat_ca, "loại hai nhà cung cấp R6": ~d["vendor"].isin(HAI_NCC_R6),
                  "loại dòng có hủy": ~d["co_huy"]}
    bang = []
    for h in CAC_H:
        for ten_nhan in NHAN:
            for ten_tap_dl, chon in cac_tap_dl.items():
                b = bang_tien_to(d, tk, moc, h, ten_nhan, ten_tap_dl, chon)
                bang.append(b)
                if ten_tap_dl == "toàn bộ" and (h == 90 or ten_nhan == "chinh") or (h == 90 and ten_nhan == "chinh"):
                    p(f"\n  {TEN_NHAN[ten_nhan]}, H = {h}, {ten_tap_dl}:")
                    in_bang(b, p)
    pd.concat(bang, ignore_index=True).to_csv(ra / "bang_tien_to.csv", index=False, encoding="utf-8-sig")
    luu()
    theo_thang(d, tk, moc, p).to_csv(ra / "nhan_theo_thang.csv", index=False, encoding="utf-8-sig")
    ro_ri_vendor_inv(d, tk, chua_ir, moc, p)
    p(f"\nXong sau {vn(round(time.time() - t0))} giây.")


# ---------------------------------------------------------------- tự kiểm
def tu_kiem():
    loi = 0

    def kt(ten, dat, ct=""):
        nonlocal loi
        loi += not dat
        print(f"{ten:72s} {'ĐẠT' if dat else 'SAI ' + str(ct)}")

    moc = pd.Timestamp("2019-01-18 13:34")
    # chia tập: biên của cửa sổ
    tao = pd.Series(pd.to_datetime(["2017-12-31 23:00", "2018-01-01 00:00", "2018-04-02 00:00", "2018-04-02 00:01",
                                    "2018-06-30 12:00", "2018-07-01 00:00", "2018-10-20 13:34", "2018-10-20 13:35"]))
    kt("Chia tập H = 90: biên 01/01, 02/04, 01/07 và 20/10/2018 13:34",
       list(tap(tao, moc, 90)) == ["", HL, HL, "", "", KT, KT, ""], list(tap(tao, moc, 90)))
    kt("Chia tập H = 120: huấn luyện đến 03/03/2018, kiểm tra đến 20/09/2018 13:34",
       list(tap(pd.Series(pd.to_datetime(["2018-03-03 00:00", "2018-03-03 00:01", "2018-09-20 13:34", "2018-09-21 00:00"])),
                moc, 120)) == [HL, "", KT, ""])

    # vết giả lập qua đúng các hàm luật của SQ2–SQ3
    t0 = pd.Timestamp("2018-03-01 08:00")
    m, ng = pd.Timedelta(minutes=1), MOT_NGAY
    CP, RPB, CL = kc.CHANGE_PRICE, kc.RPB, "Clear Invoice"
    vet = {
        # R3 ngày 10: tiền tố 2, 3 hợp lệ; tiền tố 4 kết thúc đúng lúc sửa giá nên bị loại
        "A": [(PO, 0 * m, "u1"), (GR, ng, "u2"), (IR, 2 * ng, "u3"), (CP, 10 * ng, "u1"), (RPB, 10 * ng + m, "batch_01"),
              (CL, 20 * ng, "batch_02")],
        # R3 ngày 100: nhãn 0 với H = 90, nhãn 1 với H = 120; tiền tố 4 (ngày 100) quá H = 90
        "B": [(PO, 0 * m, "u1"), (GR, ng, "u2"), (IR, 2 * ng, "u3"), (CP, 100 * ng, "u1"), (CL, 110 * ng, "batch_02")],
        # R6 cùng phút: tiền tố 2 kết thúc đúng lúc dấu hiệu nên bị loại; với nhãn R3 riêng thì vẫn dùng được
        "C": [(PO, 0 * m, "u4"), (GR, 0 * m, "u4"), (IR, 3 * ng, "u3"), (CL, 30 * ng, "batch_02")],
        # Vendor creates invoice trước khi tạo đơn: tiền tố 2 có hóa đơn nhà cung cấp, chưa có ghi nhận hóa đơn
        "D": [(VENDOR_INV, -ng, "NONE"), (PO, 0 * m, "u1"), (GR, 2 * ng, "u2"), (IR, 3 * ng, "u3"), (CL, 30 * ng, "batch_02"),
              (CANCEL_IR, 31 * ng, "u3")],
        # R2a (gỡ khóa trước nhận hàng, ngày 5) và R3 (ngày 40): nhãn chính lấy dấu hiệu sớm hơn
        "E": [(PO, 0 * m, "u1"), (IR, ng, "u3"), (RPB, 5 * ng, "u5"), (GR, 6 * ng, "u2"), (CP, 40 * ng, "u1"),
              (CL, 50 * ng, "batch_02")],
    }
    ev = pd.DataFrame([{"case": c, "act": a, "ts": t0 + dt, "user": u, "vendor": "vendorID_0741" if c == "E" else "v1"}
                       for c, ds in vet.items() for a, dt, u in ds])
    ev = sap_xep(ev)
    d = bang_dong_hang(ev, set(ev["case"]), lambda *x: None, ky_vong={})
    kt("Thời điểm dấu hiệu: R3 của A, B, E; R2a của E; R6 của C",
       d["sig_R3"].dropna().to_dict() == {"A": t0 + 10 * ng, "B": t0 + 100 * ng, "E": t0 + 40 * ng}
       and d["sig_R2a"].dropna().to_dict() == {"E": t0 + 5 * ng} and d["sig_R6"].dropna().to_dict() == {"C": t0},
       d[["sig_R2a", "sig_R3", "sig_R6"]])
    _, y90 = nhan(d, NHAN["chinh"], 90)
    _, y120 = nhan(d, NHAN["chinh"], 120)
    kt("Nhãn H = 90: A, C, E; H = 120 thêm B",
       list(d.index[y90]) == ["A", "C", "E"] and list(d.index[y120]) == ["A", "B", "C", "E"])
    sig_e, _ = nhan(d, NHAN["chinh"], 90)
    kt("Nhãn chính lấy dấu hiệu sớm nhất của E (R2a ngày 5)", sig_e["E"] == t0 + 5 * ng, sig_e["E"])
    tk, chua_ir = cuoi_tien_to(ev)
    tk = tk.reindex(d.index)
    kt("Cuối tiền tố: tiền tố 4 của A ở ngày 10; vết C chỉ có 4 sự kiện, D tiền tố 2 lúc tạo đơn",
       tk.loc["A", 4] == t0 + 10 * ng and pd.isna(tk.loc["C", 5]) and tk.loc["D", 2] == t0, tk)
    sig, _ = nhan(d, NHAN["chinh"], 90)
    hl = {k: list(d.index[hop_le(tk[k], sig, d["tao"], 90)]) for k in CAC_K}
    kt("Tiền tố hợp lệ, nhãn chính, H = 90 (E: tiền tố 3 kết thúc đúng lúc gỡ khóa)",
       hl == {2: ["A", "B", "D", "E"], 3: ["A", "B", "D"], 4: ["D"], 5: ["D"]}, hl)
    sig3, _ = nhan(d, NHAN["R3"], 90)
    hl3 = list(d.index[hop_le(tk[2], sig3, d["tao"], 90)])
    kt("Nhãn R3 riêng: dòng R6 cùng phút vẫn có tiền tố 2", hl3 == ["A", "B", "C", "D", "E"], hl3)
    kt("Tiền tố có Vendor creates invoice, chưa có ghi nhận hóa đơn: D ở tiền tố 2, 3; hết ở tiền tố 4",
       list(chua_ir.index[chua_ir[2]]) == ["D"] and chua_ir.loc["D", 3] and not chua_ir.loc["D", 4], chua_ir)
    moc2 = pd.Timestamp("2018-12-31")
    b = bang_tien_to(d, tk, moc2, 90, "chinh", "toàn bộ", pd.Series(True, index=d.index))
    r = b[(b["tap"] == HL) & (b["tien_to"] == 2)].iloc[0]
    r0 = b[(b["tap"] == HL) & (b["tien_to"] == 0)].iloc[0]
    kt("Bảng tiền tố: 5 dòng, 3 nhãn 1; tiền tố 2: 4 dòng, 2 nhãn 1 (A: R3; E: R2a và R3)",
       (r0["n"], r0["n1"], r["n"], r["n1"], r["n1_R2a"], r["n1_R3"], r["n1_R6"]) == (5, 3, 4, 2, 1, 2, 0), b)
    kt("Loại dòng có hủy bỏ D; loại hai nhà cung cấp R6 bỏ E",
       list(d.index[~d["co_huy"]]) == ["A", "B", "C", "E"] and list(d.index[~d["vendor"].isin(HAI_NCC_R6)]) == list("ABCD"))
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
    ap.add_argument("--ra", default="ket_qua_sq4_mo_ta")
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
        (ra / "tom_tat_sq4_mo_ta.txt").write_text("\n".join(out), encoding="utf-8")

    t = time.time()
    raw = sq2.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    p(f"Đọc log xong sau {vn(round(time.time() - t))} giây.")
    phan_tich(ev, pv, moc, ra, p, luu)
    luu()
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_sq4_mo_ta.txt.")


if __name__ == "__main__":
    main()
