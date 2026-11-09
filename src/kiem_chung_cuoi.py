"""Kiểm chứng cuối R1, R2a, R3, R6, R7 (G5 Mục 4, Mục 6.3 bước 5b): 20 case bị đánh dấu + 20 case không bị đánh dấu mỗi luật.

Định nghĩa theo Bảng 6 của đề cương:
  R1  Clear Invoice "thanh toán" khi chưa có Record Goods Receipt còn hiệu lực (bỏ nhận hàng đã bị Cancel Goods Receipt).
  R2a Remove Payment Block xảy ra trước Record Goods Receipt đầu tiên (dòng hàng chưa từng nhận hàng cũng bị đánh dấu,
      báo riêng số này).
  R3  Change Price xảy ra sau Record Invoice Receipt đầu tiên.
  R6  Cùng một người dùng thật (không phải batch, không phải NONE) thực hiện Create Purchase Order Item và Record Goods Receipt.
  R7  Record Invoice Receipt xảy ra trước Create Purchase Order Item.
      Biến thể phụ R7b: ngày (giờ địa phương) của Vendor creates invoice trước ngày tạo đơn hàng; cùng ngày coi là bằng nhau.
Hai sự kiện cùng timestamp không tính là vi phạm thứ tự; số case bị ảnh hưởng bởi trùng timestamp được báo riêng.

Mẫu "không bị đánh dấu" lấy trong các dòng hàng có hoạt động liên quan (ví dụ R3: có Change Price nhưng không bị đánh dấu),
để việc đọc tay kiểm tra được luật có bỏ sót hay không.

Cách chạy (cùng thư mục với g5_kiem_tra.py và các file đi kèm):
    python src/kiem_chung_cuoi.py --tu-kiem
    python src/kiem_chung_cuoi.py
Kết quả ghi vào ket_qua_kiem_chung_cuoi/; gửi lại tom_tat_kiem_chung_cuoi.txt.
Khi đọc tay, mỗi case ghi cột ket_qua: đúng / báo nhầm do dữ liệu / không xác định.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import g3_kiem_tra_bo_sung as g3
import g4_kiem_tra as g4
import g5_kiem_tra as g5
import r5_v2_phan_loai_clear as r5
from r5_v2_phan_loai_clear import CANCEL_GR, CLEAR, GR, IR, PO, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear, sap_xep
import cau_hinh

SEED = 2026
N_MAU = 20
RPB, CHANGE_PRICE, VENDOR_INV = "Remove Payment Block", "Change Price", "Vendor creates invoice"
THU_MUC_DA_DOC = g5.THU_MUC_DA_DOC + ["ket_qua_g5/mau_kiem_chung"]


def dau(ev, act):
    return ev[ev["act"] == act].groupby("case")["ts"].min()


def la_nguoi_that(u):
    u = u.astype(str)
    return ~g3.la_batch(u) & (u.str.upper() != "NONE") & (u != "nan")


def cac_luat(ev, pv, loai, ts_dp):
    """Trả về dict: luật -> (tập bị đánh dấu, tập liên quan, tập trùng timestamp)."""
    tat_ca = pd.Index(sorted(pv))
    co = {a: set(ev.loc[ev["act"] == a, "case"]) for a in [RPB, CHANGE_PRICE, IR, GR, CLEAR, VENDOR_INV]}
    kq = {}

    # R1
    tt = loai.index[loai == "thanh_toan"]
    r1 = r5.r1_theo_clear(ev, tt)
    e = ev[ev["act"].isin([GR, CANCEL_GR])]
    gr_ts = e.loc[e["act"] == GR, ["case", "ts"]]
    c_tt = ev.loc[tt, ["case", "ts"]]
    trung_r1 = set(c_tt.merge(gr_ts, on=["case", "ts"])["case"])
    kq["R1"] = (r1, set(ev.loc[tt, "case"]), trung_r1)

    # R2a
    rpb, gr1 = dau(ev, RPB), dau(ev, GR)
    x = pd.DataFrame({"rpb": rpb, "gr": gr1.reindex(rpb.index)})
    r2a = set(x.index[x["gr"].isna() | (x["rpb"] < x["gr"])])
    r2a_khong_gr = set(x.index[x["gr"].isna()])
    trung_r2a = set(x.index[x["rpb"] == x["gr"]])
    kq["R2a"] = (r2a, co[RPB], trung_r2a)

    # R3
    cp = ev.loc[ev["act"] == CHANGE_PRICE, ["case", "ts"]]
    ir1 = dau(ev, IR)
    cp = cp.assign(ir1=cp["case"].map(ir1))
    r3 = set(cp.loc[cp["ir1"].notna() & (cp["ts"] > cp["ir1"]), "case"])
    trung_r3 = set(cp.loc[cp["ts"] == cp["ir1"], "case"]) - r3
    kq["R3"] = (r3, co[CHANGE_PRICE], trung_r3)

    # R6
    u = ev[ev["act"].isin([PO, GR]) & la_nguoi_that(ev["user"])][["case", "act", "user"]].drop_duplicates()
    nguoi_po = u[u["act"] == PO][["case", "user"]]
    nguoi_gr = u[u["act"] == GR][["case", "user"]]
    r6 = set(nguoi_po.merge(nguoi_gr, on=["case", "user"])["case"])
    lien_quan_r6 = set(nguoi_po["case"]) & set(nguoi_gr["case"])
    kq["R6"] = (r6, lien_quan_r6, set())

    # R7 và R7b
    po1 = dau(ev, PO)
    x = pd.DataFrame({"ir": ir1, "po": po1.reindex(ir1.index)})
    r7 = set(x.index[x["po"].notna() & (x["ir"] < x["po"])])
    trung_r7 = set(x.index[x["ir"] == x["po"]])
    kq["R7"] = (r7, co[IR], trung_r7)
    vi_dp = ts_dp[ev["act"] == VENDOR_INV].groupby(ev.loc[ev["act"] == VENDOR_INV, "case"]).min().dt.normalize()
    po_dp = ts_dp[ev["act"] == PO].groupby(ev.loc[ev["act"] == PO, "case"]).min().dt.normalize()
    y = pd.DataFrame({"vi": vi_dp, "po": po_dp.reindex(vi_dp.index)})
    r7b = set(y.index[y["po"].notna() & (y["vi"] < y["po"])])
    kq["R7b"] = (r7b, co[VENDOR_INV], set(y.index[y["vi"] == y["po"]]))
    return kq, r2a_khong_gr


def phan_tich(ev, pv, moc, ra, p, rng):
    N = len(pv)
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    ev = ev.copy()
    ev["loai_clear"] = loai.reindex(ev.index)
    ts_dp = g4.gio_dia_phuong(ev, p)
    p(f"Phạm vi {N:,} dòng hàng. Δ = {g4.DELTA} phút. Thời điểm trích xuất: {moc}")
    kq, r2a_khong_gr = cac_luat(ev, pv, loai, ts_dp)
    da_doc = g5.doc_case_da_doc()
    for th in THU_MUC_DA_DOC[len(g5.THU_MUC_DA_DOC):]:
        for f in Path(th).glob("*.csv"):
            try:
                da_doc |= set(pd.read_csv(f, encoding="utf-8-sig", usecols=["case"])["case"].astype(str))
            except Exception:
                pass
    p(f"Loại {len(da_doc):,} case đã đọc trước đây khỏi mẫu.\n")
    rows = []
    for luat, (bi, lien_quan, trung) in kq.items():
        khong = lien_quan - bi
        p(f"{luat}: bị đánh dấu {g4.pct(len(bi), N)}; dòng hàng có hoạt động liên quan {len(lien_quan):,}; "
          f"không bị đánh dấu chỉ vì trùng timestamp {len(trung):,}")
        if luat == "R2a":
            p(f"  trong đó chưa từng nhận hàng: {len(r2a_khong_gr):,}")
        for nhan, tap in (("bi_danh_dau", bi), ("khong_bi_danh_dau", khong)):
            ds = sorted(set(tap) - da_doc)
            chon = list(rng.choice(ds, size=min(N_MAU, len(ds)), replace=False)) if ds else []
            vet = ev[ev["case"].isin(chon)][["case", "ts", "act", "loai_clear", "user"]].copy()
            vet["luat"], vet["nhom"] = luat, nhan
            vet["ket_qua"] = ""  # đúng / báo nhầm do dữ liệu / không xác định
            vet["nhan_xet"] = ""
            vet.to_csv(ra / "mau_kiem_chung" / f"{luat}_{nhan}.csv", index=False, encoding="utf-8-sig")
            p(f"  mẫu {nhan}: {len(ds):,} case đủ điều kiện, lấy {len(chon)}")
            da_doc |= set(chon)
        rows.append({"luat": luat, "bi_danh_dau": len(bi), "ty_le": round(len(bi) / N, 6), "lien_quan": len(lien_quan),
                     "trung_timestamp": len(trung)})
        pd.Series(sorted(bi), name="case").to_csv(ra / f"{luat}_bi_danh_dau.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(ra / "tong_hop_luat.csv", index=False, encoding="utf-8-sig")
    p("\nTiêu chí vào nhãn chính (Mục 6.3, bước 5b): ít nhất 15/20 case bị đánh dấu là 'đúng'.")


def tu_kiem():
    t = pd.Timestamp("2018-03-01 08:00")
    m, ngay = pd.Timedelta(minutes=1), pd.Timedelta(days=1)
    vet = {
        # (hoạt động, thời điểm, người)
        "A_binh_thuong": [(PO, 0 * m, "u1"), (GR, ngay, "u2"), (IR, 2 * ngay, "u3"), (RPB, 3 * ngay, "u3"), (CLEAR, 30 * ngay, "u4")],
        "B_r1": [(PO, 0 * m, "u1"), (IR, ngay, "u3"), (CLEAR, 10 * ngay, "u4"), (GR, 20 * ngay, "u2")],
        "C_r2a": [(PO, 0 * m, "u1"), (IR, ngay, "u3"), (RPB, 2 * ngay, "u3"), (GR, 5 * ngay, "u2")],
        "D_r2a_trung": [(PO, 0 * m, "u1"), (IR, ngay, "u3"), (RPB, 5 * ngay, "u3"), (GR, 5 * ngay, "u2")],
        "E_r3": [(PO, 0 * m, "u1"), (GR, ngay, "u2"), (IR, 2 * ngay, "u3"), (CHANGE_PRICE, 3 * ngay, "u1")],
        "F_r3_truoc": [(PO, 0 * m, "u1"), (CHANGE_PRICE, ngay, "u1"), (GR, ngay, "u2"), (IR, 2 * ngay, "u3")],
        "G_r6": [(PO, 0 * m, "u1"), (GR, ngay, "u1"), (IR, 2 * ngay, "u3")],
        "H_r6_batch": [(PO, 0 * m, "batch_01"), (GR, ngay, "batch_01"), (IR, 2 * ngay, "u3")],
        "I_r7": [(IR, 0 * m, "u3"), (PO, ngay, "u1"), (GR, 2 * ngay, "u2")],
        # Vendor creates invoice 22:59 UTC ngày 28/2 = 23:59 ngày 28/2 giờ địa phương, đơn tạo 1/3 → R7b
        "J_r7b": [(VENDOR_INV, -9 * pd.Timedelta(hours=1) - m, "NONE"), (PO, 0 * m, "u1"), (GR, ngay, "u2"), (IR, 2 * ngay, "u3")],
        # cùng ngày địa phương → không phải R7b
        "K_r7b_cung_ngay": [(PO, 0 * m, "u1"), (VENDOR_INV, 14 * pd.Timedelta(hours=1) + 59 * m, "NONE"), (GR, ngay, "u2")],
    }
    df = pd.DataFrame([{"case": c, "act": a, "ts": t + dt, "user": u} for c, ds in vet.items() for a, dt, u in ds])
    # thêm một Vendor creates invoice lúc 22:59 UTC để gio_dia_phuong nhận ra log ở UTC
    ev = sap_xep(df)
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    ts_dp = g4.gio_dia_phuong(ev, lambda *a: None)
    kq, _ = cac_luat(ev, set(ev["case"]), loai, ts_dp)
    ky_vong = {"R1": ["B_r1"], "R2a": ["C_r2a"], "R3": ["E_r3"], "R6": ["G_r6"], "R7": ["I_r7"], "R7b": ["J_r7b"]}
    loi = 0
    for k, v in ky_vong.items():
        dat = sorted(kq[k][0]) == v
        loi += not dat
        print(f"{k:5s} {sorted(kq[k][0])}  {'ĐẠT' if dat else 'SAI, kỳ vọng ' + str(v)}")
    dat = kq["R2a"][2] == {"D_r2a_trung"}
    loi += not dat
    print(f"{'R2a trùng timestamp':5s} {sorted(kq['R2a'][2])}  {'ĐẠT' if dat else 'SAI, kỳ vọng [D_r2a_trung]'}")
    print("\nTất cả đều đạt." if loi == 0 else f"\n{loi} kiểm tra SAI.")
    return loi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(cau_hinh.LOG), help="đường dẫn log; mặc định trong cau_hinh.py")
    ap.add_argument("--cong-ty", default="companyID_0000")
    ap.add_argument("--ra", default="ket_qua_kiem_chung_cuoi")
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
    (ra / "tom_tat_kiem_chung_cuoi.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_kiem_chung_cuoi.txt.")


if __name__ == "__main__":
    main()
