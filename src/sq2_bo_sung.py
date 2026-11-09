"""Số liệu bổ sung cho SQ1 và SQ2 theo góp ý ngày 21/10 (mô tả, không sửa luật đã đóng băng).

  1. R6 tách hai nhóm mô tả: "cùng phút" (người tạo dòng đơn hàng nhận hàng trong vòng một phút sau, hoặc cùng lúc) và
     "khác thời điểm"; kèm vị trí của lần nhận hàng đó trong vết (sự kiện thứ mấy) để biết nhóm nào bị loại khỏi tiền tố SQ4.
  2. R7 dạng khoảng: biến thể phụ (hóa đơn nhà cung cấp trước ngày tạo dòng đơn hàng) đến "trước hoặc cùng ngày".
  3. Tỷ lệ dòng hàng có nhiều lần nhận hàng hoặc nhiều lần ghi nhận hóa đơn (SQ1).
  4. R1 trên các dòng hàng không có lần hủy nào (Cancel Invoice Receipt, Cancel Goods Receipt): phân tích khám phá sau
     đóng băng, chỉ mô tả, không vào nhãn; lấy mẫu 20 case để đọc tay.

Cách chạy (cùng thư mục với sq2_tuan_thu.py và các file đi kèm):
    python src/sq2_bo_sung.py --tu-kiem
    python src/sq2_bo_sung.py
Kết quả ghi vào ket_qua_sq2_bo_sung/; gửi lại tom_tat_sq2_bo_sung.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import g4_kiem_tra as g4
import kiem_chung_cuoi as kc
import sq2_tuan_thu as sq2
from r5_v2_phan_loai_clear import (CANCEL_GR, CANCEL_IR, GR, IR, PO, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear,
                                   sap_xep)
import cau_hinh

SEED = 2026
MOT_PHUT = pd.Timedelta(minutes=1)


def vn(x, dang=","):
    """Định dạng số kiểu Việt Nam: 1.234 và 3,50."""
    return format(x, dang).replace(",", "X").replace(".", ",").replace("X", ".")


def r6_hai_nhom(ev, r6):
    """Với mỗi dòng R6: khoảng cách nhỏ nhất giữa Create PO Item và Record GR của cùng một người dùng thật."""
    e = ev[ev["case"].isin(r6) & ev["act"].isin([PO, GR]) & kc.la_nguoi_that(ev["user"])]
    po = e[e["act"] == PO][["case", "user", "ts", "vi_tri"]]
    gr = e[e["act"] == GR][["case", "user", "ts", "vi_tri"]]
    m = po.merge(gr, on=["case", "user"], suffixes=("_po", "_gr"))
    m["kc"] = (m["ts_gr"] - m["ts_po"]).abs()
    m = m.sort_values(["case", "kc"]).drop_duplicates("case")
    m["nhom"] = np.where(m["kc"] <= MOT_PHUT, "cùng phút", "khác thời điểm")
    return m.set_index("case")


def phan_tich(ev, pv, moc, ra, p, rng):
    N = len(pv)
    ev = ev.copy()
    ev["vi_tri"] = ev.groupby("case").cumcount() + 1  # sự kiện thứ mấy trong vết (1 = đầu tiên)
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    ts_dp = g4.gio_dia_phuong(ev, lambda *a: None)
    kq, _ = kc.cac_luat(ev, pv, loai, ts_dp)
    p(f"Phạm vi {vn(N)} dòng hàng. Thời điểm trích xuất: {moc}")

    # 1. R6
    r6 = kq["R6"][0]
    m = r6_hai_nhom(ev, r6)
    p(f"\n=== 1. R6 tách hai nhóm mô tả ({vn(len(r6))} dòng hàng) ===")
    for nhom, g in m.groupby("nhom"):
        p(f"  {nhom}: {g4.pct(len(g), len(r6))} dòng R6; nhận hàng nằm trong 2 / 3 sự kiện đầu của vết: {vn(int((g['vi_tri_gr'] <= 2).sum()))} / {vn(int((g['vi_tri_gr'] <= 3).sum()))}; "
          f"trung vị khoảng cách {vn(g['kc'].median() / pd.Timedelta(days=1), ',.2f')} ngày")
    if "vendor" in ev.columns:
        vd = ev.drop_duplicates("case").set_index("case")["vendor"]
        for nhom, g in m.groupby("nhom"):
            top = vd.reindex(g.index).value_counts().head(3)
            p(f"  {nhom}, 3 nhà cung cấp nhiều nhất: " + "; ".join(f"{k} {vn(v)}" for k, v in top.items()))
    m[["user", "ts_po", "ts_gr", "vi_tri_gr", "kc", "nhom"]].to_csv(ra / "r6_hai_nhom.csv", encoding="utf-8-sig")

    # 2. R7 dạng khoảng
    r7b, _, cung_ngay = kq["R7b"]
    p(f"\n=== 2. R7 dạng khoảng ===")
    p(f"  Hóa đơn nhà cung cấp trước ngày tạo dòng đơn hàng (biến thể phụ): {g4.pct(len(r7b), N)}")
    p(f"  Cùng ngày: {g4.pct(len(cung_ngay), N)}")
    p(f"  Trước hoặc cùng ngày: {g4.pct(len(r7b | cung_ngay), N)}")

    # 3. Nhiều lần nhận hàng / ghi nhận hóa đơn
    dem = ev[ev["act"].isin([GR, IR])].groupby(["case", "act"]).size().unstack(fill_value=0).reindex(sorted(pv), fill_value=0)
    ngr, nir = dem.get(GR, 0), dem.get(IR, 0)
    p(f"\n=== 3. Nhiều lần nhận hàng hoặc ghi nhận hóa đơn ===")
    p(f"  Có từ 2 lần nhận hàng: {g4.pct(int((ngr >= 2).sum()), N)} (trên dòng có nhận hàng: {g4.pct(int((ngr >= 2).sum()), int((ngr >= 1).sum()))})")
    p(f"  Có từ 2 lần ghi nhận hóa đơn: {g4.pct(int((nir >= 2).sum()), N)} (trên dòng có hóa đơn: {g4.pct(int((nir >= 2).sum()), int((nir >= 1).sum()))})")
    p(f"  Có ít nhất một trong hai: {g4.pct(int(((ngr >= 2) | (nir >= 2)).sum()), N)}")
    p(f"  Đúng 1 lần nhận hàng và 1 lần ghi nhận hóa đơn: {g4.pct(int(((ngr == 1) & (nir == 1)).sum()), N)}")

    # 4. R1 trên dòng hàng không có hủy
    co_huy = set(ev.loc[ev["act"].isin([CANCEL_IR, CANCEL_GR]), "case"])
    khong_huy = set(pv) - co_huy
    r1 = kq["R1"][0]
    r1_kh = r1 & khong_huy
    p(f"\n=== 4. R1 trên dòng hàng không có lần hủy nào (khám phá sau đóng băng, không vào nhãn) ===")
    p(f"  Dòng hàng có hủy (Cancel Invoice Receipt hoặc Cancel Goods Receipt): {g4.pct(len(co_huy), N)}")
    p(f"  R1 trên toàn phạm vi: {g4.pct(len(r1), N)}; trong đó trên dòng không hủy: {vn(len(r1_kh))} "
      f"({vn(len(r1_kh) / max(len(r1), 1), '.1%')} số dòng R1; {vn(len(r1_kh) / max(len(khong_huy), 1), '.3%')} số dòng không hủy)")
    da_doc = kc.g5.doc_case_da_doc()
    for th in kc.THU_MUC_DA_DOC[len(kc.g5.THU_MUC_DA_DOC):] + ["ket_qua_kiem_chung_cuoi/mau_kiem_chung"]:
        for f in Path(th).glob("*.csv"):
            try:
                da_doc |= set(pd.read_csv(f, encoding="utf-8-sig", usecols=["case"])["case"].astype(str))
            except Exception:
                pass
    ds = sorted(r1_kh - da_doc)
    chon = list(rng.choice(ds, size=min(20, len(ds)), replace=False)) if ds else []
    vet = ev[ev["case"].isin(chon)][["case", "ts", "act", "user"]].copy()
    vet["loai_clear"] = loai.reindex(vet.index)
    vet["ket_qua"], vet["nhan_xet"] = "", ""  # đúng / báo nhầm do dữ liệu / không xác định
    vet.to_csv(ra / "mau_r1_khong_huy.csv", index=False, encoding="utf-8-sig")
    p(f"  Mẫu đọc tay: {vn(len(ds))} case chưa đọc, lấy {len(chon)} (mau_r1_khong_huy.csv)")


def tu_kiem():
    t = pd.Timestamp("2018-03-01 08:00")
    m, ngay = pd.Timedelta(minutes=1), pd.Timedelta(days=1)
    vet = {
        "A_r6_cung_phut": [(PO, 0 * m, "u1"), (GR, 0 * m, "u1"), (IR, 2 * ngay, "u3")],
        "B_r6_khac": [(PO, 0 * m, "u1"), (IR, ngay, "u3"), (GR, 3 * ngay, "u1")],
        "C_hai_gr": [(PO, 0 * m, "u1"), (GR, ngay, "u2"), (GR, 2 * ngay, "u2"), (IR, 3 * ngay, "u3")],
        "D_r1_khong_huy": [(PO, 0 * m, "u1"), (IR, ngay, "u3"), ("Clear Invoice", 10 * ngay, "u4"), (GR, 20 * ngay, "u2")],
        "E_binh_thuong": [(PO, 0 * m, "u1"), (GR, ngay, "u2"), (IR, 2 * ngay, "u3"), ("Clear Invoice", 30 * ngay, "u4")],
    }
    df = pd.DataFrame([{"case": c, "act": a, "ts": t + dt, "user": u} for c, ds in vet.items() for a, dt, u in ds])
    ev = sap_xep(df)
    ev["vi_tri"] = ev.groupby("case").cumcount() + 1
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    kq, _ = kc.cac_luat(ev, set(ev["case"]), loai, g4.gio_dia_phuong(ev, lambda *a: None))
    mm = r6_hai_nhom(ev, kq["R6"][0])
    loi = 0

    def kt(ten, dat, ct=""):
        nonlocal loi
        loi += not dat
        print(f"{ten:52s} {'ĐẠT' if dat else 'SAI ' + str(ct)}")
    kt("R6 bắt hai case", sorted(kq["R6"][0]) == ["A_r6_cung_phut", "B_r6_khac"], sorted(kq["R6"][0]))
    kt("R6 cùng phút / khác thời điểm", mm["nhom"].to_dict() == {"A_r6_cung_phut": "cùng phút", "B_r6_khac": "khác thời điểm"}, mm["nhom"].to_dict())
    kt("Cùng phút: nhận hàng nằm trong 2 sự kiện đầu", int(mm.loc["A_r6_cung_phut", "vi_tri_gr"]) <= 2, mm["vi_tri_gr"].to_dict())
    kt("R1 bắt case không hủy", "D_r1_khong_huy" in kq["R1"][0], sorted(kq["R1"][0]))
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
    ap.add_argument("--ra", default="ket_qua_sq2_bo_sung")
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

    raw = sq2.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    phan_tich(ev, pv, moc, ra, p, np.random.default_rng(SEED))
    (ra / "tom_tat_sq2_bo_sung.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_sq2_bo_sung.txt.")


if __name__ == "__main__":
    main()
