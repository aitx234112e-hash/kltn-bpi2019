"""Số dòng hàng và số sự kiện theo trạng thái cuối (định nghĩa G5, 18/10), để cập nhật Bảng 3 của đề cương.

Cách chạy (cùng thư mục với sq2_tuan_thu.py và các file đi kèm):
    python src/dem_su_kien_trang_thai.py
Kết quả in ra màn hình và ghi vào so_su_kien_theo_trang_thai.txt.
"""
import argparse
import sys

import pandas as pd

import g4_kiem_tra as g4
import g5_kiem_tra as g5
import sq2_tuan_thu as sq2
from r5_v2_phan_loai_clear import buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear, sap_xep
import cau_hinh


def so(x):
    return f"{x:,}".replace(",", ".")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(cau_hinh.LOG), help="đường dẫn log; mặc định trong cau_hinh.py")
    ap.add_argument("--cong-ty", default="companyID_0000")
    a = ap.parse_args()
    raw = sq2.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    tt = g5.trang_thai_cac_dinh_nghia(g5.so_lieu_case(ev, pv, loai))["moi"]
    so_sk = ev.groupby("case").size().reindex(tt.index).fillna(0).astype(int)
    N, E = len(tt), int(so_sk.sum())
    out = [f"Phạm vi: {so(N)} dòng hàng, {so(E)} sự kiện (Bảng 3, bước 6: 210.887 dòng hàng, 1.175.650 sự kiện)"]
    for k in ["Hoàn tất", "Còn mở", "Bị xóa"]:
        m = tt == k
        pt = f"{m.mean() * 100:.2f}".replace(".", ",")
        out.append(f"  {k}: {so(int(m.sum()))} dòng hàng ({pt}%), {so(int(so_sk[m].sum()))} sự kiện")
    s = "\n".join(out)
    print(s)
    open("so_su_kien_theo_trang_thai.txt", "w", encoding="utf-8").write(s)
    print("\nĐã ghi so_su_kien_theo_trang_thai.txt. Gửi lại file này.")


if __name__ == "__main__":
    main()
