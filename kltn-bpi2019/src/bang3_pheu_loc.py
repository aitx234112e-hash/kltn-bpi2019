"""Điền Bảng 3 (phễu lọc) và số case theo trạng thái cho Mục 5.2–5.3.

Đặt file này cùng thư mục với kiem_tra_trang_thai_case.py rồi chạy:
    python bang3_pheu_loc.py duong_dan/BPI_Challenge_2019.xes
    python bang3_pheu_loc.py duong_dan/BPI_Challenge_2019.csv --cong-ty companyID_0000

Không ghi --cong-ty thì script lấy công ty có nhiều dòng hàng nhất (in ra để bạn kiểm tra).
Kết quả: in bảng ra màn hình và ghi bang3_pheu_loc.csv.
"""
import argparse

import pandas as pd

from kiem_tra_trang_thai_case import buoc2_moc_trich_xuat, buoc3_dem_va_gan_nhan, doc_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--cong-ty", help="mã công ty con chính, ví dụ companyID_0000")
    ap.add_argument("--moc", help="thời điểm trích xuất đặt tay, dạng YYYY-MM-DD")
    a = ap.parse_args()

    df = doc_log(a.log)
    attr = df.drop_duplicates("case").set_index("case")
    so_sk = df.groupby("case").size()

    print("Số dòng hàng theo công ty (5 công ty lớn nhất):")
    print(attr["company"].value_counts().head(5).to_string())
    cong_ty = a.cong_ty or attr["company"].value_counts().index[0]
    print(f"=> Công ty con chính dùng để lọc: {cong_ty}\n")

    moc, ly_do, _ = buoc2_moc_trich_xuat(df, a.moc)
    print(f"Thời điểm trích xuất: {moc} ({ly_do})\n")
    dau_po = df[df["act"].str.lower() == "create purchase order item"].groupby("case")["ts"].min()
    co_truoc_2017 = df[df["ts"] < "2017-01-01"]["case"].unique()
    co_sau_moc = df[df["ts"] > moc]["case"].unique()

    buoc = [
        ("0. Toàn bộ log", pd.Series(True, index=attr.index)),
        ("1. Item Category = 3-way match, hóa đơn trước nhận hàng",
         attr["item_category"].astype(str).str.contains("invoice before GR", case=False)),
        ("2. Document Type = Standard PO", attr["doc_type"].astype(str).eq("Standard PO")),
        ("3. Item Type = Standard", attr["item_type"].astype(str).eq("Standard")),
        (f"4. Company = {cong_ty}", attr["company"].astype(str).eq(cong_ty)),
        ("5. Create Purchase Order Item trong năm 2018", dau_po.reindex(attr.index).dt.year.eq(2018)),
        ("6. Loại case có timestamp lỗi (trước 2017 hoặc sau thời điểm trích xuất)",
         pd.Series(~attr.index.isin(co_truoc_2017) & ~attr.index.isin(co_sau_moc), index=attr.index)),
    ]
    con = pd.Series(True, index=attr.index)
    tong = len(attr)
    dong = []
    for ten, dk in buoc:
        con &= dk.fillna(False)
        n = int(con.sum())
        dong.append({"Bước lọc": ten, "Số case": n, "Số sự kiện": int(so_sk[con[con].index].sum()),
                     "% case còn lại": f"{n / tong:.1%}"})

    # Trạng thái: quy tắc gốc + kích hoạt lại (V2) + không có IR sau Clear cuối (V3)
    c = buoc3_dem_va_gan_nhan(df, moc).loc[con[con].index]
    hoan_tat = (c["n_clear"] >= 1) & (c["n_clear"] >= c["hd_hop_le"]) & (
        c["cuoi_ir"].isna() | (c["cuoi_clear"] >= c["cuoi_ir"]))
    bi_xoa = (c["n_delete"] >= 1) & (c["n_clear"] == 0) & (
        c["cuoi_reactivate"].isna() | (c["cuoi_delete"] > c["cuoi_reactivate"]))
    tt = pd.Series("Còn mở", index=c.index)
    tt[bi_xoa] = "Bị xóa"
    tt[hoan_tat] = "Hoàn tất"
    for nhan in ["Hoàn tất", "Còn mở", "Bị xóa"]:
        n = int((tt == nhan).sum())
        dong.append({"Bước lọc": f"   Trong đó: {nhan}", "Số case": n,
                     "Số sự kiện": int(so_sk[tt.index[tt == nhan]].sum()),
                     "% case còn lại": f"{n / len(tt):.1%} (trên bước 6)"})

    bang = pd.DataFrame(dong)
    bang.to_csv("bang3_pheu_loc.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(bang.to_string(index=False))
    print("\nĐã ghi bang3_pheu_loc.csv (mở được bằng Excel)")


if __name__ == "__main__":
    main()
