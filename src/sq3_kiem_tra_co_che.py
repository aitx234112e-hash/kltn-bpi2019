"""Kiểm tra cơ chế sau kết quả SQ3 (26/10/2026). Dùng lại các hàm của sq3_thoi_gian.py, không sửa thiết kế.

  1. Khoảng cách từ dấu hiệu (R2a, R3, làm lại) đến lần thanh toán cuối cùng trên dòng hoàn tất: dấu hiệu có nằm sát
     ngay trước thanh toán không (khi đó HR lớn phản ánh một bước của đường thanh toán, không phải yếu tố dự báo).
  2. Chỉ báo làm lại đổi chiều ở độ nhạy (1): bao nhiêu dòng làm lại đổi trạng thái giữa hai định nghĩa.
  3. Biến thể phụ của R7: mốc thời gian. Dòng đơn hàng được tạo sau ngày hóa đơn của nhà cung cấp thì thời gian tính từ
     Create Purchase Order Item ngắn đi; so sánh thêm thời gian tính từ ngày hóa đơn của nhà cung cấp.

Cách chạy (cùng thư mục với sq3_thoi_gian.py):
    python src/sq3_kiem_tra_co_che.py
Kết quả ghi vào ket_qua_sq3/tom_tat_co_che.txt; gửi lại file này.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import kiem_chung_cuoi as kc
import sq2_tuan_thu as sq2
import sq3_thoi_gian as sq3
from r5_v2_phan_loai_clear import MEMO, buoc2_moc_trich_xuat, loc_pham_vi, sap_xep
import cau_hinh

vn, pct, so = sq3.vn, sq3.pct, sq3.so


def phan_vi(s):
    s = pd.Series(s).dropna()
    return f"{so(s.median())} [{so(s.quantile(0.25))}–{so(s.quantile(0.75))}]" if len(s) else "–"


def phan_tich(ev, pv, moc, p):
    d, _, _ = sq3.chuan_bi(ev, pv, moc, p)
    x, _ = sq3.tap_phan_tich(d, "tt_moi", moc)
    ht = x[x["E"] == 1]

    p("\n=== 1. Từ dấu hiệu đến lần thanh toán cuối cùng (dòng hoàn tất, dấu hiệu trước thanh toán) ===")
    for l in sq3.LUAT_TV + ["R6"]:
        m = ht["co_" + l] & (ht["s_" + l] < ht["t"])
        kc_ = ht.loc[m, "t"] - ht.loc[m, "s_" + l]
        p(f"{sq3.TEN.get(l, l) if l != 'R6' else 'R6 (nhận hàng do người tạo đơn)'}: {vn(int(m.sum()))} dòng; ngày từ dấu hiệu "
          f"đến thanh toán {phan_vi(kc_)}; trong 7 ngày {pct(int((kc_ <= 7).sum()), int(m.sum()), 1)}, trong 14 ngày "
          f"{pct(int((kc_ <= 14).sum()), int(m.sum()), 1)}; dấu hiệu ở ngày {phan_vi(ht.loc[m, 's_' + l])} kể từ khi tạo")
    p("So sánh: toàn bộ dòng hoàn tất, ngày từ ghi nhận hóa đơn cuối cùng đến thanh toán "
      f"{phan_vi((ht['cuoi_clear_tt'] - ev[ev['act'] == sq3.IR].groupby('case')['ts'].max().reindex(ht.index)) / sq3.MOT_NGAY)}")

    p("\n=== 2. Làm lại và định nghĩa trạng thái ===")
    co = d["co_lam_lai"]
    bang = pd.crosstab(d.loc[co, "tt_moi"], d.loc[co, "tt_cu"])
    p("Dòng làm lại, hàng = định nghĩa chính, cột = định nghĩa có debit memo:")
    for dong in bang.to_string().splitlines():
        p("  " + dong)
    doi = d["tt_moi"] != d["tt_cu"]
    memo = d.index.isin(set(ev.loc[ev["act"] == MEMO, "case"]))
    p(f"Dòng đổi trạng thái giữa hai định nghĩa: {vn(int(doi.sum()))}; trong đó là dòng làm lại {pct(int((doi & co).sum()), int(doi.sum()), 1)}")
    p(f"Dòng làm lại có debit memo: {pct(int((co & memo).sum()), int(co.sum()), 1)}; dòng không làm lại có debit memo: "
      f"{pct(int((~co & memo).sum()), int((~co).sum()), 2)}")

    p("\n=== 3. Biến thể phụ của R7: mốc thời gian ===")
    vi = kc.dau(ev, kc.VENDOR_INV).reindex(x.index).dt.normalize()
    po = x["dau_po"].dt.normalize()
    r7 = x["co_R7b"]
    p(f"Ngày hóa đơn của nhà cung cấp sớm hơn ngày tạo dòng đơn hàng (R7b): {phan_vi(((po - vi) / sq3.MOT_NGAY)[r7])} ngày")
    for ten, goc in [("tạo dòng đơn hàng", x["dau_po"]), ("ngày hóa đơn nhà cung cấp", vi)]:
        t = (x["cuoi_clear_tt"] - goc) / sq3.MOT_NGAY
        k1 = sq3.km(np.ceil(((pd.Timestamp(moc) - goc) / sq3.MOT_NGAY).where(x["E"] == 0, t)[r7 & goc.notna()].clip(lower=1)),
                    x.loc[r7 & goc.notna(), "E"])
        k0 = sq3.km(np.ceil(((pd.Timestamp(moc) - goc) / sq3.MOT_NGAY).where(x["E"] == 0, t)[~r7 & goc.notna()].clip(lower=1)),
                    x.loc[~r7 & goc.notna(), "E"])
        p(f"  Tính từ {ten}: dòng hoàn tất R7b {phan_vi(t[r7 & (x['E'] == 1)])}, không R7b {phan_vi(t[~r7 & (x['E'] == 1)])}; "
          f"trung vị KM {so(sq3.phan_vi_km(k1, 0.5), 0)} so với {so(sq3.phan_vi_km(k0, 0.5), 0)} ngày")
    p(f"  Dòng không có ngày hóa đơn nhà cung cấp: {vn(int(vi.isna().sum()))}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(cau_hinh.LOG), help="đường dẫn log; mặc định trong cau_hinh.py")
    ap.add_argument("--cong-ty", default="companyID_0000")
    ap.add_argument("--ra", default="ket_qua_sq3")
    a = ap.parse_args()
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)
    out = []

    def p(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        out.append(s)

    t = time.time()
    raw = sq2.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    p(f"Đọc log xong sau {vn(round(time.time() - t))} giây.")
    phan_tich(ev, pv, moc, p)
    (ra / "tom_tat_co_che.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi {(ra / 'tom_tat_co_che.txt').resolve()}. Gửi lại file này.")


if __name__ == "__main__":
    main()
