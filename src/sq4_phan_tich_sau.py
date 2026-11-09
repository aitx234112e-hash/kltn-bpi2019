"""SQ4, phân tích sau khi xem kết quả (góp ý G10, 06/11/2026). Không huấn luyện lại mô hình nào: chỉ đọc các file mà
sq4_mo_hinh.py đã ghi (điểm dự báo trên tập kiểm tra, chi_so.csv, tham_so.json).

  1. Tiền tố 5 chưa có Clear Invoice "thanh toán": ở tiền tố 5, số lần Clear thanh toán đứng đầu SHAP; dòng đã thanh toán
     trước khi có dấu hiệu thì gần như không còn dấu hiệu. Tính lại PR-AUC, KTC bootstrap theo đơn mua hàng (1.000 lần) và
     chênh lệch ghép cặp trên các tiền tố 5 chưa có Clear thanh toán, dùng đúng điểm dự báo đã có; Holm trong từng nhãn.
  2. PR-AUC kiểm định chéo (của bộ tham số đã chọn, trên tập huấn luyện) cạnh PR-AUC tập kiểm tra, cả hai chia cho tỷ lệ
     nhãn 1 của tập tương ứng.

Cách chạy (sau khi đã chạy sq4_mo_hinh.py với bootstrap theo đơn, để có file diem_kiem_tra_*.csv):
    python src/sq4_phan_tich_sau.py ket_qua_sq4_mo_hinh_boot_don
Kết quả ghi vào cùng thư mục: phan_tich_sau.txt, tien_to5_chua_thanh_toan.csv, cv_va_kiem_tra.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sq4_mo_hinh as mh
import cau_hinh

MO_HINH = ["XGBoost", "Logistic Regression", "Mốc 1: biến lúc tạo", "Mốc 2: tỷ lệ của nhà cung cấp"]
CAP = [("XGBoost", "Logistic Regression"), ("XGBoost", "Mốc 1: biến lúc tạo"), ("XGBoost", "Mốc 2: tỷ lệ của nhà cung cấp"),
       ("Logistic Regression", "Mốc 1: biến lúc tạo"), ("Logistic Regression", "Mốc 2: tỷ lệ của nhà cung cấp")]
vn = mh.vn


def bo_sung_co_clear(ra, log, p):
    """Bản chạy cũ chưa lưu cờ 'tiền tố đã có Clear thanh toán': đọc log, cắt đúng như sq4_mo_hinh.py (5 sự kiện đầu,
    Clear Invoice đã tách thanh toán / bù trừ) và ghi thêm cột vào file điểm dự báo. Không huấn luyện lại."""
    from r5_v2_phan_loai_clear import buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear, sap_xep
    raw = mh.sq2.doc_log(log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, "companyID_0000", moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    loai, _, _ = phan_loai_clear(ev, mh.g4.DELTA)
    e = mh.ev_tien_to(ev, loai, False)
    x = mh.dem_tien_to(e, 5, ["Clear Invoice (thanh toán)"])
    co = x["n_Clear Invoice (thanh toán)"] > 0
    for f in sorted(ra.glob("diem_kiem_tra_*_k5.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig")
        if "tien_to_co_clear_tt" not in d:
            d.insert(3, "tien_to_co_clear_tt", d["case"].map(co).fillna(False).astype(bool).values)
            d.to_csv(f, index=False, encoding="utf-8-sig")
            p(f"  Đã thêm cột tien_to_co_clear_tt vào {f.name} (từ log, không huấn luyện lại)")


def tien_to_5(ra, p, so_boot=mh.SO_BOOT):
    p("1. Tiền tố 5: tách theo tiền tố đã có hay chưa có Clear Invoice \"thanh toán\" (phân tích sau khi xem kết quả)")
    rows, ss_all = [], []
    for f in sorted(ra.glob("diem_kiem_tra_*_k5.csv")):
        ma = f.stem.replace("diem_kiem_tra_", "").rsplit("_k", 1)[0]
        d = pd.read_csv(f, encoding="utf-8-sig")
        for nhom, m in [("chưa có Clear thanh toán", ~d["tien_to_co_clear_tt"].astype(bool)),
                        ("đã có Clear thanh toán", d["tien_to_co_clear_tt"].astype(bool)), ("toàn bộ", pd.Series(True, index=d.index))]:
            x = d[m]
            y = x["y"].values.astype(int)
            if y.sum() == 0:
                p(f"  {ma}, {nhom}: {vn(len(x))} dòng, không có nhãn 1")
                continue
            diem = {mo: x[mo].values for mo in MO_HINH}
            kq = mh.bootstrap(diem, y, so_boot, nhom=x["don"].values)
            ci, ss = mh.tom_tat_boot(kq, CAP, {mo: mh.ap_trong_so(mh.chuan_bi_ap(s, y)) for mo, s in diem.items()})
            sao = " (*)" if y.sum() < mh.NGUONG else ""
            p(f"  {ma}, {nhom}: {vn(len(x))} dòng, {vn(int(y.sum()))} nhãn 1 ({vn(y.mean(), '.2%')}){sao}")
            for mo in MO_HINH:
                ap = mh.ap_trong_so(mh.chuan_bi_ap(diem[mo], y))
                rows.append({"ma": ma, "nhom": nhom, "mo_hinh": mo, "n": len(x), "n1": int(y.sum()), "ty_le_nhan_1": y.mean(),
                             "pr_auc": ap, "ktc_duoi": ci[mo][0], "ktc_tren": ci[mo][1], "chi_bao_cao": y.sum() < mh.NGUONG})
                p(f"    {mo:32s} PR-AUC {mh.so4(ap)} ({mh.so4(ci[mo][0])}–{mh.so4(ci[mo][1])}); gấp {vn(ap / y.mean(), '.2f')} lần tỷ lệ nhãn 1")
            if nhom == "chưa có Clear thanh toán":
                ss_all += [{"ma": ma, **s} for s in ss]
    if ss_all:
        s = pd.DataFrame(ss_all)
        s["p_holm"] = np.nan
        for ma, g in s.groupby("ma"):
            s.loc[g.index, "p_holm"] = mh.holm(g["p_hai_phia"].fillna(1.0).values)
        p("  So sánh trên tiền tố 5 chưa có Clear thanh toán (Holm trong từng nhãn, 5 kiểm định):")
        for _, r in s.iterrows():
            p(f"    {r['ma']}: {r['mo_hinh']} − {r['so_voi']}: {mh.so4(r['chenh_pr_auc'])} ({mh.so4(r['ktc_duoi'])}–"
              f"{mh.so4(r['ktc_tren'])}); p Holm {mh.sq3.so_p(r['p_holm'])}")
        s.to_csv(ra / "tien_to5_chua_thanh_toan_so_sanh.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(ra / "tien_to5_chua_thanh_toan.csv", index=False, encoding="utf-8-sig")


def holm_lai(ra, p):
    """Thêm p Holm (trong từng nhãn) vào so_sanh.csv nếu bản chạy chưa có."""
    s = pd.read_csv(ra / "so_sanh.csv", encoding="utf-8-sig")
    s["p_holm"] = np.nan
    for ma, g in s.groupby("ma"):
        s.loc[g.index, "p_holm"] = mh.holm(g["p_hai_phia"].fillna(1.0).values)
    s.to_csv(ra / "so_sanh.csv", index=False, encoding="utf-8-sig")
    p("0. So sánh có ý nghĩa sau hiệu chỉnh Holm trong từng nhãn (p Holm < 0,05):")
    for ma, g in s.groupby("ma", sort=False):
        co = g[g["p_holm"] < 0.05]
        p(f"  {ma}: " + ("; ".join(f"tiền tố {int(r['tien_to'])}: {r['mo_hinh']} − {r['so_voi']} {mh.so4(r['chenh_pr_auc'])} "
                                  f"(p Holm {mh.sq3.so_p(r['p_holm'])})" for _, r in co.iterrows()) or "không có"))
    p("")


def cv_va_kiem_tra(ra, p):
    p("\n2. PR-AUC kiểm định chéo (tập huấn luyện) và tập kiểm tra, chia cho tỷ lệ nhãn 1 của tập tương ứng")
    cs = pd.read_csv(ra / "chi_so.csv", encoding="utf-8-sig")
    th = json.loads((ra / "tham_so.json").read_text(encoding="utf-8"))
    khoa = {"Logistic Regression": ("lr", "cv_lr"), "XGBoost": ("xgb", "cv_xgb"), "Mốc 1: biến lúc tạo": ("moc1", "cv_moc1")}
    rows = []
    for t in th:
        for mo, (k_ts, k_cv) in khoa.items():
            chon = t[k_ts]
            cv = next((c["pr_auc_cv"] for c in t[k_cv] if all(c.get(a) == b for a, b in chon.items())), np.nan)
            r = cs[(cs["ma"] == t["ma"]) & (cs["tien_to"] == t["tien_to"]) & (cs["mo_hinh"] == mo)]
            if r.empty:
                continue
            r = r.iloc[0]
            tl_hl = r["n1_huan_luyen"] / r["n_huan_luyen"]
            rows.append({"ma": t["ma"], "tien_to": t["tien_to"], "mo_hinh": mo, "ty_le_nhan_1_huan_luyen": tl_hl,
                         "pr_auc_cv": cv, "cv_chia_ty_le": cv / tl_hl, "ty_le_nhan_1_kiem_tra": r["ty_le_nhan_1"],
                         "pr_auc_kiem_tra": r["pr_auc"], "kiem_tra_chia_ty_le": r["pr_auc"] / r["ty_le_nhan_1"]})
    x = pd.DataFrame(rows)
    x.to_csv(ra / "cv_va_kiem_tra.csv", index=False, encoding="utf-8-sig")
    for _, r in x[x["ma"].isin(["chinh", "R3"])].iterrows():
        p(f"  {r['ma']:6s} tiền tố {int(r['tien_to'])} {r['mo_hinh']:22s}: kiểm định chéo {mh.so4(r['pr_auc_cv'])} "
          f"(gấp {vn(r['cv_chia_ty_le'], '.1f')} lần {vn(r['ty_le_nhan_1_huan_luyen'], '.2%')}); kiểm tra "
          f"{mh.so4(r['pr_auc_kiem_tra'])} (gấp {vn(r['kiem_tra_chia_ty_le'], '.1f')} lần {vn(r['ty_le_nhan_1_kiem_tra'], '.2%')})")
    if len(x):
        p(f"  Khoảng chung (mọi cấu hình): kiểm định chéo gấp {vn(x['cv_chia_ty_le'].min(), '.1f')}–"
          f"{vn(x['cv_chia_ty_le'].max(), '.1f')} lần; kiểm tra gấp {vn(x['kiem_tra_chia_ty_le'].min(), '.1f')}–"
          f"{vn(x['kiem_tra_chia_ty_le'].max(), '.1f')} lần tỷ lệ nhãn 1.")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("ra", help="thư mục kết quả của sq4_mo_hinh.py (bản bootstrap theo đơn)")
    ap.add_argument("--log", default=str(cau_hinh.LOG), help="đường dẫn log; chỉ cần khi file điểm dự báo chưa có cột tien_to_co_clear_tt")
    a = ap.parse_args()
    ra = Path(a.ra)
    out = []

    def p(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True)
        out.append(s)

    import datetime
    p(f"Phân tích sau khi xem kết quả, chạy lúc {datetime.datetime.now():%d/%m/%Y %H:%M}. Không huấn luyện lại; "
      f"dùng điểm dự báo trong {ra}.")
    if not list(ra.glob("diem_kiem_tra_*_k5.csv")):
        sys.exit("Không có file diem_kiem_tra_*_k5.csv: chạy lại sq4_mo_hinh.py bản mới (bootstrap theo đơn) trước.")
    thieu = [f for f in ra.glob("diem_kiem_tra_*_k5.csv")
             if "tien_to_co_clear_tt" not in pd.read_csv(f, nrows=1, encoding="utf-8-sig").columns]
    if thieu:
        if not a.log:
            sys.exit("File điểm dự báo chưa có cột tien_to_co_clear_tt: chạy lại với --log <đường dẫn log>.")
        bo_sung_co_clear(ra, a.log, p)
    holm_lai(ra, p)
    tien_to_5(ra, p)
    cv_va_kiem_tra(ra, p)
    (ra / "phan_tich_sau.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi {ra / 'phan_tich_sau.txt'}")


if __name__ == "__main__":
    main()
