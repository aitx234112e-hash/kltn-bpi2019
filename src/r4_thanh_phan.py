"""R4 v2 gồm những gì? (bổ sung cho G3 Mục 3, 16/10/2026)

Kết quả G3: chỉ 15/1.566 dòng hàng R4 v2 có Clear "thanh toán" lúc không còn chứng từ phải trả (nhóm B).
Script này xem với 1.551 dòng hàng còn lại, tại Clear bị R4 bắt, cái gì còn "phải trả":
hóa đơn chưa bị hủy (#IR − #Cancel IR > 0) hay chỉ còn debit memo.

Chạy (cùng thư mục với r5_v2_phan_loai_clear.py, g3_kiem_tra_bo_sung.py, kiem_tra_trang_thai_case.py):
    python src/r4_thanh_phan.py
Kết quả: ket_qua_r4/tom_tat_r4.txt, r4_tung_case.csv, mau_kiem_chung/r4v2_mau.csv (20 case để đọc tay).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import g3_kiem_tra_bo_sung as g3
import r5_v2_phan_loai_clear as r5
from r5_v2_phan_loai_clear import CANCEL_IR, CLEAR, IR, MEMO, loc_pham_vi, phan_loai_clear, sap_xep
import cau_hinh


def thanh_phan_r4(ev, loai):
    tt = set(loai.index[loai == "thanh_toan"])
    e = ev[ev["act"].isin([IR, CANCEL_IR, MEMO, CLEAR])].copy()
    e = e[(e["act"] != CLEAR) | e.index.isin(tt)]
    for k, a in {"n_ir": IR, "n_huy": CANCEL_IR, "n_memo": MEMO, "n_tra_truoc": CLEAR}.items():
        e[k] = (e["act"] == a).astype(int).groupby(e["case"]).cumsum()
    e["n_tra_truoc"] -= (e["act"] == CLEAR).astype(int)  # số lần trả TRƯỚC lần này
    # điều kiện R4 v2 tại từng Clear (giống r4_theo_clear)
    e["tc"] = e["ts"].where(e["act"] == CANCEL_IR)
    e["ti"] = e["ts"].where(e["act"] == IR)
    e[["tc", "ti"]] = e.groupby("case")[["tc", "ti"]].ffill()
    m = (e["act"] == CLEAR) & e["tc"].notna() & (e["ti"].isna() | (e["ti"] < e["tc"]))
    x = e[m].copy()
    x["hd_con"] = (x["n_ir"] - x["n_huy"]).clip(lower=0)
    x["hop_le"] = x["hd_con"] + x["n_memo"]
    x["con_phai_tra"] = x["hop_le"] - x["n_tra_truoc"]
    x["ngay_tu_huy"] = (x["ts"] - x["tc"]).dt.total_seconds() / 86400
    x["loai_r4"] = np.select(
        [x["hop_le"] == 0, (x["hd_con"] > 0) & (x["con_phai_tra"] > 0), (x["hd_con"] == 0) & (x["con_phai_tra"] > 0)],
        ["B: không còn chứng từ nào", "A1: còn hóa đơn chưa hủy, chưa trả hết", "A2: chỉ còn debit memo, chưa trả hết"],
        "A3: còn chứng từ nhưng đã trả đủ trước đó (lần này là trả thêm)")
    return x


def main():
    if len(sys.argv) < 2 and not cau_hinh.LOG.exists():
        sys.exit("cần đường dẫn log")
    ra = Path("ket_qua_r4")
    (ra / "mau_kiem_chung").mkdir(parents=True, exist_ok=True)
    out = []

    def p(*a):
        s = " ".join(str(i) for i in a)
        print(s)
        out.append(s)

    raw = g3.doc_log(sys.argv[1] if len(sys.argv) > 1 else str(cau_hinh.LOG))
    moc, _, _ = r5.buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, "companyID_0000", moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    loai, _, _ = phan_loai_clear(ev, g3.DELTA)
    ev["loai_clear"] = loai.reindex(ev.index)
    x = thanh_phan_r4(ev, loai)
    r4 = set(x["case"])
    p(f"R4 v2: {len(r4):,} dòng hàng, {len(x):,} lần Clear bị bắt")
    # mỗi dòng hàng lấy lần Clear bị R4 bắt đầu tiên
    dau = x.sort_values("ts").groupby("case").first()
    bang = dau["loai_r4"].value_counts().rename("so_case").to_frame()
    bang["ty_le"] = (bang["so_case"] / len(dau)).round(4)
    p("\nPhân loại theo lần Clear bị R4 bắt đầu tiên:\n" + bang.to_string())
    p("\nKhoảng cách từ lần hủy đến Clear (ngày), theo loại: P25 / P50 / P75")
    for k, g in dau.groupby("loai_r4"):
        q = g["ngay_tu_huy"].quantile([.25, .5, .75])
        p(f"  {k}: {q.iloc[0]:.1f} / {q.iloc[1]:.1f} / {q.iloc[2]:.1f}; người làm Clear nhiều nhất: "
          + ", ".join(f"{u} {n}" for u, n in g["user"].value_counts().head(3).items()))
    dau.to_csv(ra / "r4_tung_case.csv", encoding="utf-8-sig")
    rng = np.random.default_rng(g3.SEED)
    ds = sorted(r4 - g3.DA_DOC)
    chon = list(rng.choice(ds, size=min(20, len(ds)), replace=False))
    vet = ev[ev["case"].isin(chon)][["case", "ts", "act", "loai_clear", "user", "networth"]].copy()
    vet["loai_r4"] = vet["case"].map(dau["loai_r4"])
    vet["nhan_xet"] = ""
    vet.to_csv(ra / "mau_kiem_chung" / "r4v2_mau.csv", index=False, encoding="utf-8-sig")
    p(f"\nMẫu R4 v2: lấy {len(chon)} case")
    (ra / "tom_tat_r4.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Đã chạy xong.")


if __name__ == "__main__":
    main()
