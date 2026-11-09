"""Chạy thử định nghĩa trạng thái dòng hàng (Mục 5.2) trên log BPI Challenge 2019.

Cách chạy:
    python src/kiem_tra_trang_thai_case.py duong_dan_log.xes          # hoặc .xes.gz / .csv
    python src/kiem_tra_trang_thai_case.py duong_dan_log.csv --moc 2019-01-18
    python src/kiem_tra_trang_thai_case.py --demo                     # chạy thử trên log giả lập

Kết quả ghi vào thư mục ket_qua_5_2/ (đổi bằng --ra):
    buoc1_hoat_dong.csv        42 hoạt động: số sự kiện, số case, nhóm
    buoc2_su_kien_theo_thang.csv   số sự kiện theo tháng (để chọn thời điểm trích xuất)
    buoc3_dem_theo_case.csv    số lần mỗi hoạt động liên quan + trạng thái theo từng biến thể
    buoc4_so_sanh_bien_the.csv số case mỗi trạng thái theo từng biến thể quy tắc
    buoc5_case_bien/E*.csv     tối đa 20 case mẫu cho mỗi loại case biên, kèm toàn bộ chuỗi sự kiện
    tom_tat.txt                bản tóm tắt in ra màn hình
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import cau_hinh

# Tên hoạt động trong BPI 2019. Script so khớp không phân biệt hoa thường;
# nếu log của bạn đặt tên khác, sửa ở đây.
HD = {
    "po": "Create Purchase Order Item",
    "ir": "Record Invoice Receipt",
    "cancel_ir": "Cancel Invoice Receipt",
    "clear": "Clear Invoice",
    "sub_inv": "Record Subsequent Invoice",
    "cancel_sub": "Cancel Subsequent Invoice",
    "delete": "Delete Purchase Order Item",
    "reactivate": "Reactivate Purchase Order Item",
    "gr": "Record Goods Receipt",
    "cancel_gr": "Cancel Goods Receipt",
    "vendor_inv": "Vendor creates invoice",
}
# Hoạt động dạng debit memo: lấy mọi hoạt động có chữ "debit memo" trong tên.
MAU_DEBIT_MEMO = re.compile(r"debit\s*memo", re.I)

NHOM_TU_KHOA = [
    ("hoa_don_thanh_toan", re.compile(r"invoice|debit memo|credit memo|payment|clear", re.I)),
    ("xoa_khoa_kich_hoat", re.compile(r"delete|reactivate|block|deleted", re.I)),
    ("nhan_hang", re.compile(r"goods receipt|service entry", re.I)),
    ("thay_doi", re.compile(r"^change|update", re.I)),
]


# ---------------------------------------------------------------- đọc log
def doc_log(duong_dan: str) -> pd.DataFrame:
    p = Path(duong_dan)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, encoding="latin-1", low_memory=False)
    else:
        df = doc_xes_gio_dia_phuong(p)
    return chuan_hoa_cot(df)


def doc_xes_gio_dia_phuong(p: Path) -> pd.DataFrame:
    """Đọc XES và giữ nguyên giờ địa phương như ghi trong file.

    pm4py.read_xes đổi timestamp sang UTC (lùi 1–2 giờ), làm lệch ngày của các sự kiện gần nửa đêm,
    ví dụ Vendor creates invoice ghi 23:59 bị thành 21:59/22:59. Ở đây ta cắt bỏ phần múi giờ
    (+01:00/+02:00) và lấy đúng ngày giờ như trong file.
    """
    import gzip
    import xml.etree.ElementTree as ET

    mo = gzip.open if p.suffix.lower() == ".gz" else open
    dong, trace_attr, trong_event, ev = [], {}, False, {}
    with mo(p, "rb") as f:
        for su_kien, el in ET.iterparse(f, events=("start", "end")):
            tag = el.tag.rsplit("}", 1)[-1]
            if su_kien == "start":
                if tag == "trace":
                    trace_attr = {}
                elif tag == "event":
                    trong_event, ev = True, {}
                continue
            if tag in ("string", "date", "int", "float", "boolean", "id"):
                k, v = el.get("key"), el.get("value")
                if tag == "date":
                    v = v[:19]  # bỏ phần giây lẻ và múi giờ: 2018-01-02T23:59:00
                if trong_event:
                    ev[k] = v
                else:
                    trace_attr["case:" + k] = v
            elif tag == "event":
                dong.append({**trace_attr, **ev})
                trong_event = False
                el.clear()
            elif tag == "trace":
                el.clear()
    df = pd.DataFrame(dong)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="%Y-%m-%dT%H:%M:%S")
    return df


def chuan_hoa_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Đưa tên cột của bản XES (pm4py) hoặc bản CSV về: case, act, ts, user + thuộc tính case."""
    ung_vien = {
        "case": ["case:concept:name", "case concept:name", "case_id", "case"],
        "act": ["concept:name", "event concept:name", "activity", "act"],
        "ts": ["time:timestamp", "event time:timestamp", "timestamp", "ts"],
        "user": ["org:resource", "event org:resource", "event User", "User", "user"],
        "item_category": ["case:Item Category", "case Item Category"],
        "item_type": ["case:Item Type", "case Item Type"],
        "doc_type": ["case:Document Type", "case Document Type"],
        "company": ["case:Company", "case Company"],
    }
    doi = {}
    for chuan, ds in ung_vien.items():
        for c in ds:
            if c in df.columns:
                doi[c] = chuan
                break
    thieu = {"case", "act", "ts"} - set(doi.values())
    if thieu:
        sys.exit(f"Không tìm thấy cột {thieu}. Các cột hiện có: {list(df.columns)[:30]}")
    df = df.rename(columns=doi)[list(doi.values())].copy()

    ts = df["ts"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        # Bản CSV của BPI 2019 dùng dạng 02-01-2018 15:29:00.000 (ngày-tháng-năm)
        ts = pd.to_datetime(ts, format="%d-%m-%Y %H:%M:%S.%f", errors="coerce")
        if ts.isna().mean() > 0.5:
            ts = pd.to_datetime(df["ts"], dayfirst=True, errors="coerce", utc=False)
    if getattr(ts.dt, "tz", None) is not None:
        # Giữ giờ địa phương như trong log, chỉ bỏ thông tin múi giờ
        ts = ts.dt.tz_localize(None)
    df["ts"] = ts
    df["act"] = df["act"].astype(str)
    df["case"] = df["case"].astype(str)
    return df


# ---------------------------------------------------------------- các bước
def buoc1_hoat_dong(df):
    bang = (
        df.groupby("act")
        .agg(so_su_kien=("case", "size"), so_case=("case", "nunique"), dau_tien=("ts", "min"), cuoi_cung=("ts", "max"))
        .sort_values("so_su_kien", ascending=False)
        .reset_index()
    )

    def nhom(ten):
        for n, mau in NHOM_TU_KHOA:
            if mau.search(ten):
                return n
        return "khac"

    bang["nhom"] = bang["act"].map(nhom)
    bang["dung_trong_quy_tac"] = bang["act"].str.lower().isin({v.lower() for v in HD.values()}) | bang[
        "act"
    ].str.contains(MAU_DEBIT_MEMO)
    return bang


def buoc2_moc_trich_xuat(df, moc_tay=None, nguong_thang=1000):
    theo_thang = df.groupby(df["ts"].dt.to_period("M")).size().rename("so_su_kien").reset_index()
    theo_thang["ts"] = theo_thang["ts"].astype(str)
    if moc_tay:
        moc = pd.Timestamp(moc_tay) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        moc = min(moc, df.loc[df["ts"] <= moc, "ts"].max())
        ly_do = f"đặt tay --moc {moc_tay}; lấy sự kiện muộn nhất không sau ngày này"
    else:
        # Tháng cuối cùng còn >= nguong_thang sự kiện được coi là còn dữ liệu bình thường;
        # các tháng sau đó chỉ lác đác vài sự kiện (thường là timestamp lỗi hoặc ngày tương lai).
        dem = df.groupby(df["ts"].dt.to_period("M")).size()
        thang_cuoi = dem[dem >= nguong_thang].index.max()
        moc = df.loc[df["ts"].dt.to_period("M") <= thang_cuoi, "ts"].max()
        ly_do = f"tự động: sự kiện muộn nhất trong tháng cuối còn ≥ {nguong_thang} sự kiện ({thang_cuoi})"
    return moc, ly_do, theo_thang


def _ten_khop(df, ten):
    return df["act"].str.lower() == ten.lower()


def buoc3_dem_va_gan_nhan(df, moc):
    d = df[df["ts"] <= moc]
    cases = pd.DataFrame(index=pd.Index(df["case"].unique(), name="case"))

    for khoa, ten in HD.items():
        sub = d[_ten_khop(d, ten)]
        cases["n_" + khoa] = sub.groupby("case").size()
        cases["cuoi_" + khoa] = sub.groupby("case")["ts"].max()
        cases["dau_" + khoa] = sub.groupby("case")["ts"].min()
    sub = d[d["act"].str.contains(MAU_DEBIT_MEMO)]
    cases["n_debit_memo"] = sub.groupby("case").size()
    cases["n_su_kien_sau_moc"] = df[df["ts"] > moc].groupby("case").size()
    cases["n_su_kien_truoc_2017"] = df[df["ts"] < "2017-01-01"].groupby("case").size()

    so_cot = [c for c in cases.columns if c.startswith("n_")]
    cases[so_cot] = cases[so_cot].fillna(0).astype(int)

    # Phạm vi đề tài (nếu log có đủ cột thuộc tính case)
    attr = df.drop_duplicates("case").set_index("case")
    pv = pd.Series(True, index=cases.index)
    if "item_category" in attr:
        pv &= attr["item_category"].astype(str).str.contains("invoice before GR", case=False).reindex(cases.index)
    if "doc_type" in attr:
        pv &= attr["doc_type"].astype(str).str.contains("Standard PO", case=False).reindex(cases.index)
    if "item_type" in attr:
        pv &= (attr["item_type"].astype(str).str.lower() == "standard").reindex(cases.index)
    pv &= cases["dau_po"].dt.year.eq(2018)
    cases["trong_pham_vi"] = pv.fillna(False)

    # Số hóa đơn hợp lệ theo đề xuất gốc và theo biến thể có tính hóa đơn bổ sung
    cases["hd_hop_le"] = cases["n_ir"] - cases["n_cancel_ir"]
    cases["hd_hop_le_co_bo_sung"] = cases["hd_hop_le"] + cases["n_sub_inv"] - cases["n_cancel_sub"]

    def gan(hoan_tat, bi_xoa):
        return pd.Series(
            pd.Categorical(
                ["hoan_tat" if h else "bi_xoa" if x else "con_mo" for h, x in zip(hoan_tat, bi_xoa)],
                categories=["hoan_tat", "con_mo", "bi_xoa"],
            ),
            index=cases.index,
        )

    co_clear = cases["n_clear"] >= 1
    du_clear = cases["n_clear"] >= cases["hd_hop_le"]
    xoa_goc = (cases["n_delete"] >= 1) & (cases["n_clear"] == 0)
    # Xóa còn hiệu lực: lần xóa cuối muộn hơn lần kích hoạt lại cuối (hoặc không kích hoạt lại)
    xoa_hieu_luc = (cases["n_delete"] >= 1) & (
        cases["cuoi_reactivate"].isna() | (cases["cuoi_delete"] > cases["cuoi_reactivate"])
    )
    khong_ir_sau_clear = cases["cuoi_ir"].isna() | (cases["cuoi_clear"] >= cases["cuoi_ir"])

    cases["V0_de_xuat_goc"] = gan(co_clear & du_clear, xoa_goc)
    cases["V1_tinh_hd_bo_sung"] = gan(co_clear & (cases["n_clear"] >= cases["hd_hop_le_co_bo_sung"]), xoa_goc)
    cases["V2_xoa_tru_kich_hoat_lai"] = gan(co_clear & du_clear, xoa_hieu_luc & (cases["n_clear"] == 0))
    cases["V3_clear_cuoi_sau_ir_cuoi"] = gan(co_clear & du_clear & khong_ir_sau_clear, xoa_goc)
    return cases


BIEN_THE = ["V0_de_xuat_goc", "V1_tinh_hd_bo_sung", "V2_xoa_tru_kich_hoat_lai", "V3_clear_cuoi_sau_ir_cuoi"]


def buoc4_so_sanh(cases):
    dong = []
    for pham_vi, c in [("toan_log", cases), ("trong_pham_vi", cases[cases["trong_pham_vi"]])]:
        for v in BIEN_THE:
            dem = c[v].value_counts().reindex(["hoan_tat", "con_mo", "bi_xoa"], fill_value=0)
            doi = int((c[v] != c["V0_de_xuat_goc"]).sum())
            dong.append({"pham_vi": pham_vi, "bien_the": v, **dem.to_dict(), "tong": len(c), "so_case_doi_so_voi_V0": doi})
    return pd.DataFrame(dong)


def buoc5_case_bien(cases):
    v0 = cases["V0_de_xuat_goc"]
    return {
        "E1_clear_nhieu_hon_hd_hop_le": (v0 == "hoan_tat") & (cases["n_clear"] > cases["hd_hop_le"]),
        "E2_co_clear_nhung_khong_co_hd_hop_le": (cases["n_clear"] >= 1) & (cases["hd_hop_le"] <= 0),
        "E3_bi_xoa_nhung_co_clear": (cases["n_delete"] >= 1) & (cases["n_clear"] >= 1),
        "E4_xoa_roi_kich_hoat_lai": (cases["n_delete"] >= 1) & (cases["n_reactivate"] >= 1),
        "E5_co_hd_bo_sung": cases["n_sub_inv"] >= 1,
        "E6_co_debit_memo": cases["n_debit_memo"] >= 1,
        "E7_hoan_tat_nhung_co_ir_sau_clear_cuoi": (v0 == "hoan_tat") & (cases["cuoi_ir"] > cases["cuoi_clear"]),
        "E8_huy_hd_nhieu_hon_ghi_nhan": cases["hd_hop_le"] < 0,
        "E9_co_su_kien_sau_moc": cases["n_su_kien_sau_moc"] >= 1,
        "E10_con_mo_khong_co_ir": (v0 == "con_mo") & (cases["n_ir"] == 0),
    }


# ---------------------------------------------------------------- log giả lập
def log_gia_lap() -> pd.DataFrame:
    """Vài case viết tay, mỗi case minh họa một tình huống; chỉ để thử script."""
    t0 = pd.Timestamp("2018-03-01 08:00")
    kich_ban = {
        "A_binh_thuong": ["PO", "GR", "IR", "CLEAR"],
        "B_hai_hd_mot_clear": ["PO", "GR", "IR", "IR", "CLEAR"],
        "C_huy_hd": ["PO", "GR", "IR", "IR", "CANCEL_IR", "CLEAR"],
        "D_xoa": ["PO", "DELETE"],
        "E_xoa_roi_kich_hoat": ["PO", "DELETE", "REACT", "GR", "IR"],
        "F_hd_bo_sung": ["PO", "GR", "IR", "CLEAR", "SUB"],
        "G_debit_memo": ["PO", "GR", "IR", "MEMO", "CLEAR", "CLEAR"],
        "H_ir_sau_clear": ["PO", "GR", "IR", "IR", "CLEAR", "CLEAR", "CANCEL_IR", "IR"],
        "I_chua_hd": ["PO", "GR"],
        "J_sau_moc": ["PO", "GR", "IR", "CLEAR*"],
    }
    ten = {"PO": HD["po"], "GR": HD["gr"], "IR": HD["ir"], "CLEAR": HD["clear"], "CANCEL_IR": HD["cancel_ir"],
           "DELETE": HD["delete"], "REACT": HD["reactivate"], "SUB": HD["sub_inv"], "MEMO": "Vendor creates debit memo"}
    dong = []
    for case, seq in kich_ban.items():
        for i, a in enumerate(seq):
            ts = t0 + pd.Timedelta(days=7 * i)
            if a.endswith("*"):
                a, ts = a[:-1], pd.Timestamp("2020-04-01")
            dong.append({"case": case, "act": ten[a], "ts": ts})
    # thêm nền để bước 2 thấy đủ "tháng bình thường"
    for k in range(13):
        dong += [{"case": f"Z{k}_{j}", "act": HD["po"], "ts": pd.Timestamp("2018-01-05") + pd.DateOffset(months=k)} for j in range(3)]
    return pd.DataFrame(dong)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(cau_hinh.LOG), help="đường dẫn log; mặc định trong cau_hinh.py")
    ap.add_argument("--moc", help="thời điểm trích xuất đặt tay, dạng YYYY-MM-DD")
    ap.add_argument("--nguong-thang", type=int, default=1000)
    ap.add_argument("--ra", default="ket_qua_5_2")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        df, a.nguong_thang = log_gia_lap(), 3
    elif a.log:
        df = doc_log(a.log)
    else:
        ap.error("cần đường dẫn log hoặc --demo")

    ra = Path(a.ra)
    (ra / "buoc5_case_bien").mkdir(parents=True, exist_ok=True)
    out = []

    def p(*x):
        s = " ".join(str(i) for i in x)
        print(s)
        out.append(s)

    with pd.option_context("display.width", 200, "display.max_rows", 100, "display.max_columns", 20):
        p(f"Số case: {df['case'].nunique():,} | số sự kiện: {len(df):,} | số hoạt động: {df['act'].nunique()}")
        p(f"Timestamp lỗi (không đọc được): {df['ts'].isna().sum():,}")

        hd = buoc1_hoat_dong(df)
        hd.to_csv(ra / "buoc1_hoat_dong.csv", index=False)
        p("\n=== BƯỚC 1. Danh sách hoạt động ===")
        p(hd[["act", "so_su_kien", "so_case", "nhom", "dung_trong_quy_tac"]].to_string(index=False))
        thieu = [t for t in HD.values() if t.lower() not in set(hd["act"].str.lower())]
        if thieu:
            p("Lưu ý: không thấy các tên này trong log (kiểm tra chính tả):", thieu)
        p("\nHoạt động nhóm hóa đơn/thanh toán CHƯA dùng trong quy tắc (cần quyết định có tính hay không):")
        p(hd.loc[(hd["nhom"] == "hoa_don_thanh_toan") & ~hd["dung_trong_quy_tac"], ["act", "so_su_kien", "so_case"]].to_string(index=False))

        moc, ly_do, theo_thang = buoc2_moc_trich_xuat(df, a.moc, a.nguong_thang)
        theo_thang.to_csv(ra / "buoc2_su_kien_theo_thang.csv", index=False)
        p("\n=== BƯỚC 2. Thời điểm trích xuất ===")
        p(f"Mốc = {moc}  ({ly_do})")
        p(f"Sự kiện sau mốc: {(df['ts'] > moc).sum():,} sự kiện / {df.loc[df['ts'] > moc, 'case'].nunique():,} case")
        p(f"Sự kiện trước 2017: {(df['ts'] < '2017-01-01').sum():,} sự kiện / {df.loc[df['ts'] < '2017-01-01', 'case'].nunique():,} case")
        p("10 tháng cuối:\n" + theo_thang.tail(10).to_string(index=False))

        cases = buoc3_dem_va_gan_nhan(df, moc)
        cases.to_csv(ra / "buoc3_dem_theo_case.csv")

        ss = buoc4_so_sanh(cases)
        ss.to_csv(ra / "buoc4_so_sanh_bien_the.csv", index=False)
        p("\n=== BƯỚC 3–4. Số case theo trạng thái, từng biến thể quy tắc ===")
        p(ss.to_string(index=False))

        p("\n=== BƯỚC 5. Case biên (toàn log | trong phạm vi) ===")
        for ten_e, mask in buoc5_case_bien(cases).items():
            mask = mask.fillna(False)
            ids = cases.index[mask][:20]
            df[df["case"].isin(ids)].sort_values(["case", "ts"]).to_csv(ra / "buoc5_case_bien" / f"{ten_e}.csv", index=False)
            p(f"{ten_e:45s} {int(mask.sum()):>8,} | {int((mask & cases['trong_pham_vi']).sum()):>8,}")

    (ra / "tom_tat.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}")


if __name__ == "__main__":
    main()
