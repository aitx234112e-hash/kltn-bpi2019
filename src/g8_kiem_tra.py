"""Kiểm tra theo góp ý ngày 28/10/2026 (G8). Phân tích độ nhạy và mô tả; không sửa luật đã đóng băng, không sửa thiết kế SQ3.

  Mục 2. Mẫu bổ sung R6 "khác thời điểm": tập "đã đọc" chỉ lấy từ file mẫu và file nhận xét đọc tay; rút 20 case chưa đọc
         (10 của vendorID_0741 và vendorID_0572, 10 của các nhà cung cấp khác). Phép đếm thêm cho R3: người sửa giá có là
         người tạo dòng đơn hàng không, và có Change Approval for Purchase Order sau lần sửa giá không.
  Mục 3.1. R6 và nhà cung cấp: Cox phân tầng theo nhà cung cấp (≥ 50 dòng hàng một tầng, nhà cung cấp nhỏ gộp một tầng);
           loại vendorID_0741 và vendorID_0572 rồi chạy lại mô hình chính và RMST; tách R6 thành "cùng phút" và "khác thời điểm".
  Mục 3.2. Debit memo và chỉ báo làm lại: ngày debit memo so với ngày hủy và ngày ghi lại hóa đơn; bảng chéo k (G4) theo
           nhóm làm lại / không làm lại; 10 dòng làm lại để đọc.
  Mục 3.3. R7 biến thể phụ với mốc chung = thời điểm sớm hơn giữa tạo dòng đơn hàng và ngày hóa đơn của nhà cung cấp (dòng
           không có ngày hóa đơn dùng ngày tạo dòng đơn hàng); bản loại vendorID_0550.
  Mục 3.4. Mô hình chính chỉ dùng biến kiểm soát biết được lúc tạo dòng đơn hàng (hạng mục, log giá trị kèm biến giá trị 0,
           quy mô nhà cung cấp).
  Hình 4.4 vẽ lại thành hai hàng, ô đầu mỗi hàng có nhãn và số trên trục tung.

Cách chạy (cùng thư mục với sq3_thoi_gian.py; cần các thư mục kết quả có file mẫu đã đọc tay):
    python src/g8_kiem_tra.py --tu-kiem
    python src/g8_kiem_tra.py
Kết quả ghi vào ket_qua_g8/; gửi lại tom_tat_g8.txt. Hai file *_nhan_xet.csv là file để đọc tay.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import g3_kiem_tra_bo_sung as g3
import g4_kiem_tra as g4
import g5_kiem_tra as g5
import kiem_chung_cuoi as kc
import sq2_bo_sung as sq2b
import sq2_tuan_thu as sq2
import sq3_thoi_gian as sq3
from r5_v2_phan_loai_clear import (CANCEL_IR, IR, MEMO, PO, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear,
                                   sap_xep)
import cau_hinh

SEED = 2026
HAI_NCC_R6 = ["vendorID_0741", "vendorID_0572"]
NCC_R7 = "vendorID_0550"
NGUONG_TANG = 50  # nhà cung cấp có từ 50 dòng hàng trở lên là một tầng riêng
CHANGE_APPROVAL = "Change Approval for Purchase Order"
THU_MUC_MAU_DOC = ["ket_qua_g3/mau_kiem_chung", "ket_qua_r4/mau_kiem_chung", "ket_qua_g4/mau_kiem_chung",
                   "ket_qua_g5/mau_kiem_chung", "ket_qua_kiem_chung_cuoi/mau_kiem_chung"]
THU_MUC_MAU_KHAC = ["ket_qua_sq2_bo_sung", "ket_qua_kiem_chung_bo_sung"]  # chỉ file mau_*.csv
KS_BIET_SAU = ["nhanh_b", "nhanh_khac", "nhieu_gr_ir"]
MOT_NGAY = sq3.MOT_NGAY
vn, pct, so, so_p = sq3.vn, sq3.pct, sq3.so, sq3.so_p
TEN = {**sq3.TEN, "R6_cp": "R6 cùng phút", "R6_ktd": "R6 khác thời điểm"}


def phan_vi(s, k=1):
    s = pd.Series(s).dropna()
    return f"{so(s.median(), k)} [{so(s.quantile(0.25), k)}–{so(s.quantile(0.75), k)}]" if len(s) else "–"


# ---------------------------------------------------------------- Mục 2. mẫu R6 và phép đếm R3
def tap_da_doc(goc=Path("."), p=print):
    """Case đã đọc tay: danh sách cứng của G3, mọi file trong các thư mục mau_kiem_chung và các file mau_*.csv (mẫu và nhận
    xét đọc tay). Không lấy file liệt kê dòng hàng bị đánh dấu (ví dụ r6_hai_nhom.csv)."""
    da = set(g3.DA_DOC)
    nguon = [(goc / th, "*.csv") for th in THU_MUC_MAU_DOC] + [(goc / th, "mau_*.csv") for th in THU_MUC_MAU_KHAC]
    so_file = 0
    for th, mau in nguon:
        for f in sorted(Path(th).glob(mau)):
            try:
                da |= set(pd.read_csv(f, encoding="utf-8-sig", usecols=["case"])["case"].astype(str))
                so_file += 1
            except Exception:
                pass
    p(f"  Tập đã đọc: {vn(len(da))} case từ {so_file} file mẫu và file nhận xét (cùng {len(g3.DA_DOC)} case ghi trong G3).")
    return da


def rut_r6(nhom, vendor, da, rng, n=10):
    """nhom: Series (case → 'cùng phút' / 'khác thời điểm'). Rút n case 'khác thời điểm' chưa đọc của hai nhà cung cấp
    và n case của các nhà cung cấp khác. Trả về (chọn, kích thước hai khung)."""
    ktd = sorted(set(nhom.index[nhom == "khác thời điểm"].astype(str)) - set(da))
    v = vendor.reindex(ktd).astype(str)
    khung = {"hai_ncc": sorted(v.index[v.isin(HAI_NCC_R6)]), "ncc_khac": sorted(v.index[~v.isin(HAI_NCC_R6)])}
    chon = {}
    for k in ["hai_ncc", "ncc_khac"]:
        ds = khung[k]
        chon[k] = [str(c) for c in rng.choice(ds, size=min(n, len(ds)), replace=False)] if ds else []
    return chon, {k: len(v_) for k, v_ in khung.items()}


def mau_r6(ev, d, vendor, ts_dp, ra, p):
    p("\n=== Mục 2. Mẫu bổ sung R6 khác thời điểm ===")
    e = ev.copy()
    e["vi_tri"] = e.groupby("case").cumcount() + 1
    r6 = set(d.index[d["co_R6"]])
    m = sq2b.r6_hai_nhom(e, r6)
    da = tap_da_doc(p=p)
    ktd = m.index[m["nhom"] == "khác thời điểm"].astype(str)
    p(f"  R6: {vn(len(m))} dòng; khác thời điểm {vn(len(ktd))}, trong đó đã đọc {vn(len(set(ktd) & da))}.")
    chon, khung = rut_r6(m["nhom"], vendor, da, np.random.default_rng(SEED))
    p(f"  Khung mẫu (chưa đọc): hai nhà cung cấp {vn(khung['hai_ncc'])}, nhà cung cấp khác {vn(khung['ncc_khac'])}; "
      f"rút {len(chon['hai_ncc'])} + {len(chon['ncc_khac'])} case (seed {SEED}).")
    tat_ca = chon["hai_ncc"] + chon["ncc_khac"]
    cot = [c for c in ["case", "ts", "act", "user", "vendor", "spend_area", "networth"] if c in ev.columns]
    vet = ev.loc[ev["case"].isin(tat_ca), cot].copy()
    vet.insert(2, "ts_dia_phuong", ts_dp.reindex(vet.index))
    vet.to_csv(ra / "mau_r6_khac_thoi_diem_g8.csv", index=False, encoding="utf-8-sig")
    nx = m.reindex(tat_ca)[["user", "ts_po", "ts_gr", "kc"]].copy()
    nx.insert(0, "nhom_ncc", ["hai nhà cung cấp"] * len(chon["hai_ncc"]) + ["nhà cung cấp khác"] * len(chon["ncc_khac"]))
    nx.insert(1, "vendor", vendor.reindex(tat_ca).values)
    nx["co_che"], nx["ghi_chu"] = "", ""
    nx.index.name = "case"
    nx.to_csv(ra / "mau_r6_khac_thoi_diem_g8_nhan_xet.csv", encoding="utf-8-sig")
    p("  Ghi mau_r6_khac_thoi_diem_g8.csv (vết) và mau_r6_khac_thoi_diem_g8_nhan_xet.csv (điền cột co_che, ghi_chu).")
    return m["nhom"]


def dem_r3(ev, d, p):
    p("\n=== Mục 2. Phép đếm thêm cho R3 ===")
    r3 = set(d.index[d["co_R3"]])
    ir1 = kc.dau(ev, IR)
    nguoi_po = ev[ev["act"] == PO].groupby("case")["user"].first()
    cp = ev.loc[ev["act"] == kc.CHANGE_PRICE, ["case", "ts", "user"]].copy()
    cp["sau_hd"] = cp["ts"] > cp["case"].map(ir1)
    cp["la_nguoi_tao"] = cp["user"].astype(str).values == cp["case"].map(nguoi_po).astype(str).values
    ca = ev.loc[ev["act"] == CHANGE_APPROVAL, ["case", "ts", "user"]]
    rows = []
    for ten, chon in [("R3: sửa giá sau hóa đơn", cp["sau_hd"] & cp["case"].isin(r3)),
                      ("Đối chứng: dòng không R3 có sửa giá (trước hóa đơn hoặc không có hóa đơn)", ~cp["sau_hd"] & ~cp["case"].isin(r3))]:
        c = cp[chon]
        cases = sorted(set(c["case"]))
        n = len(cases)
        dau = c.sort_values("ts").groupby("case").first()
        tao = c.groupby("case")["la_nguoi_tao"].any()
        moi_lan_tao = c.groupby("case")["la_nguoi_tao"].all()
        a = ca[ca["case"].isin(cases)].reset_index(drop=True).merge(
            dau[["ts", "user"]].rename(columns={"ts": "ts_cp", "user": "user_cp"}).rename_axis("case").reset_index(), on="case")
        a_sau = a[a["ts"] >= a["ts_cp"]]
        co_pd = a_sau.groupby("case").size().reindex(cases).fillna(0) > 0
        pd_khac_nguoi = a_sau[a_sau["user"].astype(str) != a_sau["user_cp"].astype(str)].groupby("case").size()
        co_pd_khac = pd_khac_nguoi.reindex(cases).fillna(0) > 0
        r = {"nhom": ten, "so_dong": n, "nguoi_sua_la_nguoi_tao_it_nhat_mot_lan": int(tao.sum()),
             "moi_lan_sua_deu_do_nguoi_tao": int(moi_lan_tao.sum()), "co_change_approval_sau_lan_sua_dau": int(co_pd.sum()),
             "change_approval_do_nguoi_khac": int(co_pd_khac.sum()),
             "nguoi_tao_sua_va_khong_phe_duyet": int((tao & ~co_pd).sum())}
        rows.append(r)
        p(f"  {ten}: {vn(n)} dòng hàng")
        p(f"    người sửa giá là người tạo dòng đơn hàng (ít nhất một lần): {pct(r['nguoi_sua_la_nguoi_tao_it_nhat_mot_lan'], n, 1)}; "
          f"mọi lần sửa đều do người tạo: {pct(r['moi_lan_sua_deu_do_nguoi_tao'], n, 1)}")
        p(f"    có {CHANGE_APPROVAL} từ lần sửa giá đầu tiên trở đi: {pct(r['co_change_approval_sau_lan_sua_dau'], n, 1)}; "
          f"trong đó do người khác người sửa giá: {pct(r['change_approval_do_nguoi_khac'], n, 1)}")
        p(f"    người tạo tự sửa giá và không có phê duyệt sau đó: {pct(r['nguoi_tao_sua_va_khong_phe_duyet'], n, 1)}")
    nguoi_tao_r3 = nguoi_po.reindex(sorted(r3))
    loai_tao = g3.la_batch(nguoi_tao_r3.astype(str))
    p(f"  Người tạo dòng đơn hàng R3 là tài khoản batch: {pct(int(loai_tao.sum()), len(r3), 1)}.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- công cụ mô hình
def chuan_bi_tap(xs, ks):
    """Đánh lại id cho tập con và căn chỉnh biến kiểm soát theo id (như sq3.buoc_do_nhay)."""
    xs = xs.copy()
    xs["id"] = np.arange(len(xs))
    k2 = ks.loc[xs.index].set_index(xs["id"].values)
    return xs.set_index("id", drop=False), k2


def mo_hinh(xs, ks, ten, luat_tv=sq3.LUAT_TV, luat_cd=sq3.LUAT_CO_DINH, tang=None):
    """Cox có biến thay đổi theo thời gian; tang: Series (theo index của xs) để phân tầng, None = không phân tầng."""
    from lifelines import CoxTimeVaryingFitter
    xi, k2 = chuan_bi_tap(xs, ks)
    if tang is None:
        _, h, n_dong, bo, _ = sq3.mo_hinh_tv(xi, k2, ten, luat_tv, luat_cd)
        return h, n_dong, bo
    xx = xi[["id", "T", "E"] + ["S_" + l for l in luat_tv]].join(k2)
    for l in luat_cd:
        xx[l] = xi["co_" + l].astype(float)
    xx["tang"] = pd.Series(pd.factorize(tang.reindex(xs.index).astype(str))[0], index=xi.index)
    dl = sq3.dang_dai(xx.reset_index(drop=True), luat_tv)
    cot = list(luat_tv) + list(luat_cd) + list(k2.columns)
    # bỏ biến hằng trong mọi tầng (ví dụ quy mô nhà cung cấp khi mỗi tầng là một nhà cung cấp)
    hang = [c for c in cot if dl.groupby("tang")[c].nunique().max() > 1]
    for pen in (0.0, 1e-5):
        try:
            m = CoxTimeVaryingFitter(penalizer=pen)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(dl[["id", "start", "stop", "event", "tang"] + hang], id_col="id", event_col="event",
                      start_col="start", stop_col="stop", strata=["tang"], show_progress=False)
            break
        except Exception as e:
            loi = e
    else:
        raise loi
    return sq3.bang_hr(m, ten), len(dl), [c for c in cot if c not in hang]


def in_mo_hinh(h, p, luat=sq3.LUAT_SQ3):
    sq3.in_hr(h, p, luat, TEN)


def rmst_luat(xs, cot, p, ten, h=sq3.RMST_NGAY, nhom_so_sanh=None):
    """RMST dòng có dấu hiệu (cot) so với nhóm so sánh (mặc định: dòng không có dấu hiệu)."""
    co = xs[cot].astype(bool)
    so_sanh = ~co if nhom_so_sanh is None else nhom_so_sanh & ~co
    m = co | so_sanh
    if not (co & m).any() or not (so_sanh & m).any():
        p(f"  {ten}: một nhóm không có dòng nào, bỏ qua.")
        return {"phan_tich": ten, "n_co": int(co.sum()), "n_khong": int(so_sanh.sum())}
    r = sq3.so_sanh_hai_nhom(xs.loc[m, "T"], xs.loc[m, "E"], co[m], h)
    p(f"  {ten}: có {vn(r['n_co'])}, so sánh {vn(r['n_khong'])}; trung vị KM {so(r['trung_vi_co'], 0)} so với "
      f"{so(r['trung_vi_khong'], 0)} ngày; log-rank p {so_p(r['log_rank_p'])}; RMST đến {h} ngày {so(r['rmst_co'])} so với "
      f"{so(r['rmst_khong'])}, chênh {so(r['rmst_chenh'])} (KTC 95%: {so(r['rmst_duoi'])}–{so(r['rmst_tren'])}); chênh > 0 là chậm hơn")
    return {"phan_tich": ten, **r}


# ---------------------------------------------------------------- Mục 3.1. R6 và nhà cung cấp
def tang_ncc(vendor, nguong=NGUONG_TANG):
    dem = vendor.map(vendor.value_counts())
    return vendor.where(dem >= nguong, "(nhà cung cấp nhỏ)")


def kiem_tra_r6(x, ks_all, vendor, nhom_r6, ra, p, luu):
    p("\n=== Mục 3.1. R6 và nhà cung cấp ===")
    hr, rm = [], []
    v = vendor.reindex(x.index).astype(str)
    tang = tang_ncc(v)
    co6 = x["co_R6"]
    t6 = tang[co6]
    ca_hai = pd.crosstab(tang, co6)
    ca_hai = ca_hai[(ca_hai.get(True, 0) > 0) & (ca_hai.get(False, 0) > 0)]
    p(f"  (1) Cox phân tầng theo nhà cung cấp: {vn(tang.nunique())} tầng ({vn(int((tang != '(nhà cung cấp nhỏ)').groupby(tang).any().sum()))} "
      f"nhà cung cấp có từ {NGUONG_TANG} dòng hàng, còn lại gộp một tầng gồm {vn(int((tang == '(nhà cung cấp nhỏ)').sum()))} dòng). "
      f"Dòng R6 nằm trong tầng có cả dòng không R6: {pct(int(t6.isin(ca_hai.index).sum()), int(co6.sum()), 1)}; "
      f"trong tầng nhà cung cấp nhỏ: {vn(int((t6 == '(nhà cung cấp nhỏ)').sum()))}.")
    t = time.time()
    h, n_dong, bo = mo_hinh(x, ks_all, "phan_tang_ncc", tang=tang)
    p(f"  {vn(n_dong)} đoạn; chạy {vn(round(time.time() - t))} giây" + (f"; bỏ biến hằng trong tầng {bo}" if bo else "") + ".")
    in_mo_hinh(h, p)
    hr.append(h)
    luu()

    bo2 = v.isin(HAI_NCC_R6)
    x2 = x[~bo2]
    p(f"\n  (2) Loại {vn(int(bo2.sum()))} dòng hàng của {', '.join(HAI_NCC_R6)} ({vn(int((bo2 & co6).sum()))} dòng R6): "
      f"{vn(len(x2))} dòng, R6 {vn(int(x2['co_R6'].sum()))}.")
    h, _, _ = mo_hinh(x2, ks_all, "loai_hai_ncc_r6")
    in_mo_hinh(h, p)
    hr.append(h)
    rm.append(rmst_luat(x2, "co_R6", p, "RMST R6, đã loại hai nhà cung cấp"))
    rm.append(rmst_luat(x[bo2], "co_R6", p, "RMST R6, chỉ trong hai nhà cung cấp"))
    luu()

    x3 = x.copy()
    nh = nhom_r6.reindex(x3.index)
    x3["co_R6_cp"] = x3["co_R6"] & (nh == "cùng phút")
    x3["co_R6_ktd"] = x3["co_R6"] & (nh == "khác thời điểm")
    p(f"\n  (3) Tách R6: cùng phút {vn(int(x3['co_R6_cp'].sum()))}, khác thời điểm {vn(int(x3['co_R6_ktd'].sum()))} dòng "
      f"(trong tập phân tích).")
    h, _, _ = mo_hinh(x3, ks_all, "tach_r6", luat_cd=["R6_cp", "R6_ktd", "R7b"])
    in_mo_hinh(h, p, sq3.LUAT_TV + ["R6_cp", "R6_ktd", "R7b"])
    hr.append(h)
    khong6 = ~x3["co_R6"]
    rm.append(rmst_luat(x3, "co_R6_cp", p, "RMST R6 cùng phút so với dòng không R6", nhom_so_sanh=khong6))
    rm.append(rmst_luat(x3, "co_R6_ktd", p, "RMST R6 khác thời điểm so với dòng không R6", nhom_so_sanh=khong6))
    # khác thời điểm, tách theo hai nhà cung cấp
    x3["co_R6_ktd_hai"] = x3["co_R6_ktd"] & bo2
    x3["co_R6_ktd_khac"] = x3["co_R6_ktd"] & ~bo2
    rm.append(rmst_luat(x3, "co_R6_ktd_hai", p, "RMST R6 khác thời điểm của hai nhà cung cấp, so với dòng không R6 cùng hai nhà cung cấp",
                        nhom_so_sanh=khong6 & bo2))
    rm.append(rmst_luat(x3, "co_R6_ktd_khac", p, "RMST R6 khác thời điểm của nhà cung cấp khác, so với dòng không R6 ngoài hai nhà cung cấp",
                        nhom_so_sanh=khong6 & ~bo2))
    luu()
    return pd.concat(hr, ignore_index=True), pd.DataFrame(rm)


# ---------------------------------------------------------------- Mục 3.2. debit memo và làm lại
def so_ngay(a, b):
    """Số ngày lịch (theo ngày địa phương) từ a đến b; dương là b sau a."""
    return (b.dt.normalize() - a.dt.normalize()) / MOT_NGAY


def nhom_vi_tri(k):
    return pd.Series(np.select([k < 0, k == 0, k > 0], ["trước", "cùng ngày", "sau"], ""), index=k.index).where(k.notna(), "không có")


def nhom_k(k, m):
    return pd.Series(np.select([k == m, k == 0, k < 0, (k > 0) & (k < m), k > m],
                               ["k = số debit memo", "k = 0", "k < 0", "0 < k < số debit memo", "k > số debit memo"], ""),
                     index=k.index)


def kiem_tra_memo(ev, pv, d, ts_dp, moc, ra, p):
    p("\n=== Mục 3.2. Debit memo và chỉ báo làm lại ===")
    lam_lai = sorted(d.index[d["co_lam_lai"]])
    e = ev.assign(ts_dp=ts_dp)
    huy = e[(e["act"] == CANCEL_IR) & ~g3.la_batch(e["user"])]
    c0 = huy.groupby("case")["ts_dp"].min()
    ir = e.loc[(e["act"] == IR) & e["case"].isin(c0.index), ["case", "ts_dp"]]
    ir_lai = ir[ir["ts_dp"] > ir["case"].map(c0)].groupby("case")["ts_dp"].min()
    ir_dau = e[e["act"] == IR].groupby("case")["ts_dp"].min()
    memo = e[e["act"] == MEMO]
    memo1 = memo.groupby("case")["ts_dp"].min()
    n_memo = memo.groupby("case").size()
    b = pd.DataFrame(index=pd.Index(lam_lai, name="case"))
    b["huy"], b["ir_lai"], b["ir_dau"], b["memo1"] = c0.reindex(lam_lai), ir_lai.reindex(lam_lai), ir_dau.reindex(lam_lai), memo1.reindex(lam_lai)
    b["so_memo"] = n_memo.reindex(lam_lai).fillna(0).astype(int)
    # debit memo gần lần hủy nhất (theo số ngày tuyệt đối)
    mm = memo[memo["case"].isin(lam_lai)][["case", "ts_dp"]].reset_index(drop=True).merge(b[["huy"]].reset_index(), on="case")
    mm["kc"] = so_ngay(mm["huy"], mm["ts_dp"])
    gan = mm.loc[mm["kc"].abs().groupby(mm["case"]).idxmin()].set_index("case")["ts_dp"] if len(mm) else pd.Series(dtype="datetime64[ns]")
    b["memo_gan_huy"] = gan.reindex(lam_lai)
    b["memo_so_voi_huy"] = so_ngay(b["huy"], b["memo_gan_huy"])
    b["memo_so_voi_ir_lai"] = so_ngay(b["ir_lai"], b["memo_gan_huy"])
    b["memo_so_voi_ir_dau"] = so_ngay(b["ir_dau"], b["memo_gan_huy"])
    b["huy_den_ir_lai"] = so_ngay(b["huy"], b["ir_lai"])
    p(f"  {vn(len(b))} dòng làm lại; có debit memo {pct(int((b['so_memo'] > 0).sum()), len(b), 1)}; số debit memo: "
      + "; ".join(f"{k} {vn(v)}" for k, v in b["so_memo"].clip(upper=3).value_counts().sort_index().items()) + " (3 = từ 3 trở lên).")
    p("  Ngày (giờ địa phương) của debit memo gần lần hủy nhất, so với (số ngày lịch, dương là debit memo xảy ra sau):")
    for cot, ten in [("memo_so_voi_huy", "lần hủy hóa đơn đầu tiên do người dùng thật"),
                     ("memo_so_voi_ir_lai", "lần ghi lại hóa đơn"), ("memo_so_voi_ir_dau", "lần ghi nhận hóa đơn đầu tiên")]:
        g = nhom_vi_tri(b[cot])
        p(f"    {ten}: trước {pct(int((g == 'trước').sum()), len(b), 1)}, cùng ngày {pct(int((g == 'cùng ngày').sum()), len(b), 1)}, "
          f"sau {pct(int((g == 'sau').sum()), len(b), 1)}; khoảng cách {phan_vi(b[cot], 0)} ngày")
    p(f"  Từ lần hủy đến lần ghi lại hóa đơn: {phan_vi(b['huy_den_ir_lai'], 0)} ngày; cùng ngày "
      f"{pct(int((b['huy_den_ir_lai'] == 0).sum()), len(b), 1)}.")
    b.to_csv(ra / "lam_lai_thoi_diem_memo.csv", encoding="utf-8-sig")

    # bảng chéo k như G4: dòng có debit memo, không hoạt động trong 90 ngày trước thời điểm trích xuất
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    s = g5.so_lieu_case(ev, pv, loai)
    nghi = (pd.Timestamp(moc) - ev.groupby("case")["ts"].max().reindex(s.index)).dt.days > g4.NGAY_NGHI
    rows = []
    for ten_tap, chon in [(f"G4: có debit memo, không hoạt động > {g4.NGAY_NGHI} ngày", (s["n_memo"] >= 1) & nghi),
                          ("mọi dòng có debit memo", s["n_memo"] >= 1)]:
        xk = s[chon].copy()
        xk["k"] = xk["n_clear_tt"] - (xk["n_ir"] - xk["n_cancel"]).clip(lower=0)
        xk["nhom_k"] = nhom_k(xk["k"], xk["n_memo"])
        xk["lam_lai"] = np.where(xk.index.isin(lam_lai), "làm lại", "không làm lại")
        bang = pd.crosstab(xk["nhom_k"], xk["lam_lai"], margins=True, margins_name="tổng")
        p(f"\n  Bảng chéo k = số Clear thanh toán − max(0, #IR − #Cancel IR), {ten_tap} (n = {vn(len(xk))}):")
        ty_le = pd.crosstab(xk["nhom_k"], xk["lam_lai"], normalize="columns")
        for dong in bang.to_string().splitlines():
            p("    " + dong)
        p("    Tỷ lệ theo cột: " + "; ".join(f"{c}: " + ", ".join(f"{i} {vn(v, '.1%')}" for i, v in ty_le[c].items() if v > 0)
                                          for c in ty_le.columns))
        bang.assign(tap=ten_tap).to_csv(ra / f"bang_k_lam_lai_{len(rows)}.csv", encoding="utf-8-sig")
        rows.append(ten_tap)

    # 10 dòng làm lại để đọc
    rng = np.random.default_rng(SEED)
    chon = [str(c) for c in rng.choice(lam_lai, size=min(10, len(lam_lai)), replace=False)] if lam_lai else []
    cot = [c for c in ["case", "ts", "act", "user", "vendor", "networth"] if c in ev.columns]
    vet = ev.loc[ev["case"].isin(chon), cot].copy()
    vet.insert(2, "ts_dia_phuong", ts_dp.reindex(vet.index))
    vet.to_csv(ra / "mau_lam_lai_g8.csv", index=False, encoding="utf-8-sig")
    nx = b.reindex(chon)[["huy", "ir_lai", "memo_gan_huy", "so_memo"]].copy()
    nx["trinh_tu"], nx["ghi_chu"] = "", ""
    nx.to_csv(ra / "mau_lam_lai_g8_nhan_xet.csv", encoding="utf-8-sig")
    p(f"\n  Rút {len(chon)} dòng làm lại để đọc (seed {SEED}): mau_lam_lai_g8.csv và mau_lam_lai_g8_nhan_xet.csv.")
    return b


# ---------------------------------------------------------------- Mục 3.3. R7 biến thể phụ với mốc chung
def doi_moc(x, goc, moc):
    """Tính lại t, T, s_, S_ khi mốc thời gian là goc (Timestamp theo dòng hàng, ≤ ngày tạo dòng đơn hàng)."""
    x = x.copy()
    lech = ((x["dau_po"] - goc.reindex(x.index)) / MOT_NGAY).fillna(0.0).clip(lower=0.0)
    x["lech_moc"] = lech
    x["t"] = x["t"] + lech
    x["T"] = np.ceil(x["t"]).clip(lower=1)
    for l in sq3.LUAT_TV + ["R6"]:
        x["s_" + l] = x["s_" + l] + lech
        x["S_" + l] = sq3.ngay_chia(x["s_" + l], x["t"])
    return x


def kiem_tra_r7(ev, x, ks_all, vendor, moc, ra, p, luu):
    p("\n=== Mục 3.3. R7 biến thể phụ với mốc chung ===")
    vi = kc.dau(ev, kc.VENDOR_INV).reindex(x.index)
    goc = pd.concat([x["dau_po"], vi], axis=1).min(axis=1)
    x7 = doi_moc(x, goc, moc)
    co = x7["co_R7b"]
    p(f"  Mốc = thời điểm sớm hơn giữa tạo dòng đơn hàng và Vendor creates invoice đầu tiên (sự kiện chỉ ghi đến ngày). "
      f"Không có ngày hóa đơn nhà cung cấp: {vn(int(vi.isna().sum()))} dòng, dùng ngày tạo dòng đơn hàng.")
    p(f"  Mốc lùi sớm hơn ngày tạo dòng đơn hàng: {pct(int((x7['lech_moc'] > 0).sum()), len(x7), 1)} dòng; "
      f"số ngày lùi ở dòng R7b {phan_vi(x7.loc[co, 'lech_moc'])}, ở dòng không R7b có lùi {phan_vi(x7.loc[~co & (x7['lech_moc'] > 0), 'lech_moc'])}.")
    hr, rm = [], []
    h, _, _ = mo_hinh(x7, ks_all, "r7_moc_chung")
    in_mo_hinh(h, p)
    hr.append(h)
    rm.append(rmst_luat(x7, "co_R7b", p, "RMST R7b, mốc chung"))
    luu()
    v = vendor.reindex(x7.index).astype(str)
    bo = v == NCC_R7
    x8 = x7[~bo]
    p(f"\n  Loại {NCC_R7}: {vn(int(bo.sum()))} dòng ({vn(int((bo & co).sum()))} dòng R7b); còn R7b {vn(int(x8['co_R7b'].sum()))}.")
    h, _, _ = mo_hinh(x8, ks_all, "r7_moc_chung_loai_0550")
    in_mo_hinh(h, p)
    hr.append(h)
    rm.append(rmst_luat(x8, "co_R7b", p, f"RMST R7b, mốc chung, loại {NCC_R7}"))
    luu()
    return pd.concat(hr, ignore_index=True), pd.DataFrame(rm)


# ---------------------------------------------------------------- Mục 3.4. biến kiểm soát biết lúc tạo
def kiem_tra_bien_ban_dau(x, ks_all, p):
    p("\n=== Mục 3.4. Mô hình chỉ dùng biến kiểm soát biết được lúc tạo dòng đơn hàng ===")
    ks = ks_all[[c for c in ks_all.columns if c not in KS_BIET_SAU]]
    p(f"  Bỏ {', '.join(KS_BIET_SAU)}; giữ hạng mục chi tiêu, log giá trị, giá trị 0, quy mô nhà cung cấp.")
    h, _, _ = mo_hinh(x, ks, "chi_bien_luc_tao")
    in_mo_hinh(h, p)
    return h


# ---------------------------------------------------------------- Hình 4.4
def ve_minh_hoa_hai_hang(x, ra, p):
    plt = sq3.nap_plt()
    if plt is None:
        p("(Không có matplotlib, bỏ qua hình.)")
        return
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    o = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1]]
    for i, (ax, l) in enumerate(zip(o, sq3.LUAT_SQ3)):
        m = x["co_" + l]
        sq3.ve_duong(ax, x.loc[~m, "T"], x.loc[~m, "E"], f"không (n = {vn(int((~m).sum()))})", sq3.MAU[0])
        sq3.ve_duong(ax, x.loc[m, "T"], x.loc[m, "E"], f"có (n = {vn(int(m.sum()))})", sq3.MAU[1])
        sq3.trang_tri(ax, sq3.TEN[l] + (" (minh họa, nhóm cuối)" if l in sq3.LUAT_TV else ""))
        if i not in (0, 3):  # nhãn và số trên trục tung chỉ ở ô đầu mỗi hàng
            ax.set_ylabel("")
            ax.set_yticklabels([])
    axs[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(ra / "hinh_4_4_hai_hang.png", dpi=150)
    plt.close(fig)
    p("\nĐã vẽ hinh_4_4_hai_hang.png (Hình 4.4, hai hàng).")


# ---------------------------------------------------------------- chạy
def phan_tich(ev, pv, moc, ra, p, luu=lambda: None):
    t0 = time.time()
    d, _, top = sq3.chuan_bi(ev, pv, moc, p)
    x, _ = sq3.tap_phan_tich(d, "tt_moi", moc)
    ks_all, _ = sq3.bien_kiem_soat(d, top)
    vendor = ev.drop_duplicates("case").set_index("case")["vendor"].astype(str) if "vendor" in ev else pd.Series("?", index=d.index)
    ts_dp = g4.gio_dia_phuong(ev, lambda *a: None)
    p(f"Tập phân tích: {vn(len(x))} dòng hàng (hoàn tất {vn(int(x['E'].sum()))}).")
    nhom_r6 = mau_r6(ev, d, vendor, ts_dp, ra, p)
    dem_r3(ev, d, p).to_csv(ra / "r3_nguoi_sua_va_phe_duyet.csv", index=False, encoding="utf-8-sig")
    luu()
    hr1, rm1 = kiem_tra_r6(x, ks_all, vendor, nhom_r6, ra, p, luu)
    kiem_tra_memo(ev, pv, d, ts_dp, moc, ra, p)
    luu()
    hr3, rm3 = kiem_tra_r7(ev, x, ks_all, vendor, moc, ra, p, luu)
    hr4 = kiem_tra_bien_ban_dau(x, ks_all, p)
    pd.concat([hr1, hr3, hr4], ignore_index=True).to_csv(ra / "hr_g8.csv", index=False, encoding="utf-8-sig")
    pd.concat([rm1, rm3], ignore_index=True).to_csv(ra / "rmst_g8.csv", index=False, encoding="utf-8-sig")
    ve_minh_hoa_hai_hang(x, ra, p)
    p(f"\nXong sau {vn(round(time.time() - t0))} giây.")


# ---------------------------------------------------------------- tự kiểm
def tu_kiem():
    loi = 0

    def kt(ten, dat, ct=""):
        nonlocal loi
        loi += not dat
        print(f"{ten:72s} {'ĐẠT' if dat else 'SAI ' + str(ct)}")

    # tập đã đọc: chỉ file mẫu và nhận xét
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        goc = Path(tmp)
        (goc / "ket_qua_kiem_chung_cuoi" / "mau_kiem_chung").mkdir(parents=True)
        (goc / "ket_qua_sq2_bo_sung").mkdir()
        pd.DataFrame({"case": ["a", "a", "b"]}).to_csv(goc / "ket_qua_kiem_chung_cuoi/mau_kiem_chung/R6_bi_danh_dau.csv", index=False)
        pd.DataFrame({"case": ["c"]}).to_csv(goc / "ket_qua_sq2_bo_sung/mau_r1.csv", index=False)
        pd.DataFrame({"case": [f"r{i}" for i in range(600)]}).to_csv(goc / "ket_qua_sq2_bo_sung/r6_hai_nhom.csv", index=False)
        da = tap_da_doc(goc, p=lambda *a: None)
        kt("Tập đã đọc: lấy file mẫu, bỏ file liệt kê mọi dòng R6", da == set(g3.DA_DOC) | {"a", "b", "c"}, len(da))
    # rút mẫu R6 10 + 10, không lấy case đã đọc
    nhom = pd.Series(["khác thời điểm"] * 60 + ["cùng phút"] * 10, index=[f"c{i:02d}" for i in range(70)])
    vend = pd.Series(["vendorID_0741"] * 15 + ["vendorID_0572"] * 5 + ["v9"] * 50, index=nhom.index)
    chon, khung = rut_r6(nhom, vend, {"c00", "c30"}, np.random.default_rng(SEED))
    kt("Mẫu R6: 10 của hai nhà cung cấp, 10 của nhà cung cấp khác, bỏ case đã đọc",
       len(chon["hai_ncc"]) == 10 and len(chon["ncc_khac"]) == 10 and khung == {"hai_ncc": 19, "ncc_khac": 39}
       and not ({"c00", "c30"} & set(chon["hai_ncc"] + chon["ncc_khac"]))
       and all(vend[c] in HAI_NCC_R6 for c in chon["hai_ncc"]) and all(nhom[c] == "khác thời điểm" for c in chon["ncc_khac"]), khung)
    # tầng nhà cung cấp
    v = pd.Series(["A"] * 60 + ["B"] * 49 + ["C"] * 1)
    tg = tang_ncc(v)
    kt("Tầng: nhà cung cấp ≥ 50 dòng riêng, còn lại gộp", list(tg.unique()) == ["A", "(nhà cung cấp nhỏ)"], tg.unique())
    # vị trí debit memo và nhóm k
    a = pd.Series(pd.to_datetime(["2018-01-05 10:00", "2018-01-05 10:00", "2018-01-05 23:00", None]))
    b = pd.Series(pd.to_datetime(["2018-01-04 23:59", "2018-01-05 23:59", "2018-01-07 00:10", "2018-01-01 00:00"]))
    kc_ = so_ngay(a, b)
    kt("Số ngày lịch giữa hai sự kiện", list(kc_.iloc[:3]) == [-1.0, 0.0, 2.0] and np.isnan(kc_.iloc[3]), list(kc_))
    kt("Nhóm vị trí: trước / cùng ngày / sau / không có", list(nhom_vi_tri(kc_)) == ["trước", "cùng ngày", "sau", "không có"])
    nk = nhom_k(pd.Series([1, 0, -1, 1, 3]), pd.Series([1, 2, 1, 2, 2]))
    kt("Nhóm k theo số debit memo", list(nk) == ["k = số debit memo", "k = 0", "k < 0", "0 < k < số debit memo", "k > số debit memo"],
       list(nk))
    # đổi mốc
    t0 = pd.Timestamp("2018-03-10")
    x = pd.DataFrame({"dau_po": [t0, t0], "t": [10.2, 4.0], "s_R2a": [3.5, np.nan], "s_R3": [np.nan] * 2,
                      "s_lam_lai": [np.nan] * 2, "s_R6": [8.0, np.nan]}, index=["p", "q"])
    goc = pd.Series([t0 - pd.Timedelta(days=5), pd.NaT], index=["p", "q"], dtype="datetime64[ns]")
    y = doi_moc(x, goc, None)
    kt("Đổi mốc: cộng số ngày lùi vào t và thời điểm dấu hiệu; không có ngày hóa đơn thì giữ nguyên",
       y.loc["p", "t"] == 15.2 and y.loc["p", "T"] == 16 and y.loc["p", "s_R2a"] == 8.5 and y.loc["p", "S_R2a"] == 8
       and y.loc["p", "S_R6"] == 13 and y.loc["q", "t"] == 4.0 and y.loc["q", "lech_moc"] == 0, y)

    import importlib.util
    if importlib.util.find_spec("lifelines") is None:
        print("Chưa có lifelines: python -m pip install lifelines")
        return loi + 1
    # Cox phân tầng: dấu hiệu tập trung ở tầng có nguy cơ nền thấp, không ảnh hưởng thật → phân tầng đưa HR về gần 1
    rng = np.random.default_rng(SEED)
    n = 8000
    tang = rng.integers(0, 2, n)
    co = rng.random(n) < np.where(tang == 1, 0.30, 0.03)
    T = np.ceil(rng.exponential(np.where(tang == 1, 120, 40))).clip(1)
    xs = pd.DataFrame({"T": T, "E": 1, "t": T, "co_R2a": False, "co_R3": False, "co_lam_lai": False, "co_R6": co,
                       "co_R7b": False, "S_R2a": np.inf, "S_R3": np.inf, "S_lam_lai": np.inf}, index=[f"i{i}" for i in range(n)])
    ks = pd.DataFrame({"z": rng.normal(size=n)}, index=xs.index)
    h0, _, _ = mo_hinh(xs, ks, "khong_phan_tang", luat_tv=[], luat_cd=["R6"])
    h1, _, _ = mo_hinh(xs, ks, "phan_tang", luat_tv=[], luat_cd=["R6"], tang=pd.Series(tang, index=xs.index))
    hr0 = float(h0.set_index("bien").loc["R6", "HR"])
    hr1 = float(h1.set_index("bien").loc["R6", "HR"])
    kt(f"Phân tầng: không phân tầng HR = {hr0:.2f} (< 0,8, lệch do tầng)", hr0 < 0.8, hr0)
    kt(f"Phân tầng: phân tầng HR = {hr1:.2f} (0,88–1,12)", 0.88 < hr1 < 1.12, hr1)
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
    ap.add_argument("--ra", default="ket_qua_g8")
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
        (ra / "tom_tat_g8.txt").write_text("\n".join(out), encoding="utf-8")

    t = time.time()
    raw = sq2.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    p(f"Đọc log xong sau {vn(round(time.time() - t))} giây.")
    phan_tich(ev, pv, moc, ra, p, luu)
    luu()
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_g8.txt.")


if __name__ == "__main__":
    main()
