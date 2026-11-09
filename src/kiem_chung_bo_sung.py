"""Kiểm tra bổ sung cho Chương 3 và Mục 4.1–4.2 (góp ý ngày 22/10/2026). Không sửa luật đã đóng băng.

  1. Mẫu kiểm chứng cuối: rút lại mẫu bằng đúng code, seed và khung mẫu của kiem_chung_cuoi.py rồi so với các file mẫu
     đã đọc tay; với R6, đối chiếu cờ "cùng phút" tính bằng code với nhận xét đọc tay và tính xác suất siêu bội của
     số case cùng phút trong mẫu. Rút thêm một mẫu 20 case R6 "khác thời điểm" chưa đọc để đọc bổ sung.
  2. Giá trị dòng hàng: căn cứ cho giả định ẩn danh bằng phép nhân, và hiệu ứng Mann-Whitney U / (n1·n2) theo từng luật
     (đúng với mọi biến đổi đơn điệu tăng), kèm kiểm định và hiệu chỉnh Holm.
  3. Cơ chế của R3: ai thực hiện Change Price sau hóa đơn, và Remove Payment Block có theo sau trong 1 phút, 1 giờ, 1 ngày.
  4. Bổ sung nhỏ: dòng hàng nhánh (b) không có Remove Payment Block đã được thanh toán chưa; R2a trên dòng hàng không có
     hủy; độ nhạy của Δ (60 phút, 1 ngày, mở rộng 1.000 phút cùng người); cận trên tỷ lệ bỏ sót khi 0/20.
  5. Alignment và so sánh IMf trên 100% biến thể (Bảng 4.3, 4.6, 4.15), cần pm4py. Bước này lâu nhất nên chạy sau cùng;
     bỏ qua bằng --bo-alignment.

Cách chạy (cùng thư mục với sq2_bo_sung.py, sq2_tuan_thu.py, sq1_imf.py và các file đi kèm; cần các thư mục kết quả
ket_qua_kiem_chung_cuoi, ket_qua_sq1, ket_qua_sq2_bo_sung để đối chiếu mẫu):
    python src/kiem_chung_bo_sung.py --tu-kiem
    python src/kiem_chung_bo_sung.py
Kết quả ghi vào ket_qua_kiem_chung_bo_sung/; gửi lại tom_tat_kiem_chung_bo_sung.txt.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import g3_kiem_tra_bo_sung as g3
import g4_kiem_tra as g4
import g5_kiem_tra as g5
import kiem_chung_cuoi as kc
import r5_v2_phan_loai_clear as r5
import sq1_kham_pha as sq1
import sq2_bo_sung as sq2b
import sq2_tuan_thu as sq2
from r5_v2_phan_loai_clear import CANCEL_GR, CANCEL_IR, IR, buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear, sap_xep
import cau_hinh

SEED = 2026
MOT_PHUT, MOT_GIO, MOT_NGAY = pd.Timedelta(minutes=1), pd.Timedelta(hours=1), pd.Timedelta(days=1)
THU_MUC_MAU = Path("ket_qua_kiem_chung_cuoi") / "mau_kiem_chung"
LUAT_GIA_TRI = ["R1", "R2a", "R3", "R5", "R6", "R7b", "nhan_chinh"]
vn = sq2b.vn


def pct(n, N, so_le=2):
    return f"{vn(int(n))} ({vn(n / N, f'.{so_le}%')})" if N else f"{vn(int(n))} (–)"


# ---------------------------------------------------------------- thống kê
def sieu_boi_tu_k(N, K, n, k):
    """P(X ≥ k) với X ~ siêu bội: rút n từ N phần tử, trong đó K phần tử có đặc điểm."""
    return sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(k, min(K, n) + 1)) / math.comb(N, n)


def clopper_pearson_tren(k, n, alpha=0.05):
    """Cận trên khoảng tin cậy Clopper-Pearson hai phía (1 − alpha) của tỷ lệ k/n."""
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1 - alpha / 2, k + 1, n - k))


def holm(p):
    """Giá trị p đã hiệu chỉnh theo Holm (1979), cùng thứ tự đầu vào."""
    p = np.asarray(p, dtype=float)
    thu_tu = np.argsort(p)
    m = len(p)
    hc = np.empty(m)
    cao_nhat = 0.0
    for i, j in enumerate(thu_tu):
        cao_nhat = max(cao_nhat, min(1.0, (m - i) * p[j]))
        hc[j] = cao_nhat
    return hc


def hieu_ung_mann_whitney(x, y):
    """A = U / (n1·n2) = P(X > Y) + 0,5·P(X = Y); KTC 95% theo Hanley-McNeil (1982); p hai phía (xấp xỉ chuẩn)."""
    from scipy.stats import mannwhitneyu
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    kq = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    a = float(kq.statistic) / (n1 * n2)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt(max(a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a), 0) / (n1 * n2))
    return {"n_co_dau_hieu": n1, "n_khong_dau_hieu": n2, "A": a, "ktc95_duoi": a - 1.96 * se, "ktc95_tren": a + 1.96 * se,
            "p": float(kq.pvalue)}


# ---------------------------------------------------------------- 1. mẫu kiểm chứng
def da_doc_truoc_kiem_chung_cuoi():
    """Tập case đã đọc trước kiểm chứng cuối, đúng như kiem_chung_cuoi.phan_tich tính."""
    da = g5.doc_case_da_doc()
    for th in kc.THU_MUC_DA_DOC[len(g5.THU_MUC_DA_DOC):]:
        for f in Path(th).glob("*.csv"):
            try:
                da |= set(pd.read_csv(f, encoding="utf-8-sig", usecols=["case"])["case"].astype(str))
            except Exception:
                pass
    return da


def rut_lai_mau(kq, da_doc, rng):
    """Lặp lại vòng rút mẫu của kiem_chung_cuoi.phan_tich (không ghi file): mỗi luật một mẫu bị đánh dấu rồi một mẫu
    không bị đánh dấu, dùng chung một rng. Trả về {(luật, nhóm): (khung, mẫu)}."""
    da = set(da_doc)
    ra = {}
    for luat, (bi, lien_quan, _) in kq.items():
        for nhan, tap in (("bi_danh_dau", bi), ("khong_bi_danh_dau", lien_quan - bi)):
            ds = sorted(set(tap) - da)
            chon = [str(c) for c in rng.choice(ds, size=min(kc.N_MAU, len(ds)), replace=False)] if ds else []
            ra[(luat, nhan)] = (len(ds), chon)
            da |= set(chon)
    return ra


def kiem_tra_mau(ev, kq, ra, p):
    p("\n=== 1. Mẫu kiểm chứng cuối ===")
    if not THU_MUC_MAU.exists():
        p(f"  Không thấy {THU_MUC_MAU}, bỏ qua phần đối chiếu mẫu.")
        return None
    mau = rut_lai_mau(kq, da_doc_truoc_kiem_chung_cuoi(), np.random.default_rng(kc.SEED))
    rows = []
    for (luat, nhan), (khung, chon) in mau.items():
        f = THU_MUC_MAU / f"{luat}_{nhan}.csv"
        try:
            thuc = set(pd.read_csv(f, encoding="utf-8-sig", usecols=["case"])["case"].astype(str))
        except Exception:
            thuc = set()
        rows.append({"luat": luat, "nhom": nhan, "khung_mau": khung, "so_case_rut_lai": len(chon),
                     "so_case_trong_file": len(thuc), "trung": len(set(chon) & thuc)})
    b = pd.DataFrame(rows)
    b.to_csv(ra / "doi_chieu_mau.csv", index=False, encoding="utf-8-sig")
    p(f"  Rút lại bằng seed {kc.SEED}, rng.choice không hoàn lại trên danh sách case đã sắp xếp, sau khi loại case đã đọc:")
    p(b.to_string(index=False))
    khop = bool((b["trung"] == b["so_case_trong_file"]).all() and (b["so_case_rut_lai"] == b["so_case_trong_file"]).all())
    p("  Mọi mẫu rút lại trùng hoàn toàn với file đã đọc tay." if khop else "  !!! Có mẫu không trùng, xem doi_chieu_mau.csv.")
    return khop


def mau_r6(ev, kq, ra, p, rng):
    r6 = kq["R6"][0]
    m = sq2b.r6_hai_nhom(ev, r6)
    m.to_csv(ra / "r6_hai_nhom.csv", encoding="utf-8-sig")
    N, K = len(m), int((m["nhom"] == "cùng phút").sum())
    p(f"\n  R6: {vn(N)} dòng hàng, cùng phút (|GR − PO| ≤ 1 phút, cùng người dùng thật, xét mọi cặp) {pct(K, N, 1)}.")
    f = THU_MUC_MAU / "R6_bi_danh_dau.csv"
    if f.exists():
        mau = sorted(set(pd.read_csv(f, encoding="utf-8-sig", usecols=["case"])["case"].astype(str)))
        x = m.reindex(mau)[["user", "ts_po", "ts_gr", "kc", "nhom"]]
        nx = THU_MUC_MAU / "kiem_chung_cuoi_nhan_xet.csv"
        if nx.exists():
            t = pd.read_csv(nx, encoding="utf-8-sig")
            t = t[(t["luat"] == "R6") & (t["nhom"] == "bi_danh_dau")].set_index("case")
            x["doc_tay"] = np.where(t["ghi_chu"].reindex(mau).astype(str).str.contains("1 phút"), "cùng phút", "khác thời điểm")
            x["ket_qua_doc_tay"] = t["ket_qua"].reindex(mau)
        x.to_csv(ra / "mau_r6_doi_chieu.csv", encoding="utf-8-sig")
        k = int((x["nhom"] == "cùng phút").sum())
        p(f"  Mẫu kiểm chứng R6: {k}/{len(mau)} case cùng phút theo code.")
        if "doc_tay" in x:
            p(f"  Theo nhận xét đọc tay: {int((x['doc_tay'] == 'cùng phút').sum())}/{len(mau)}; "
              f"khớp từng case giữa code và đọc tay: {int((x['doc_tay'] == x['nhom']).sum())}/{len(mau)}.")
            for nhom, g in x.groupby("nhom"):
                p(f"    {nhom}: đúng {int((g['ket_qua_doc_tay'] == 'đúng').sum())}/{len(g)}")
        p(f"  Nếu rút ngẫu nhiên 20 từ {vn(N)} dòng, xác suất có từ {k} case cùng phút trở lên: "
          f"{vn(sieu_boi_tu_k(N, K, len(mau), k), '.4%')}; kỳ vọng {vn(len(mau) * K / N, ',.2f')} case.")
    # mẫu bổ sung: 20 case "khác thời điểm" chưa đọc
    da = da_doc_truoc_kiem_chung_cuoi()
    # chỉ file mẫu đã đọc; r6_hai_nhom.csv trong ket_qua_sq2_bo_sung liệt kê mọi dòng R6, không phải case đã đọc
    for th, mau_ten in [(THU_MUC_MAU, "*.csv"), (Path("ket_qua_sq2_bo_sung"), "mau_*.csv")]:
        for f in Path(th).glob(mau_ten):
            try:
                da |= set(pd.read_csv(f, encoding="utf-8-sig", usecols=["case"])["case"].astype(str))
            except Exception:
                pass
    ds = sorted(set(m.index[m["nhom"] == "khác thời điểm"].astype(str)) - da)
    chon = [str(c) for c in rng.choice(ds, size=min(20, len(ds)), replace=False)] if ds else []
    vet = ev[ev["case"].isin(chon)][["case", "ts", "act", "user"]].copy()
    vet["ket_qua"], vet["nhan_xet"] = "", ""
    vet.to_csv(ra / "mau_r6_khac_thoi_diem.csv", index=False, encoding="utf-8-sig")
    p(f"  Mẫu bổ sung R6 khác thời điểm: {vn(len(ds))} case chưa đọc, lấy {len(chon)} (mau_r6_khac_thoi_diem.csv).")


# ---------------------------------------------------------------- 2. giá trị
def gia_tri(ev, d, ra, p):
    p("\n=== 2. Giá trị dòng hàng (đã ẩn danh) ===")
    v = sq2.gia_tri_dong_hang(ev, d.index)
    co = v.notna()
    p(f"  Dòng hàng có giá trị: {pct(int(co.sum()), len(d))}; bằng 0: {vn(int((v == 0).sum()))}; "
      f"âm: {vn(int((v < 0).sum()))}; nhỏ nhất {vn(float(v.min()), ',.2f')}; lớn nhất {vn(float(v.max()), ',.2f')}.")
    tat_ca = ev.loc[ev["networth"].notna(), "networth"] if "networth" in ev else pd.Series(dtype=float)
    p(f"  Trên mọi sự kiện có giá trị: {vn(len(tat_ca))} sự kiện; bằng 0: {vn(int((tat_ca == 0).sum()))}; âm: {vn(int((tat_ca < 0).sum()))}.")
    if not co.any():
        return
    rows = []
    for k in LUAT_GIA_TRI:
        mk = d[k] & co
        mkh = ~d[k] & co
        if mk.sum() < 2:
            continue
        r = {"luat": k, **hieu_ung_mann_whitney(v[mk], v[mkh]),
             "trung_vi_co": float(v[mk].median()), "trung_vi_khong": float(v[mkh].median())}
        rows.append(r)
    b = pd.DataFrame(rows)
    b["p_holm"] = holm(b["p"])
    b.to_csv(ra / "mann_whitney_gia_tri.csv", index=False, encoding="utf-8-sig")
    p("  A = P(giá trị dòng có dấu hiệu > dòng không có dấu hiệu) + 0,5·P(bằng nhau); A = 0,5 là không khác biệt.")
    hien = b[["luat", "n_co_dau_hieu", "A", "ktc95_duoi", "ktc95_tren", "p_holm", "trung_vi_co", "trung_vi_khong"]]
    p(hien.to_string(index=False, formatters={"A": "{:.4f}".format, "ktc95_duoi": "{:.4f}".format,
                                               "ktc95_tren": "{:.4f}".format, "p_holm": "{:.2e}".format}))


# ---------------------------------------------------------------- 3. cơ chế R3
def loai_nguoi(u):
    u = pd.Series(u).astype(str)
    return pd.Series(np.select([g3.la_batch(u), u == sq1.USER_002, u.str.upper().isin(["NONE", "NAN"])],
                               ["tài khoản batch", "user_002", "NONE"], "người dùng thật"), index=u.index)


def khoang_den_rpb_sau(cp, rpb):
    """Với mỗi Change Price: khoảng thời gian tới Remove Payment Block gần nhất xảy ra cùng lúc hoặc sau nó (cùng dòng hàng),
    kèm người gỡ khóa. cp, rpb: DataFrame có case, ts, user."""
    a = cp[["case", "ts", "user"]].sort_values("ts").reset_index(drop=True)
    b = rpb[["case", "ts", "user"]].rename(columns={"user": "user_rpb"}).assign(ts_rpb=lambda x: x["ts"]).sort_values("ts")
    m = pd.merge_asof(a, b, on="ts", by="case", direction="forward", allow_exact_matches=True)
    m["kc"] = m["ts_rpb"] - m["ts"]
    return m


def co_che_r3(ev, kq, ra, p):
    p("\n=== 3. Cơ chế của R3 ===")
    r3 = kq["R3"][0]
    ir1 = ev[ev["act"] == IR].groupby("case")["ts"].min()
    cp_all = ev[ev["act"] == kc.CHANGE_PRICE][["case", "ts", "user"]].copy()
    cp_all["ir1"] = cp_all["case"].map(ir1)
    cp = cp_all[cp_all["ir1"].notna() & (cp_all["ts"] > cp_all["ir1"])].copy()
    cp["nguoi"] = loai_nguoi(cp["user"]).values
    N = len(r3)
    p(f"  {vn(N)} dòng hàng R3; {vn(len(cp))} lần Change Price sau Record Invoice Receipt đầu tiên.")
    p("  Người thực hiện Change Price sau hóa đơn (theo lần sửa / theo dòng hàng có ít nhất một lần):")
    for nhom in ["người dùng thật", "tài khoản batch", "user_002", "NONE"]:
        p(f"    {nhom}: {pct(int((cp['nguoi'] == nhom).sum()), len(cp))} lần; "
          f"{pct(cp.loc[cp['nguoi'] == nhom, 'case'].nunique(), N)} dòng hàng")
    rpb = ev[ev["act"] == kc.RPB][["case", "ts", "user"]]
    m = khoang_den_rpb_sau(cp, rpb[rpb["case"].isin(r3)])
    m["nguoi_rpb"] = loai_nguoi(m["user_rpb"].fillna("NONE")).where(m["user_rpb"].notna(), "")
    gan = m.sort_values("kc").drop_duplicates("case").set_index("case").reindex(sorted(r3))
    co_rpb = rpb.loc[rpb["case"].isin(r3), "case"].nunique()
    p(f"  Dòng R3 có Remove Payment Block: {pct(co_rpb, N)}.")
    rows = []
    for ten, nguong in [("trong 1 phút", MOT_PHUT), ("trong 1 giờ", MOT_GIO), ("trong 1 ngày", MOT_NGAY)]:
        mk = gan["kc"] <= nguong
        dong = {"khoang": ten, "so_dong_r3": int(mk.sum()), "ty_le": mk.mean()}
        for nhom in ["tài khoản batch", "người dùng thật", "user_002"]:
            dong["rpb_do_" + nhom] = int((mk & (gan["nguoi_rpb"] == nhom)).sum())
        rows.append(dong)
        p(f"  Gỡ khóa {ten} sau lần sửa giá: {pct(int(mk.sum()), N, 1)} dòng R3; người gỡ: "
          + "; ".join(f"{k[7:]} {vn(v)}" for k, v in dong.items() if k.startswith("rpb_do_")))
    sau_hon = gan["kc"].notna() & (gan["kc"] > MOT_NGAY)
    p(f"  Gỡ khóa sau hơn 1 ngày: {pct(int(sau_hon.sum()), N, 1)}; không có gỡ khóa nào sau lần sửa giá: "
      f"{pct(int(gan['kc'].isna().sum()), N, 1)}.")
    # đối chứng: dòng hàng có Change Price nhưng không phải R3 (sửa giá trước hoặc không có hóa đơn)
    khac = kq["R3"][1] - r3
    cp_k = cp_all[cp_all["case"].isin(khac)]
    gan_k = khoang_den_rpb_sau(cp_k, rpb[rpb["case"].isin(khac)]).sort_values("kc").drop_duplicates("case")
    gan_k = gan_k.set_index("case").reindex(sorted(khac))
    p(f"  Đối chứng: {vn(len(khac))} dòng có Change Price nhưng không bị R3 đánh dấu; gỡ khóa trong 1 giờ sau lần sửa giá: "
      f"{pct(int((gan_k['kc'] <= MOT_GIO).sum()), len(khac), 1)}; trong 1 phút: {pct(int((gan_k['kc'] <= MOT_PHUT).sum()), len(khac), 1)}.")
    pd.DataFrame(rows).to_csv(ra / "r3_go_khoa_sau_sua_gia.csv", index=False, encoding="utf-8-sig")
    gan.to_csv(ra / "r3_tung_dong_hang.csv", encoding="utf-8-sig")


# ---------------------------------------------------------------- 4. bổ sung nhỏ
def bo_sung(ev, pv, moc, kq, loai, d, ra, p):
    N = len(pv)
    tt_idx = loai.index[loai == "thanh_toan"]
    da_tra = set(ev.loc[tt_idx, "case"])

    p("\n=== 4a. Nhánh (b) không có Remove Payment Block ===")
    _, x = sq1.nhanh(ev, pv)
    b = x[(x["nhanh"] == sq1.NHOM_NHANH[1]) & ~x["co_rpb"]]
    tra = b.index.isin(da_tra)
    p(f"  {vn(len(b))} dòng hàng nhánh (b) không có gỡ khóa; đã có Clear Invoice \"thanh toán\": {pct(int(tra.sum()), len(b), 1)}.")
    p("  Trạng thái: " + "; ".join(f"{k} {vn(v)}" for k, v in d["trang_thai"].reindex(b.index).value_counts().items()))
    if tra.any():
        tra_dau = ev.loc[tt_idx].groupby("case")["ts"].min().reindex(b.index[tra])
        gr_dau = x.loc[b.index[tra], "gr_dau"]
        p(f"  Trong số đã thanh toán: thanh toán trước lần nhận hàng đầu tiên {vn(int((tra_dau < gr_dau).sum()))}; "
          f"có R1 {vn(int(d.loc[b.index[tra], 'R1'].sum()))}; trung vị từ nhận hàng đầu tiên đến thanh toán đầu tiên "
          f"{vn(((tra_dau - gr_dau).dt.total_seconds() / 86400).median(), ',.1f')} ngày.")
    b.assign(da_thanh_toan=tra, trang_thai=d["trang_thai"].reindex(b.index)).to_csv(
        ra / "nhanh_b_khong_rpb.csv", encoding="utf-8-sig")

    p("\n=== 4b. R2a trên dòng hàng không có hủy (khám phá sau đóng băng, chỉ đếm) ===")
    co_huy = set(ev.loc[ev["act"].isin([CANCEL_IR, CANCEL_GR]), "case"])
    r2a = set(d.index[d["R2a"]])
    r2a_kh = r2a - co_huy
    chua_gr = set(d.index[d["R2a_chua_nhan_hang"]])
    r1 = set(d.index[d["R1"]])
    p(f"  R2a: {vn(len(r2a))} dòng; trên dòng không hủy: {pct(len(r2a_kh), len(r2a), 1)} số dòng R2a "
      f"({vn(len(r2a_kh) / max(N - len(co_huy), 1), '.3%')} số dòng không hủy).")
    p(f"  Trong {vn(len(r2a_kh))} dòng R2a không hủy: chưa từng nhận hàng {vn(len(r2a_kh & chua_gr))}; "
      f"trùng R1 {vn(len(r2a_kh & r1))}. Trên dòng có hủy: R2a {vn(len(r2a & co_huy))}, trong đó chưa từng nhận hàng "
      f"{vn(len(r2a & co_huy & chua_gr))}, trùng R1 {vn(len(r2a & co_huy & r1))}.")

    p("\n=== 4c. Độ nhạy của Δ ===")
    rows = []
    cach = [("Δ = 60 phút (chính)", lambda: loai),
            ("Δ = 1 ngày", lambda: phan_loai_clear(ev, 1440)[0]),
            ("Δ = 60 phút, mở rộng 1.000 phút cùng người hủy", lambda: g4.phan_loai_clear_mo_rong(ev, g4.DELTA, g4.DELTA_MO_RONG))]
    for ten, ham in cach:
        lc = ham()
        tt = lc.index[lc == "thanh_toan"]
        sl = g5.so_lieu_case(ev, pv, lc)
        trang = g5.trang_thai_cac_dinh_nghia(sl)["moi"]
        tg = g5.thoi_gian(sl, trang, moc)
        rows.append({"cach_phan_loai": ten, "clear_bu_tru": int((lc == "bu_tru_huy").sum()),
                     "R1": len(r5.r1_theo_clear(ev, tt)), "R5_v22": len(g4.r5_v22(g4.dem_r5(ev, lc))[0]),
                     "hoan_tat": tg["n_ht"], "con_mo": tg["n_mo"], "trung_vi_ngay": tg["trung_vi"], "p90_ngay": tg["p90"],
                     "trung_vi_km_ngay": tg["km"]})
    b = pd.DataFrame(rows)
    b.to_csv(ra / "do_nhay_delta.csv", index=False, encoding="utf-8-sig")
    p(b.to_string(index=False, formatters={c: "{:.1f}".format for c in ["trung_vi_ngay", "p90_ngay", "trung_vi_km_ngay"]}))

    p("\n=== 4d. Cận trên tỷ lệ bỏ sót ===")
    p(f"  0/20 case bỏ sót: cận trên KTC 95% Clopper-Pearson hai phía {vn(clopper_pearson_tren(0, 20), '.1%')}; "
      f"một phía {vn(1 - 0.05 ** (1 / 20), '.1%')}.")


# ---------------------------------------------------------------- 5. alignment 100%
def alignment_toan_bo(ev, pv, moc, d, ra, p):
    p("\n=== 5. Alignment và IMf trên 100% biến thể ===")
    pm4py = sq1.nap_pm4py()
    if pm4py is None:
        p("  Chưa có pm4py nên bỏ qua. Cài bằng:  pip install pm4py  rồi chạy lại.")
        return
    import sq1_imf
    goc = sq2.chon_bien_the
    sq2.chon_bien_the = lambda vt, phu=1.0: goc(vt, 1.0)  # mọi biến thể; sq1_imf và sq2.alignment gọi qua sq2
    try:
        ra_a = ra / "alignment_100"
        ra_a.mkdir(exist_ok=True)
        t = time.time()
        p("\n  [Bảng 4.6 và 4.15: alignment với mô hình chuẩn]")
        sq2.alignment(ev, d, ra_a, p, pm4py)
        p(f"  ({time.time() - t:,.0f} giây)")
        ra_i = ra / "imf_100"
        ra_i.mkdir(exist_ok=True)
        t = time.time()
        p("\n  [Bảng 4.3: IMf so với mô hình chuẩn]")
        sq1_imf.phan_tich(ev, pv, moc, ra_i, lambda *x: p("  " + " ".join(str(i) for i in x)), pm4py)
        p(f"  ({time.time() - t:,.0f} giây)")
    finally:
        sq2.chon_bien_the = goc


# ---------------------------------------------------------------- phân tích
def phan_tich(ev, pv, moc, ra, p, luu, bo_alignment=False):
    N = len(pv)
    ev = ev.copy()
    ev["vi_tri"] = ev.groupby("case").cumcount() + 1
    p(f"Phạm vi {vn(N)} dòng hàng, {vn(len(ev))} sự kiện. Δ = {g4.DELTA} phút. Thời điểm trích xuất: {moc}")
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    kq, _ = kc.cac_luat(ev, pv, loai, g4.gio_dia_phuong(ev, lambda *a: None))
    d = sq2.dau_hieu(ev, pv)
    lech = {k: v for k, v in sq2.KET_QUA_KIEM_CHUNG.items() if int(d[k].sum()) != v}
    p("Số dòng hàng theo luật khớp kiểm chứng cuối 18/10." if not lech else f"!!! Lệch kiểm chứng cuối: {lech}")
    rng = np.random.default_rng(SEED)
    for ham in (lambda: kiem_tra_mau(ev, kq, ra, p), lambda: mau_r6(ev, kq, ra, p, rng), lambda: gia_tri(ev, d, ra, p),
                lambda: co_che_r3(ev, kq, ra, p), lambda: bo_sung(ev, pv, moc, kq, loai, d, ra, p)):
        ham()
        luu()
    if not bo_alignment:
        alignment_toan_bo(ev, pv, moc, d, ra, p)
        luu()


# ---------------------------------------------------------------- tự kiểm
def tu_kiem():
    loi = 0

    def kt(ten, dat, ct=""):
        nonlocal loi
        loi += not dat
        print(f"{ten:60s} {'ĐẠT' if dat else 'SAI ' + str(ct)}")

    kt("Siêu bội: P(X ≥ 0) = 1", abs(sieu_boi_tu_k(650, 67, 20, 0) - 1) < 1e-12)
    kt("Siêu bội: P(X ≥ 8 | 650, 67, 20) ≈ 0,039%", abs(sieu_boi_tu_k(650, 67, 20, 8) - 0.000390) < 2e-6,
       sieu_boi_tu_k(650, 67, 20, 8))
    kt("Clopper-Pearson 0/20 ≈ 16,8%", abs(clopper_pearson_tren(0, 20) - (1 - 0.025 ** (1 / 20))) < 1e-9)
    kt("Holm", np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]), holm([0.01, 0.04, 0.03]))
    a = hieu_ung_mann_whitney([3, 4, 5], [1, 2, 3])["A"]
    kt("Mann-Whitney A = (8 + 0,5) / 9", abs(a - 8.5 / 9) < 1e-12, a)
    x = np.random.default_rng(1).lognormal(size=200)
    y = np.random.default_rng(2).lognormal(size=300)
    a1, a2 = hieu_ung_mann_whitney(x, y)["A"], hieu_ung_mann_whitney(np.log(x) * 7 + 3, np.log(y) * 7 + 3)["A"]
    kt("Mann-Whitney không đổi qua biến đổi đơn điệu tăng", abs(a1 - a2) < 1e-12, (a1, a2))
    t0 = pd.Timestamp("2018-03-01 08:00")
    cp = pd.DataFrame({"case": ["A", "B", "C"], "ts": [t0, t0, t0], "user": ["u1", "u2", "u3"]})
    rpb = pd.DataFrame({"case": ["A", "B", "B", "C"], "ts": [t0, t0 - MOT_GIO, t0 + 2 * MOT_GIO, t0 - MOT_PHUT],
                        "user": ["batch_02", "u9", "u9", "u9"]})
    m = khoang_den_rpb_sau(cp, rpb).set_index("case")
    kt("Gỡ khóa cùng phút sau sửa giá", m.loc["A", "kc"] == pd.Timedelta(0) and m.loc["A", "user_rpb"] == "batch_02", m)
    kt("Bỏ qua gỡ khóa trước sửa giá, lấy lần sau", m.loc["B", "kc"] == 2 * MOT_GIO, m)
    kt("Không có gỡ khóa sau sửa giá", pd.isna(m.loc["C", "kc"]), m)
    nguoi = loai_nguoi(pd.Series(["batch_01", "user_002", "NONE", "user_123"])).tolist()
    kt("Phân loại người thực hiện", nguoi == ["tài khoản batch", "user_002", "NONE", "người dùng thật"], nguoi)
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
    ap.add_argument("--ra", default="ket_qua_kiem_chung_bo_sung")
    ap.add_argument("--bo-alignment", action="store_true", help="bỏ mục 5 (alignment 100%%), chạy nhanh hơn")
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
        (ra / "tom_tat_kiem_chung_bo_sung.txt").write_text("\n".join(out), encoding="utf-8")

    t = time.time()
    raw = sq2.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    p(f"Đọc log xong sau {time.time() - t:,.0f} giây.")
    phan_tich(ev, pv, moc, ra, p, luu, a.bo_alignment)
    luu()
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_kiem_chung_bo_sung.txt.")


if __name__ == "__main__":
    main()
