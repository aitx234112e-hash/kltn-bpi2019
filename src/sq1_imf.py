"""Mô hình khám phá bằng Inductive Miner (IMf) cho SQ1, so sánh với mô hình chuẩn hai nhánh.

Cùng góc nhìn và cách tính như mục 4 của sq2_tuan_thu.py: chỉ giữ 7 hoạt động của mô hình chuẩn; mô hình khám phá
từ toàn bộ biến thể (có tần suất); fitness và precision tính theo biến thể trên toàn bộ biến thể (sq2_tuan_thu.PHU).
  - IMf (ngưỡng nhiễu 0,2) khám phá từ toàn bộ phạm vi, đánh giá trên toàn bộ phạm vi và trên dòng hàng hoàn tất.
  - IMf khám phá chỉ từ dòng hàng hoàn tất, đánh giá trên dòng hàng hoàn tất.
  - Mô hình chuẩn trên cùng các tập, để đặt cạnh nhau trong một bảng.
Inductive Miner chỉ dùng để mô tả quy trình thực tế, không dùng làm chuẩn (Mục 3.3.2).

Cách chạy (cùng thư mục với sq2_tuan_thu.py và các file đi kèm; cần pm4py):
    python src/sq1_imf.py --tu-kiem
    python src/sq1_imf.py
Kết quả ghi vào ket_qua_sq1_imf/ (đổi bằng --ra); gửi lại tom_tat_sq1_imf.txt.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import g4_kiem_tra as g4
import g5_kiem_tra as g5
import sq1_kham_pha as sq1
import sq2_tuan_thu as sq2
from r5_v2_phan_loai_clear import buoc2_moc_trich_xuat, loc_pham_vi, phan_loai_clear, sap_xep
import cau_hinh

NGUONG_NHIEU = 0.2


def log_theo_tan_suat(dem):
    """EventLog gồm mỗi biến thể lặp lại đúng số dòng hàng (dùng chung một đối tượng Trace)."""
    from pm4py.objects.log.obj import EventLog
    log = EventLog()
    for i, (v, n) in enumerate(dem.items()):
        tr = sq2.tao_trace(v, f"bt{i}")
        for _ in range(int(n)):
            log.append(tr)
    return log


def kham_pha_imf(pm4py, dem, nguong=NGUONG_NHIEU):
    """Cây quy trình và Petri net IMf từ toàn bộ biến thể (dem: biến thể -> số dòng hàng)."""
    cay = pm4py.discover_process_tree_inductive(log_theo_tan_suat(dem), noise_threshold=nguong)
    net, im, fm = pm4py.convert_to_petri_net(cay)
    return cay, net, im, fm


def danh_gia(dem, net, im, fm):
    al = sq2.can_bien_the(list(dem.index), net, im, fm)
    fit, khop, lech = sq2.tong_hop_alignment(dem, al)
    prec = sq2.precision_theo_tan_suat(dem, net, im, fm)
    return {"so_bien_the": len(dem), "so_dong_hang_duoc_tinh": int(dem.sum()), "fitness": fit,
            "ty_le_khop_hoan_toan": khop, "precision": prec}, lech


def mo_ta_net(net):
    hien = sorted(t.label for t in net.transitions if t.label is not None)
    return (f"{len(net.places)} place, {len(net.transitions)} transition "
            f"({sum(t.label is None for t in net.transitions)} chuyển tiếp ẩn); nhãn: {', '.join(hien)}")


def phan_tich(ev, pv, moc, ra, p, pm4py):
    N = len(pv)
    idx = pd.Index(sorted(pv), name="case")
    p(f"Phạm vi {N:,} dòng hàng. Thời điểm trích xuất: {moc}. Ngưỡng nhiễu IMf: {NGUONG_NHIEU}")
    loai, _, _ = phan_loai_clear(ev, g4.DELTA)
    xong = g5.trang_thai_cac_dinh_nghia(g5.so_lieu_case(ev, pv, loai))["moi"].reindex(idx) == "Hoàn tất"
    p(f"Dòng hàng hoàn tất (định nghĩa G5): {g4.pct(int(xong.sum()), N)}")
    vt = sq2.bien_the_mo_hinh(ev, idx)
    tap = {"toàn bộ phạm vi": vt, "dòng hàng hoàn tất": vt[xong]}

    duong_dan = Path("ket_qua_sq1") / sq1.TEN_PNML
    chuan = pm4py.read_pnml(str(duong_dan)) if duong_dan.exists() else sq1.mo_hinh_chuan(pm4py)
    mo_hinh = {"Mô hình chuẩn hai nhánh": (chuan, None)}
    for ten_tap, ten_tep in [("toàn bộ phạm vi", "imf_toan_bo"), ("dòng hàng hoàn tất", "imf_hoan_tat")]:
        t = time.time()
        cay, net, im, fm = kham_pha_imf(pm4py, tap[ten_tap].value_counts())
        p(f"\n=== IMf khám phá từ {ten_tap} ({time.time() - t:,.0f} giây) ===")
        p(f"  Cây quy trình: {cay}")
        p(f"  Petri net: {mo_ta_net(net)}")
        pm4py.write_pnml(net, im, fm, str(ra / f"{ten_tep}.pnml"))
        (ra / f"{ten_tep}_cay.txt").write_text(str(cay), encoding="utf-8")
        try:
            pm4py.save_vis_process_tree(cay, str(ra / f"{ten_tep}_cay.png"))
            p(f"  Đã vẽ {ten_tep}_cay.png.")
        except Exception as e:
            p(f"  Không vẽ được hình cây ({type(e).__name__}); thường do chưa cài Graphviz. File .pnml và _cay.txt vẫn được ghi.")
        mo_hinh[f"IMf khám phá từ {ten_tap}"] = ((net, im, fm), ten_tap)

    rows = []
    for ten_mh, ((net, im, fm), chi_tap) in mo_hinh.items():
        for ten_tap, v in tap.items():
            if chi_tap == "dòng hàng hoàn tất" and ten_tap != chi_tap:
                continue
            t = time.time()
            kq, lech = danh_gia(sq2.chon_bien_the(v), net, im, fm)
            rows.append({"mo_hinh": ten_mh, "tap_danh_gia": ten_tap, "so_dong_hang": len(v), **kq})
            p(f"\n[{ten_mh} | đánh giá trên {ten_tap}] {kq['so_bien_the']} biến thể phủ "
              f"{g4.pct(kq['so_dong_hang_duoc_tinh'], len(v))}; fitness {kq['fitness']:.4f}; "
              f"khớp hoàn toàn {kq['ty_le_khop_hoan_toan']:.2%}; precision {kq['precision']:.4f} ({time.time() - t:,.0f} giây)")
            if len(lech):
                p("  Bước lệch phổ biến (tỷ lệ % trên số dòng hàng được tính):")
                p(lech.head(6).assign(ty_le_dong_hang=lambda x: (x["ty_le_dong_hang"] * 100).round(2)).to_string(index=False))
    bang = pd.DataFrame(rows)
    bang.to_csv(ra / "so_sanh_mo_hinh.csv", index=False, encoding="utf-8-sig")
    p("\n=== Bảng so sánh ===")
    p(bang.drop(columns=["so_dong_hang_duoc_tinh"]).round(4).to_string(index=False))


def tu_kiem():
    pm4py = sq1.nap_pm4py()
    if pm4py is None:
        print("Cần pm4py (pip install pm4py).")
        return 1
    loi = 0

    def kt(ten, dat, chi_tiet=""):
        nonlocal loi
        loi += not dat
        print(f"{ten:58s} {'ĐẠT' if dat else 'SAI ' + str(chi_tiet)}")

    V = sq1.VET_MO_HINH
    dem = pd.Series({sq1.MUI_TEN.join(V[k][0]): n for k, n in [("a_hop_le", 60), ("a_hop_le_co_rpb", 20), ("b_hop_le", 15),
                                                               ("b_thieu_rpb", 4), ("thieu_po", 1)]})
    log = log_theo_tan_suat(dem)
    kt("Log theo tần suất có đủ 100 vết", len(log) == 100, len(log))
    cay, net, im, fm = kham_pha_imf(pm4py, dem, 0.0)
    kq, _ = danh_gia(dem, net, im, fm)
    kt("IMf không lọc nhiễu: fitness = 1 trên chính log của nó", np.isclose(kq["fitness"], 1.0) and kq["ty_le_khop_hoan_toan"] == 1, kq)
    cay2, net2, im2, fm2 = kham_pha_imf(pm4py, dem, 0.2)
    kq2, _ = danh_gia(dem, net2, im2, fm2)
    kt("IMf ngưỡng 0,2: fitness trong (0, 1]", 0 < kq2["fitness"] <= 1, kq2)
    kt("Precision trong (0, 1]", 0 < kq["precision"] <= 1 and 0 < kq2["precision"] <= 1, (kq["precision"], kq2["precision"]))
    nhan = {t.label for t in net2.transitions if t.label}
    kt("IMf ngưỡng 0,2 giữ đủ 5 hoạt động của log", nhan == {sq1.PO, sq1.GR, sq1.IR, sq1.RPB, sq1.CLEAR}, nhan)
    chuan = sq1.mo_hinh_chuan(pm4py)
    kq_c, _ = danh_gia(dem, *chuan)
    kt("Mô hình chuẩn: 95% vết khớp hoàn toàn", np.isclose(kq_c["ty_le_khop_hoan_toan"], 0.95), kq_c)
    print(f"  (cây IMf ngưỡng 0,2 trên log giả lập: {cay2})")
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
    ap.add_argument("--ra", default="ket_qua_sq1_imf")
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        sys.exit(tu_kiem())
    if not a.log:
        ap.error("cần đường dẫn log hoặc --tu-kiem")
    if tu_kiem():
        sys.exit("Kiểm thử giả lập không đạt, dừng.")
    pm4py = sq1.nap_pm4py()
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
    phan_tich(ev, pv, moc, ra, p, pm4py)
    (ra / "tom_tat_sq1_imf.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_sq1_imf.txt.")


if __name__ == "__main__":
    main()
