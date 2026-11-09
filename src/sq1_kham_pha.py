"""Khám phá quy trình cho SQ1: các phần không phụ thuộc bộ luật kiểm soát.

  1. Đồ thị trực tiếp kế tiếp (DFG): mỗi cạnh có số lần, số case, trung vị / trung bình thời gian (giờ); tần suất hoạt động.
     dfg_95.dot (Graphviz) chỉ giữ các cạnh phổ biến nhất phủ khoảng 95% tần suất cạnh; có pm4py thì vẽ thêm dfg_95.png.
  2. Biến thể (chuỗi hoạt động của mỗi dòng hàng): số biến thể, 10 biến thể đầu, số biến thể cần để phủ 95% case.
     Làm thêm trên góc nhìn mô hình: chỉ giữ 7 hoạt động của mô hình chuẩn, các hoạt động khác bị bỏ như trước khi alignment.
  3. Tỷ lệ hai nhánh: (a) Record Goods Receipt đầu tiên trước Record Invoice Receipt đầu tiên, (b) ngược lại,
     trùng thời điểm, chỉ có GR, chỉ có IR, không có cả hai; với (a) và (b) báo tỷ lệ có Remove Payment Block.
  4. Tài khoản batch (g3.la_batch): số sự kiện, tỷ lệ, 3 hoạt động nhiều nhất, giờ địa phương phổ biến nhất;
     đối chiếu user_002 (hành vi giống đợt thanh toán tự động): số sự kiện và tỷ lệ trong các Clear Invoice.
  5. Mô hình chuẩn hai nhánh (Petri net, cần pm4py) ghi ra QuyTrinhChuan_P2P_2nhanh.pnml, kiểm thử bằng alignment
     trên 8 vết giả lập. Không chạy alignment trên log thật ở đây (làm sau khi chốt bộ luật).
Thứ tự và khoảng thời gian tính trên timestamp gốc (UTC); chỉ giờ trong ngày ở mục 4 đổi sang Europe/Amsterdam.
Ở mục 1–2, sự kiện chỉ ghi đến ngày (Vendor creates invoice, Vendor creates debit memo, 23:59 giờ địa phương) được dời về
00:00 giờ địa phương cùng ngày để không tạo cạnh giả (Bảng 7 của đề cương); mục 3–5 không dùng hai hoạt động này.
Sự kiện cùng timestamp xếp theo sap_xep của r5 (GR trước IR trước Clear).

Cách chạy (cùng thư mục với kiem_chung_cuoi.py, g5_kiem_tra.py, g4_kiem_tra.py, g3_kiem_tra_bo_sung.py,
r5_v2_phan_loai_clear.py, r4_thanh_phan.py, kiem_tra_trang_thai_case.py):
    python src/sq1_kham_pha.py --tu-kiem
    python src/sq1_kham_pha.py
Mục 1–4 chỉ cần pandas/numpy. Mục 5 và hình PNG cần pm4py (pip install pm4py); PNG cần thêm Graphviz trong PATH.
Kết quả ghi vào ket_qua_sq1/ (đổi bằng --ra); gửi lại tom_tat_sq1.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import g3_kiem_tra_bo_sung as g3
import g4_kiem_tra as g4
from r5_v2_phan_loai_clear import CLEAR, GR, IR, PO, buoc2_moc_trich_xuat, loc_pham_vi, sap_xep
import cau_hinh

PR, OC, RPB = "Create Purchase Requisition Item", "Receive Order Confirmation", "Remove Payment Block"
HD_MO_HINH = [PR, PO, OC, GR, IR, RPB, CLEAR]
PHU = 0.95
MUI_TEN = " → "
NHOM_NHANH = ["(a) GR trước IR", "(b) IR trước GR", "trùng thời điểm", "chỉ có GR", "chỉ có IR", "không có cả hai"]
USER_002 = "user_002"
TEN_PNML = "QuyTrinhChuan_P2P_2nhanh.pnml"


def nap_pm4py():
    try:
        import pm4py
        return pm4py
    except ImportError:
        return None


def dau(ev, act):
    return ev[ev["act"] == act].groupby("case")["ts"].min()


HD_CHI_NGAY = ["Vendor creates invoice", "Vendor creates debit memo"]


def doi_ve_dau_ngay(ev):
    """Dời sự kiện chỉ ghi đến ngày về 00:00 giờ địa phương cùng ngày (timestamp vẫn ở UTC), rồi xếp lại."""
    ev = ev.copy()
    m = ev["act"].isin(HD_CHI_NGAY)
    dp = ev.loc[m, "ts"].dt.tz_localize("UTC").dt.tz_convert(g4.MUI_GIO)
    ev.loc[m, "ts"] = dp.dt.normalize().dt.tz_convert("UTC").dt.tz_localize(None)
    return sap_xep(ev)


# ---------------------------------------------------------------- 1. DFG
def dfg(ev):
    """Cạnh trực tiếp kế tiếp trong cùng case (ev đã qua sap_xep)."""
    cung = ev["case"].eq(ev["case"].shift(-1))
    c = pd.DataFrame({"case": ev["case"], "tu": ev["act"], "den": ev["act"].shift(-1),
                      "gio": (ev["ts"].shift(-1) - ev["ts"]).dt.total_seconds() / 3600})[cung]
    bang = c.groupby(["tu", "den"]).agg(so_lan=("case", "size"), so_case=("case", "nunique"),
                                        trung_vi_gio=("gio", "median"), trung_binh_gio=("gio", "mean")).reset_index()
    bang = bang.sort_values(["so_lan", "tu", "den"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    bang["ty_le"] = bang["so_lan"] / bang["so_lan"].sum()
    bang["ty_le_tich_luy"] = bang["ty_le"].cumsum()
    return bang


def tan_suat_hoat_dong(ev):
    hd = ev.groupby("act").agg(so_su_kien=("case", "size"), so_case=("case", "nunique"))
    hd["so_case_bat_dau"] = ev.groupby("case")["act"].first().value_counts()
    hd["so_case_ket_thuc"] = ev.groupby("case")["act"].last().value_counts()
    hd = hd.fillna(0).astype(int).sort_values("so_su_kien", ascending=False)
    hd["ty_le_su_kien"] = (hd["so_su_kien"] / hd["so_su_kien"].sum()).round(6)
    return hd


def loc_canh(bang, phu=PHU):
    """Ít cạnh nhất (theo thứ tự tần suất giảm dần) sao cho tổng tần suất đạt >= phu."""
    k = int((bang["ty_le_tich_luy"] < phu - 1e-12).sum()) + 1
    return bang.head(min(k, len(bang)))


def ghi_dot(canh, hd, duong_dan):
    def q(s):
        return '"' + str(s).replace('"', "'") + '"'
    nut = sorted(set(canh["tu"]) | set(canh["den"]))
    lon = canh["so_lan"].max()
    tieu_de = "DFG phạm vi, giữ %d cạnh phủ %.1f%% tần suất cạnh" % (len(canh), canh["ty_le"].sum() * 100)
    dong = ["digraph DFG {", "  rankdir=LR;", '  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Arial"];',
            '  edge [fontname="Arial", fontsize=10];', "  label=" + q(tieu_de) + ";"]
    for n in nut:
        nhan = "%s\\n%s" % (n, f"{int(hd.loc[n, 'so_su_kien']):,}")  # \\n = xuống dòng trong nhãn Graphviz
        dong.append("  %s [label=%s];" % (q(n), q(nhan)))
    for r in canh.itertuples():
        nhan = f"{r.so_lan:,} | {r.trung_vi_gio:,.1f} giờ"
        dong.append("  %s -> %s [label=%s, penwidth=%.2f];" % (q(r.tu), q(r.den), q(nhan), 1 + 5 * r.so_lan / lon))
    dong.append("}")
    Path(duong_dan).write_text("\n".join(dong) + "\n", encoding="utf-8")


def ve_png(pm4py, canh, ev, duong_dan, p):
    giu = set(canh["tu"]) | set(canh["den"])
    d = {(r.tu, r.den): int(r.so_lan) for r in canh.itertuples()}
    bd = ev.groupby("case")["act"].first().value_counts()
    kt = ev.groupby("case")["act"].last().value_counts()
    try:
        pm4py.save_vis_dfg(d, {k: int(v) for k, v in bd.items() if k in giu}, {k: int(v) for k, v in kt.items() if k in giu},
                           str(duong_dan))
        p(f"  Đã vẽ {Path(duong_dan).name} bằng pm4py.")
    except Exception as e:
        p(f"  Không vẽ được PNG bằng pm4py ({type(e).__name__}: {e}). Thường do chưa cài Graphviz (https://graphviz.org) "
          f"hoặc chưa thêm vào PATH; có thể dán dfg_95.dot vào một trình xem Graphviz.")


# ---------------------------------------------------------------- 2. biến thể
def bien_the(ev, cases, chi_giu=None):
    e = ev if chi_giu is None else ev[ev["act"].isin(chi_giu)]
    vt = e.groupby("case", sort=False)["act"].agg(MUI_TEN.join).reindex(pd.Index(sorted(cases), name="case")).fillna("(rỗng)")
    bang = vt.value_counts().rename_axis("bien_the").reset_index(name="so_case")
    bang = bang.sort_values(["so_case", "bien_the"], ascending=[False, True], kind="stable").reset_index(drop=True)
    bang.insert(0, "hang", np.arange(1, len(bang) + 1))
    bang["so_hoat_dong"] = bang["bien_the"].map(lambda s: 0 if s == "(rỗng)" else s.count(MUI_TEN) + 1)
    bang["ty_le"] = bang["so_case"] / bang["so_case"].sum()
    bang["ty_le_tich_luy"] = bang["ty_le"].cumsum()
    n95 = int((bang["ty_le_tich_luy"] < PHU - 1e-12).sum()) + 1
    return bang, min(n95, len(bang))


# ---------------------------------------------------------------- 3. hai nhánh
def nhanh(ev, cases):
    idx = pd.Index(sorted(cases), name="case")
    gr, ir = dau(ev, GR).reindex(idx), dau(ev, IR).reindex(idx)
    co_ca_hai = gr.notna() & ir.notna()
    loai = np.select([co_ca_hai & (gr < ir), co_ca_hai & (ir < gr), co_ca_hai & (gr == ir), gr.notna(), ir.notna()],
                     NHOM_NHANH[:5], NHOM_NHANH[5])
    x = pd.DataFrame({"nhanh": loai, "gr_dau": gr, "ir_dau": ir}, index=idx)
    x["co_rpb"] = idx.isin(set(ev.loc[ev["act"] == RPB, "case"]))
    bang = x.groupby("nhanh")["co_rpb"].agg(so_case="size", so_case_co_rpb="sum").reindex(NHOM_NHANH).fillna(0).astype(int)
    bang["ty_le"] = (bang["so_case"] / len(idx)).round(6)
    bang["ty_le_co_rpb_trong_nhom"] = (bang["so_case_co_rpb"] / bang["so_case"].replace(0, np.nan)).round(6)
    return bang, x


# ---------------------------------------------------------------- 4. tài khoản batch
def tai_khoan_batch(ev, ts_dp):
    u = ev["user"].astype(str)
    gio = ts_dp.dt.hour
    N, n_clear = len(ev), int((ev["act"] == CLEAR).sum())
    rows = []
    ds = sorted(u[g3.la_batch(u)].unique()) + ([USER_002] if (u == USER_002).any() else [])
    for tk in ds:
        m = u == tk
        hd = ev.loc[m, "act"].value_counts()
        hd = hd.sort_values(ascending=False, kind="stable")
        g = gio[m].value_counts().sort_index().sort_values(ascending=False, kind="stable")
        rows.append({"tai_khoan": tk, "loai": "batch" if tk != USER_002 else "đối chiếu (giống thanh toán tự động)",
                     "so_su_kien": int(m.sum()), "ty_le_su_kien": round(m.sum() / N, 6),
                     "hoat_dong_nhieu_nhat": "; ".join(f"{a} ({n:,})" for a, n in hd.head(3).items()),
                     "gio_dia_phuong_pho_bien": int(g.index[0]), "ty_le_su_kien_trong_gio_do": round(g.iloc[0] / m.sum(), 4),
                     "so_clear": int((m & (ev["act"] == CLEAR)).sum()),
                     "ty_le_trong_clear": round((m & (ev["act"] == CLEAR)).sum() / n_clear, 6) if n_clear else np.nan})
    return pd.DataFrame(rows, columns=["tai_khoan", "loai", "so_su_kien", "ty_le_su_kien", "hoat_dong_nhieu_nhat",
                                       "gio_dia_phuong_pho_bien", "ty_le_su_kien_trong_gio_do", "so_clear", "ty_le_trong_clear"])


# ---------------------------------------------------------------- 5. mô hình chuẩn
def mo_hinh_chuan(pm4py):
    """PR? → PO → OC? → XOR{ (a) GR → IR → RPB? → Clear ; (b) IR → GR → RPB → Clear }.
    Nhãn trùng giữa hai nhánh là cố ý; bước tùy chọn = chuyển tiếp ẩn (label None) để bỏ qua."""
    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
    net = PetriNet("QuyTrinhChuan_P2P_2nhanh")
    cho = {}
    for ten in ["bat_dau", "sau_pr", "sau_po", "chon_nhanh", "a_sau_gr", "a_sau_ir", "a_truoc_clear",
                "b_sau_ir", "b_sau_gr", "b_truoc_clear", "ket_thuc"]:
        cho[ten] = PetriNet.Place(ten)
        net.places.add(cho[ten])

    def t(ten, nhan, tu, den):
        tr = PetriNet.Transition(ten, nhan)
        net.transitions.add(tr)
        add_arc_from_to(cho[tu], tr, net)
        add_arc_from_to(tr, cho[den], net)

    t("pr", PR, "bat_dau", "sau_pr")
    t("bo_qua_pr", None, "bat_dau", "sau_pr")
    t("po", PO, "sau_pr", "sau_po")
    t("oc", OC, "sau_po", "chon_nhanh")
    t("bo_qua_oc", None, "sau_po", "chon_nhanh")
    # nhánh (a): nhận hàng trước hóa đơn
    t("a_gr", GR, "chon_nhanh", "a_sau_gr")
    t("a_ir", IR, "a_sau_gr", "a_sau_ir")
    t("a_rpb", RPB, "a_sau_ir", "a_truoc_clear")
    t("a_bo_qua_rpb", None, "a_sau_ir", "a_truoc_clear")
    t("a_clear", CLEAR, "a_truoc_clear", "ket_thuc")
    # nhánh (b): hóa đơn trước nhận hàng, bắt buộc gỡ chặn thanh toán
    t("b_ir", IR, "chon_nhanh", "b_sau_ir")
    t("b_gr", GR, "b_sau_ir", "b_sau_gr")
    t("b_rpb", RPB, "b_sau_gr", "b_truoc_clear")
    t("b_clear", CLEAR, "b_truoc_clear", "ket_thuc")
    im, fm = Marking(), Marking()
    im[cho["bat_dau"]] = 1
    fm[cho["ket_thuc"]] = 1
    return net, im, fm


VET_MO_HINH = {  # tên: (vết, kỳ vọng fitness = 1?)
    "a_hop_le": ([PO, GR, IR, CLEAR], True),
    "a_hop_le_co_rpb": ([PO, GR, IR, RPB, CLEAR], True),
    "b_hop_le": ([PO, IR, GR, RPB, CLEAR], True),
    "day_du_pr_oc": ([PR, PO, OC, GR, IR, CLEAR], True),
    "b_thieu_rpb": ([PO, IR, GR, CLEAR], False),
    "tra_truoc_nhan_hang": ([PO, IR, CLEAR, GR], False),
    "clear_truoc_ir": ([PO, GR, CLEAR, IR], False),
    "thieu_po": ([GR, IR, CLEAR], False),
}


def kiem_thu_mo_hinh(net, im, fm):
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
    from pm4py.objects.log.obj import Event, EventLog, Trace
    log = EventLog()
    t0 = pd.Timestamp("2018-03-01 08:00")
    for ten, (vet, _) in VET_MO_HINH.items():
        tr = Trace(attributes={"concept:name": ten})
        for i, a in enumerate(vet):
            tr.append(Event({"concept:name": a, "time:timestamp": (t0 + pd.Timedelta(days=i)).to_pydatetime()}))
        log.append(tr)
    kq = alignments.apply(log, net, im, fm, parameters={"show_progress_bar": False})
    rows = []
    for (ten, (vet, ky_vong)), a in zip(VET_MO_HINH.items(), kq):
        f = float(a["fitness"])
        dat = (f == 1.0) if ky_vong else (f < 1.0)
        lech = [f"{x[0]}/{x[1]}" for x in a["alignment"] if x[0] != x[1] and not (x[0] == ">>" and x[1] is None)]
        rows.append({"vet": ten, "chuoi": MUI_TEN.join(vet), "ky_vong": "fitness = 1" if ky_vong else "fitness < 1",
                     "fitness": round(f, 4), "lech": "; ".join(lech), "ket_qua": "ĐẠT" if dat else "SAI"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- phân tích
def phan_tich(ev, pv, moc, ra, p):
    N = len(pv)
    p(f"Phạm vi {N:,} dòng hàng, {len(ev):,} sự kiện, {ev['act'].nunique()} hoạt động. Thời điểm trích xuất: {moc}")

    ev_goc, ev = ev, doi_ve_dau_ngay(ev)
    p("\n=== 1. DFG ===")
    p(f"(Mục 1–2: {', '.join(HD_CHI_NGAY)} dời về 00:00 giờ địa phương cùng ngày.)")
    hd = tan_suat_hoat_dong(ev)
    hd.to_csv(ra / "tan_suat_hoat_dong.csv", encoding="utf-8-sig")
    p("Tần suất hoạt động:\n" + hd.to_string())
    bang = dfg(ev)
    bang.round({"trung_vi_gio": 2, "trung_binh_gio": 2, "ty_le": 6, "ty_le_tich_luy": 6}).to_csv(
        ra / "dfg_canh.csv", index=False, encoding="utf-8-sig")
    canh = loc_canh(bang)
    p(f"\n{len(bang):,} cạnh, {int(bang['so_lan'].sum()):,} lần chuyển. Giữ {len(canh)} cạnh phủ {canh['ty_le'].sum():.2%} "
      f"tần suất cạnh, {len(set(canh['tu']) | set(canh['den']))} hoạt động.")
    p("20 cạnh nhiều nhất:\n" + bang.head(20).round(2).to_string(index=False))
    ghi_dot(canh, hd, ra / "dfg_95.dot")
    pm4py = nap_pm4py()
    if pm4py is None:
        p("  Chưa có pm4py nên không vẽ PNG (pip install pm4py); file dfg_95.dot vẫn được ghi.")
    else:
        ve_png(pm4py, canh, ev, ra / "dfg_95.png", p)

    p("\n=== 2. Biến thể ===")
    for nhan, chi_giu, ten in [("toàn bộ hoạt động", None, "bien_the.csv"),
                               ("góc nhìn mô hình (7 hoạt động)", HD_MO_HINH, "bien_the_mo_hinh.csv")]:
        bt, n95 = bien_the(ev, pv, chi_giu)
        bt.to_csv(ra / ten, index=False, encoding="utf-8-sig")
        p(f"{nhan}: {len(bt):,} biến thể; cần {n95:,} biến thể để phủ {PHU:.0%} case "
          f"(10 biến thể đầu phủ {bt['ty_le_tich_luy'].iloc[min(9, len(bt) - 1)]:.2%}).")
        p(bt.head(10).assign(ty_le=lambda d: (d["ty_le"] * 100).round(2), ty_le_tich_luy=lambda d: (d["ty_le_tich_luy"] * 100).round(2))
          [["hang", "so_case", "ty_le", "ty_le_tich_luy", "so_hoat_dong", "bien_the"]].to_string(index=False))
        p("(ty_le, ty_le_tich_luy tính bằng %)\n")

    ev = ev_goc
    p("=== 3. Tỷ lệ hai nhánh (theo GR đầu tiên và IR đầu tiên) ===")
    bn, x = nhanh(ev, pv)
    bn.to_csv(ra / "ty_le_nhanh.csv", encoding="utf-8-sig")
    x.to_csv(ra / "nhanh_tung_case.csv", encoding="utf-8-sig")
    for nhom, r in bn.iterrows():
        dong = f"  {nhom:18s} {g4.pct(int(r['so_case']), N)}"
        if nhom in NHOM_NHANH[:2]:
            dong += f"; có Remove Payment Block: {g4.pct(int(r['so_case_co_rpb']), int(r['so_case']))}"
        p(dong)

    p("\n=== 4. Tài khoản batch ===")
    ts_dp = g4.gio_dia_phuong(ev, p)
    tk = tai_khoan_batch(ev, ts_dp)
    tk.to_csv(ra / "tai_khoan_batch.csv", index=False, encoding="utf-8-sig")
    p(tk.to_string(index=False) if len(tk) else "  Không có tài khoản batch trong phạm vi.")
    if USER_002 in set(tk["tai_khoan"]):
        r = tk.set_index("tai_khoan").loc[USER_002]
        p(f"{USER_002}: {g4.pct(int(r['so_su_kien']), len(ev))} sự kiện; thực hiện "
          f"{g4.pct(int(r['so_clear']), int((ev['act'] == CLEAR).sum()))} Clear Invoice.")
    else:
        p(f"{USER_002} không xuất hiện trong phạm vi.")

    p("\n=== 5. Mô hình chuẩn hai nhánh ===")
    if pm4py is None:
        p("  Chưa có pm4py nên bỏ qua mục này. Cài bằng:  pip install pm4py  rồi chạy lại.")
        return 0
    net, im, fm = mo_hinh_chuan(pm4py)
    pm4py.write_pnml(net, im, fm, str(ra / TEN_PNML))
    p(f"  Đã ghi {TEN_PNML}: {len(net.places)} place, {len(net.transitions)} transition "
      f"({sum(t.label is None for t in net.transitions)} chuyển tiếp ẩn).")
    net, im, fm = pm4py.read_pnml(str(ra / TEN_PNML))  # kiểm thử trên chính file vừa ghi
    kt = kiem_thu_mo_hinh(net, im, fm)
    kt.to_csv(ra / "kiem_thu_mo_hinh.csv", index=False, encoding="utf-8-sig")
    p(kt.drop(columns="chuoi").to_string(index=False))
    loi = int((kt["ket_qua"] != "ĐẠT").sum())
    p("  Mô hình đạt cả 8 vết giả lập." if loi == 0 else f"  !!! {loi} vết giả lập SAI, mô hình chưa đúng.")
    p("  (Chưa chạy alignment trên log thật; làm sau khi chốt bộ luật.)")
    return loi


# ---------------------------------------------------------------- tự kiểm
def log_tu_kiem():
    t = pd.Timestamp("2018-03-01 08:00")
    gio, ngay = pd.Timedelta(hours=1), pd.Timedelta(days=1)
    vet = {
        # (hoạt động, thời điểm, người)
        "A1": [(PO, 0 * gio, "user_001"), (GR, ngay, "user_005"), (IR, 2 * ngay, "user_003"), (CLEAR, 30 * ngay, USER_002)],
        "A2_rpb": [(PR, 0 * gio, "user_001"), (PO, 5 * gio, "user_001"), (GR, 5 * gio + ngay, "user_005"),
                   (IR, 5 * gio + 2 * ngay, "user_003"), (RPB, 5 * gio + 3 * ngay, "user_003"), (CLEAR, 5 * gio + 30 * ngay, USER_002)],
        "A3": [(PO, 0 * gio, "batch_01"), (GR, 3 * ngay, "user_005"), (IR, 4 * ngay, "user_003"), (CLEAR, 30 * ngay, USER_002)],
        "A4_lap": [(PO, 0 * gio, "user_001"), (GR, ngay, "user_005"), (IR, 2 * ngay, "user_003"), (GR, 3 * ngay, "user_005"),
                   (IR, 4 * ngay, "user_003"), (CLEAR, 30 * ngay, "user_004")],
        "B1_rpb": [(PO, 0 * gio, "user_001"), (IR, ngay, "user_003"), (GR, 2 * ngay, "user_005"), (RPB, 3 * ngay, "user_003"),
                   (CLEAR, 30 * ngay, USER_002)],
        "B2_change_price": [(PO, 0 * gio, "user_001"), ("Change Price", ngay, "user_001"), (IR, 2 * ngay, "user_003"),
                            (GR, 3 * ngay, "user_005"), (CLEAR, 30 * ngay, USER_002)],
        "B3": [(PO, 0 * gio, "user_001"), (IR, ngay, "user_003"), (GR, 3 * ngay, "user_005"), (CLEAR, 30 * ngay, USER_002)],
        # IR được ghi trước GR nhưng cùng timestamp: sap_xep đưa GR lên trước, vẫn là nhóm "trùng thời điểm"
        "T_trung": [(PO, 0 * gio, "user_001"), (IR, ngay, "user_003"), (GR, ngay, "user_005"), (CLEAR, 30 * ngay, USER_002)],
        "G_chi_gr": [(PO, 0 * gio, "user_001"), (GR, ngay, "user_005")],
        "I_chi_ir": [(PO, 0 * gio, "user_001"), (IR, ngay, "user_003")],
        # Vendor creates invoice lúc 22:59 UTC (= 23:59 Amsterdam, giờ mùa đông) để gio_dia_phuong nhận ra log ở UTC
        "N_khong": [("Vendor creates invoice", -9 * gio - pd.Timedelta(minutes=1), "NONE"), (PR, 0 * gio, "batch_02"),
                    (PO, gio, "batch_02")],
    }
    return pd.DataFrame([{"case": c, "act": a, "ts": t + dt, "user": u} for c, ds in vet.items() for a, dt, u in ds])


def tu_kiem():
    ev = sap_xep(log_tu_kiem())
    pv = set(ev["case"])
    loi = 0

    def kt(ten, dat, chi_tiet=""):
        nonlocal loi
        loi += not dat
        print(f"{ten:42s} {'ĐẠT' if dat else 'SAI ' + str(chi_tiet)}")

    # 1. DFG: so với cách đếm trực tiếp từng vết
    dem = {}
    for _, g in ev.groupby("case"):
        a, ts = list(g["act"]), list(g["ts"])
        for i in range(len(a) - 1):
            dem.setdefault((a[i], a[i + 1]), []).append((g["case"].iloc[0], (ts[i + 1] - ts[i]).total_seconds() / 3600))
    bang = dfg(ev).set_index(["tu", "den"])
    kt("DFG: tập cạnh", set(bang.index) == set(dem), set(bang.index) ^ set(dem))
    kt("DFG: số lần, số case, trung vị, trung bình", all(
        bang.loc[k, "so_lan"] == len(v) and bang.loc[k, "so_case"] == len({c for c, _ in v})
        and np.isclose(bang.loc[k, "trung_vi_gio"], np.median([h for _, h in v]))
        and np.isclose(bang.loc[k, "trung_binh_gio"], np.mean([h for _, h in v])) for k, v in dem.items()))
    r = bang.loc[(GR, IR)]
    kt("DFG: GR→IR = 6 lần, 5 case, trung vị 24 giờ", (r["so_lan"], r["so_case"], r["trung_vi_gio"]) == (6, 5, 24.0), r.to_dict())
    r = bang.loc[(PO, GR)]
    kt("DFG: PO→GR trung bình 32 giờ", np.isclose(r["trung_binh_gio"], 32.0), r.to_dict())
    canh = loc_canh(dfg(ev))
    kt("DFG: lọc 95% là tập nhỏ nhất đủ 95%", canh["ty_le"].sum() >= PHU and canh["ty_le"].iloc[:-1].sum() < PHU)
    hd = tan_suat_hoat_dong(ev)
    kt("Tần suất: GR 10 sự kiện / 9 case, Clear 8", (hd.loc[GR, "so_su_kien"], hd.loc[GR, "so_case"], hd.loc[CLEAR, "so_su_kien"]) == (10, 9, 8))

    # 2. biến thể
    bt, n95 = bien_the(ev, pv)
    kt("Biến thể: 9 biến thể, phủ 95% cần 9", (len(bt), n95) == (9, 9), (len(bt), n95))
    kt("Biến thể đầu: PO→GR→IR→Clear, 3 case (có T_trung)",
       bt.loc[0, "bien_the"] == MUI_TEN.join([PO, GR, IR, CLEAR]) and bt.loc[0, "so_case"] == 3, bt.head(1).to_dict("records"))
    btm, n95m = bien_the(ev, pv, HD_MO_HINH)
    kt("Góc nhìn mô hình: 8 biến thể, B2 gộp với B3",
       len(btm) == 8 and btm.set_index("bien_the").loc[MUI_TEN.join([PO, IR, GR, CLEAR]), "so_case"] == 2, len(btm))
    kt("Góc nhìn mô hình: tỷ lệ tích lũy cuối = 1", np.isclose(btm["ty_le_tich_luy"].iloc[-1], 1.0) and n95m == 8)

    # 3. hai nhánh
    bn, x = nhanh(ev, pv)
    ky_vong = {"A1": 0, "A2_rpb": 0, "A3": 0, "A4_lap": 0, "B1_rpb": 1, "B2_change_price": 1, "B3": 1, "T_trung": 2,
               "G_chi_gr": 3, "I_chi_ir": 4, "N_khong": 5}
    sai = {c: x.loc[c, "nhanh"] for c, i in ky_vong.items() if x.loc[c, "nhanh"] != NHOM_NHANH[i]}
    kt("Nhánh từng case", not sai, sai)
    kt("Nhánh (a): 4 case, 1 có RPB (25%)", (bn.loc[NHOM_NHANH[0], "so_case"], bn.loc[NHOM_NHANH[0], "so_case_co_rpb"]) == (4, 1))
    kt("Nhánh (b): 3 case, 1 có RPB", (bn.loc[NHOM_NHANH[1], "so_case"], bn.loc[NHOM_NHANH[1], "so_case_co_rpb"]) == (3, 1))
    kt("Nhánh: tổng = số case", bn["so_case"].sum() == len(pv))

    # 4. tài khoản batch
    tk = tai_khoan_batch(ev, g4.gio_dia_phuong(ev, lambda *a: None)).set_index("tai_khoan")
    kt("Batch: batch_01, batch_02 và user_002", list(tk.index) == ["batch_01", "batch_02", USER_002], list(tk.index))
    kt("Batch: batch_02 2 sự kiện, giờ địa phương 9",
       (tk.loc["batch_02", "so_su_kien"], tk.loc["batch_02", "gio_dia_phuong_pho_bien"]) == (2, 9), tk.loc["batch_02"].to_dict())
    kt("Batch: batch_01 làm PO lúc 9 giờ địa phương",
       tk.loc["batch_01", "hoat_dong_nhieu_nhat"] == f"{PO} (1)" and tk.loc["batch_01", "gio_dia_phuong_pho_bien"] == 9)
    # sự kiện chỉ ghi đến ngày: 23:59 địa phương (22:59 UTC mùa đông) → 00:00 địa phương (23:00 UTC hôm trước), xếp trước GR cùng ngày
    t0 = pd.Timestamp("2018-01-10")
    x = doi_ve_dau_ngay(pd.DataFrame({"case": ["X", "X", "X"], "act": [PO, GR, "Vendor creates invoice"],
                                      "ts": [t0 + pd.Timedelta(hours=8), t0 + pd.Timedelta(hours=10), t0 + pd.Timedelta(hours=22, minutes=59)],
                                      "user": ["u", "u", "NONE"]}))
    kt("Dời sự kiện chỉ ghi đến ngày về 00:00 địa phương",
       list(x["act"]) == ["Vendor creates invoice", PO, GR] and x["ts"].iloc[0] == pd.Timestamp("2018-01-09 23:00"), x.to_dict("list"))
    kt("user_002: 7 sự kiện, 7/8 Clear", (tk.loc[USER_002, "so_su_kien"], tk.loc[USER_002, "so_clear"]) == (7, 7)
       and np.isclose(tk.loc[USER_002, "ty_le_trong_clear"], 7 / 8), tk.loc[USER_002].to_dict())

    # 5. mô hình chuẩn
    pm4py = nap_pm4py()
    if pm4py is None:
        print("Mô hình chuẩn: chưa có pm4py, bỏ qua (pip install pm4py).")
    else:
        net, im, fm = mo_hinh_chuan(pm4py)
        bang_mh = kiem_thu_mo_hinh(net, im, fm)
        print(bang_mh[["vet", "ky_vong", "fitness", "ket_qua"]].to_string(index=False))
        kt("Mô hình chuẩn: 8 vết giả lập", (bang_mh["ket_qua"] == "ĐẠT").all())
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
    ap.add_argument("--ra", default="ket_qua_sq1")
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

    raw = g3.doc_log(a.log)
    moc, _, _ = buoc2_moc_trich_xuat(raw, None)
    pv = loc_pham_vi(raw, a.cong_ty, moc)
    ev = sap_xep(raw[raw["case"].isin(pv)])
    loi = phan_tich(ev, pv, moc, ra, p)
    (ra / "tom_tat_sq1.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {ra.resolve()}. Gửi lại file tom_tat_sq1.txt.")
    if loi:
        sys.exit("Mô hình chuẩn không đạt kiểm thử trên vết giả lập, xem kiem_thu_mo_hinh.csv.")


if __name__ == "__main__":
    main()
