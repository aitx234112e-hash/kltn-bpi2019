# Mã nguồn khóa luận: Process Mining cho quy trình Mua hàng – Thanh toán (BPI Challenge 2019)

Sinh viên: Trần Xuân Ái (UEL)

Thư mục này chứa toàn bộ mã nguồn dùng để tính các con số trong Chương 3 và Chương 4 (Mục 4.1–4.4), kèm các file kết quả tóm tắt và file đọc tay của các mẫu kiểm chứng.

## 1. Dữ liệu

Log BPI Challenge 2019 (van Dongen, 2019), DOI: 10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1. File log không kèm theo vì dung lượng lớn (khoảng 700 MB). Tải `BPI_Challenge_2019.xes` rồi đặt vào thư mục `du_lieu/`.

Nếu để log ở chỗ khác, có hai cách:

- đặt biến môi trường `KLTN_LOG` trỏ tới file log, hoặc
- ghi đường dẫn log ngay sau tên script khi chạy, ví dụ `python src/sq2_tuan_thu.py D:\du_lieu\BPI_Challenge_2019.xes`.

Đường dẫn mặc định nằm trong `src/cau_hinh.py`.

## 2. Cài đặt

```
pip install -r requirements.txt
```

Graphviz (lệnh `dot` có trong PATH) chỉ cần để vẽ hình PNG; thiếu Graphviz thì các script vẫn chạy và ghi file `.dot`. Phiên bản thư viện đã dùng khi chạy được ghi trong `requirements_da_chay.txt`.

## 3. Cách chạy

Mọi lệnh chạy từ thư mục gốc này (thư mục chứa README). Kết quả của mỗi script ghi vào một thư mục `ket_qua_...` ở thư mục gốc. Script nào có `--tu-kiem` thì nên chạy lệnh đó trước: script kiểm thử các hàm bằng vết giả lập và chỉ chạy trên log thật khi tất cả đều đạt.

Các script dùng chung hàm của nhau, nên chạy theo đúng thứ tự dưới đây. Một số script đọc lại thư mục kết quả của script trước (ví dụ mẫu đã đọc tay để không rút lại case đã đọc, hoặc file mô hình chuẩn `.pnml`).

| Bước | Lệnh | Nội dung | Dùng ở |
|---:|---|---|---|
| 1 | `python src/kiem_tra_trang_thai_case.py` | Đọc log; 42 hoạt động; chọn thời điểm trích xuất; các case biên | Mục 3.1.4 |
| 2 | `python src/bang3_pheu_loc.py` | Phễu lọc dữ liệu | Bảng 3.4 |
| 3 | `python src/r5_v2_phan_loai_clear.py --delta-phut 60 --ra ket_qua_r5_v2_d60` | Phân loại Clear Invoice thành thanh toán và bù trừ do hủy; bốn luồng đối chiếu | Mục 3.2.1, Bảng 3.2 |
| 4 | `python src/g3_kiem_tra_bo_sung.py` | Chọn Δ (Hình 3.1); Clear khi không còn chứng từ phải trả; độ chi tiết timestamp | Mục 3.1.6, 3.2.1 |
| 5 | `python src/g4_kiem_tra.py` | R5 v2.2, debit memo, R4 gộp vào R5; múi giờ; chỉ báo làm lại (dùng `r4_thanh_phan.py`) | Mục 3.4.2, 3.5, 3.6 |
| 6 | `python src/g5_kiem_tra.py` | Định nghĩa trạng thái hoàn tất; thời gian thanh toán; phân loại R5 theo đơn mua hàng | Mục 3.2.2, 3.5 |
| 7 | `python src/kiem_chung_cuoi.py` | Bộ luật R1–R7b; rút mẫu kiểm chứng cuối (seed 2026) | Mục 3.4, Bảng 3.6, 3.8 |
| 8 | `python src/dem_su_kien_trang_thai.py` | Số dòng hàng và số sự kiện theo trạng thái | Mục 3.2.2 |
| 9 | `python src/sq1_kham_pha.py` | Tần suất hoạt động, DFG, biến thể, hai nhánh, tài khoản batch; dựng và kiểm thử mô hình chuẩn `.pnml` | Mục 3.3, 4.1; Bảng 3.5, 3.7, 4.1, 4.2, 4.4 |
| 10 | `python src/sq1_imf.py` | Inductive Miner (IMf) so với mô hình chuẩn | Mục 4.1.4, Bảng 4.3 |
| 11 | `python src/sq2_tuan_thu.py` | Tỷ lệ dấu hiệu theo luật, theo hạng mục, nhà cung cấp; giá trị; alignment | Mục 4.2, Bảng 4.6–4.15 |
| 12 | `python src/sq2_bo_sung.py` | R6 hai nhóm; R7 dạng khoảng; nhận hàng và ghi nhận hóa đơn nhiều lần; R1 trên dòng không hủy | Mục 4.1.6, 4.2.1, Bảng 4.5 |
| 13 | `python src/kiem_chung_bo_sung.py` | Đối chiếu mẫu kiểm chứng; hiệu ứng Mann-Whitney của giá trị; cơ chế R3; độ nhạy Δ; alignment trên 100% biến thể | Mục 3.4.4, 3.4.6, 4.1.4, 4.1.7, 4.2 |
| 14 | `jupyter notebook 02_kham_pha.ipynb` | Tái hiện các bảng và hình của Mục 4.1 | Mục 4.1 |
| 15 | `python src/sq3_thoi_gian.py` | Thời gian thanh toán: Kaplan-Meier, Cox có biến thay đổi theo thời gian, mốc ngày 30, Schoenfeld, RMST, ba phân tích độ nhạy | Mục 3.7.3, 4.3; Bảng 4.16–4.20, 4.22; Hình 4.3–4.5 |
| 16 | `python src/sq3_kiem_tra_co_che.py` | Từ dấu hiệu đến thanh toán; chỉ báo làm lại và định nghĩa trạng thái; mốc thời gian của biến thể phụ R7 | Mục 4.3.5, Bảng 4.21 |
| 17 | `jupyter notebook 04_thoi_gian.ipynb` | Tái hiện bước 15 và 16 trong một notebook | Mục 4.3 |
| 18 | `python src/g8_kiem_tra.py` | Mẫu bổ sung R6 và phép đếm R3; R6 và nhà cung cấp (Cox phân tầng); debit memo và làm lại; R7 biến thể phụ với mốc chung; mô hình chỉ dùng biến biết lúc tạo; vẽ Hình 4.4 | Mục 3.5, 4.2.1, 4.3; Bảng 3.12, 4.22, 4.23; Hình 4.4 |
| 19 | `python src/sq4_mo_ta.py` | SQ4, phần mô tả trước khi huấn luyện: dấu hiệu trong 90 và 120 ngày; cỡ tập huấn luyện, kiểm tra và số dòng nhãn 1 theo độ dài tiền tố và theo luật; nhãn 1 theo tháng tạo | Mục 3.7.4, 4.4.1; Bảng 3.10, 4.24–4.27 |
| 20 | `python src/sq4_mo_hinh.py --tu-kiem`, rồi `python src/sq4_mo_hinh.py --chi-chinh --ra ket_qua_sq4_mo_hinh_boot_don` (nhãn chính và nhãn R3) và `python src/sq4_mo_hinh.py --ra ket_qua_sq4_mo_hinh_day_du_boot_don` (đủ 9 cấu hình × 4 độ dài tiền tố) | SQ4, huấn luyện và đánh giá: Logistic Regression, XGBoost và hai mốc so sánh, mỗi độ dài tiền tố 2–5 một mô hình; PR-AUC kèm KTC bootstrap theo đơn mua hàng, so sánh bootstrap ghép cặp, hiệu chỉnh Holm trong từng nhãn, Precision@k, lift; PR-AUC theo tháng và theo luật; SHAP; năm độ nhạy và hai nhãn độ nhạy | Mục 3.7.4, 4.4; Bảng 3.10 |
| 21 | `python src/sq4_phan_tich_sau.py ket_qua_sq4_mo_hinh_boot_don` | SQ4, phân tích sau khi xem kết quả, không huấn luyện lại: Holm trong từng nhãn; tiền tố 5 tách theo đã có hay chưa có Clear Invoice thanh toán; PR-AUC kiểm định chéo cạnh PR-AUC tập kiểm tra | Mục 4.4 |

`r4_thanh_phan.py` là module dùng trong bước 5; có thể chạy riêng bằng `python src/r4_thanh_phan.py du_lieu/BPI_Challenge_2019.xes`.

Bước 10 và 11 cần pm4py; bước 13 cần thêm scipy; bước 15 đến 18 cần lifelines. Bước 13 lâu nhất vì tính alignment cho mọi biến thể; thêm `--bo-alignment` để bỏ phần này. Bước 15 có `--tu-kiem` (kiểm thử bằng log giả lập, gồm mô phỏng thiên lệch thời gian bất tử) ; trên máy đã dùng, bước 15 chạy khoảng 50 phút và notebook ở bước 17 lâu hơn một chút vì làm cả bước 16.

Bước 20 cần thêm scikit-learn và xgboost (shap tùy chọn, chỉ để vẽ hình beeswarm). Script dùng lại các hàm nhãn, chia tập và tiền tố của `sq4_mo_ta.py`, nên số dòng hàng và nhãn 1 khớp Bảng 4.24–4.27. Trên máy đã dùng, `--chi-chinh` chạy khoảng 25 phút và bản đủ 36 cấu hình khoảng 2,5 giờ. File tóm tắt được ghi sau mỗi tiền tố, và dòng đầu file ghi ngày giờ chạy và lệnh đã dùng. Nếu lần chạy bị ngắt, chạy lại phần còn thiếu bằng `--ma <mã cấu hình> --tien-to <độ dài> --bo-tu-kiem --ra <thư mục mới>`. Không dùng `--so-luong` khi cần lặp lại đúng kết quả: đổi số luồng CPU làm XGBoost cho điểm hơi khác và có thể chọn bộ tham số khác.

Bước 21 đọc file điểm dự báo `diem_kiem_tra_*.csv` do bước 20 ghi. Các file này lớn nên không kèm theo; cần chạy bước 20 (`--chi-chinh --ra ket_qua_sq4_mo_hinh_boot_don`) trước.

## 4. Kết quả kèm theo

Các thư mục `ket_qua_...` đi kèm chỉ giữ file tóm tắt (`tom_tat_*.txt`), các bảng nhỏ, file mô hình `.pnml`, các mẫu kiểm chứng và file nhận xét đọc tay (`*_nhan_xet.csv`). Các file chi tiết theo từng dòng hàng (vài chục MB) không kèm theo; chúng được tạo lại khi chạy script.

Mẫu kiểm chứng cuối được rút bằng `kiem_chung_cuoi.py` ngày 18/10/2026, seed 2026, sau khi chốt định nghĩa luật. Khung mẫu của mỗi luật là toàn bộ dòng hàng bị luật đánh dấu (hoặc không bị đánh dấu nhưng có hoạt động liên quan), trừ các case đã đọc ở những lần kiểm tra trước. `kiem_chung_bo_sung.py` rút lại các mẫu này bằng cùng code và seed, và đối chiếu với file đã đọc tay.

Trong `ket_qua_kiem_chung_bo_sung`, file `mau_r6_khac_thoi_diem.csv` rỗng do lỗi ở lần chạy cũ và không được dùng. Mẫu 20 vết R6 khác thời điểm đã đọc tay nằm ở `ket_qua_g8/mau_r6_khac_thoi_diem_g8_nhan_xet.csv`.

### Kết quả SQ4 và ngày chạy

Chi tiết từng lần chạy (lệnh, thời gian, ghi chú) nằm trong `NHAT_KY_CHAY_SQ4.md`. Kết quả dùng trong khóa luận là bản bootstrap theo đơn mua hàng:

| Thư mục | Ngày chạy | Nội dung |
|---|---|---|
| `ket_qua_sq4_mo_hinh_day_du_boot_don_tong_hop` | 06/11 và 07/11/2026 | **Bảng dùng trong khóa luận**: đủ 9 cấu hình × 4 độ dài tiền tố, bootstrap theo đơn mua hàng, cột `p_holm` (Holm trong từng nhãn, 20 kiểm định). Ghép từ sáu thư mục dưới |
| `ket_qua_sq4_mo_hinh_day_du_boot_don` | 06/11/2026, 22:40 | Bốn cấu hình đầu (chinh, R3, dn1, dn2); máy tắt sau 16/36 lần huấn luyện |
| `ket_qua_sq4_mo_hinh_boot_don_dn3`, `_dn4`, `_dn5`, `_R5`, `_R1` | 07/11/2026, 00:21–11:53 | Năm cấu hình còn lại, mỗi lần một cấu hình (`--ma <mã>`), đủ luồng CPU |
| `ket_qua_sq4_mo_hinh_boot_don` | 06/11/2026, 14:26–14:46 | Nhãn chính và nhãn R3 (`--chi-chinh`), kèm SHAP và kết quả bước 21 (`phan_tich_sau.txt`, chạy 06/11/2026, 22:34) |

Bootstrap theo đơn mua hàng chỉ thay phần đánh giá: tham số được chọn và PR-AUC của mọi cấu hình trùng hoàn toàn với lần chạy đầu. Kết quả lần chạy đầu (bootstrap theo dòng hàng, 05/11/2026) giữ cho phụ lục: `ket_qua_sq4_mo_hinh` (`--chi-chinh`), `ket_qua_sq4_mo_hinh_day_du` (34/36 lần huấn luyện trước khi máy tắt) và `ket_qua_sq4_mo_hinh_bo_sung` (hai lần còn thiếu, `--ma nhan_cong_R1 --tien-to 4 5`). Lần chạy ngày 06/11/2026, 23:24 với `--so-luong 2` không dùng và không kèm theo (lý do trong `NHAT_KY_CHAY_SQ4.md`).

Các file SHAP dạng `.npy` và file điểm dự báo `diem_kiem_tra_*.csv` không kèm theo vì dung lượng lớn; hình `.png` và `shap.csv` có kèm.

## 5. Tên module khi đưa vào phụ lục

Tên file hiện tại theo thứ tự các lần kiểm tra. Khi đưa vào phụ lục khóa luận, các module sẽ được đổi tên theo nội dung:

| Tên hiện tại | Tên trong phụ lục |
|---|---|
| kiem_tra_trang_thai_case.py | 01_doc_log_va_trang_thai.py |
| bang3_pheu_loc.py | 02_pheu_loc.py |
| r5_v2_phan_loai_clear.py | 03_phan_loai_clear.py |
| g3_kiem_tra_bo_sung.py | 04_chon_delta.py |
| r4_thanh_phan.py | 05_thanh_phan_r4.py |
| g4_kiem_tra.py | 06_r5_debit_memo.py |
| g5_kiem_tra.py | 07_trang_thai_hoan_tat.py |
| kiem_chung_cuoi.py | 08_bo_luat_kiem_chung.py |
| dem_su_kien_trang_thai.py | 09_dem_su_kien.py |
| sq1_kham_pha.py | 10_kham_pha_quy_trinh.py |
| sq1_imf.py | 11_inductive_miner.py |
| sq2_tuan_thu.py | 12_kiem_tra_tuan_thu.py |
| sq2_bo_sung.py | 13_tuan_thu_bo_sung.py |
| kiem_chung_bo_sung.py | 14_kiem_chung_bo_sung.py |
| sq3_thoi_gian.py | 15_thoi_gian_thanh_toan.py |
| sq3_kiem_tra_co_che.py | 16_kiem_tra_co_che.py |
| g8_kiem_tra.py | 17_do_nhay_sq3.py |
| sq4_mo_ta.py | 18_mo_ta_sq4.py |
| sq4_mo_hinh.py | 19_du_bao_sq4.py |
| sq4_phan_tich_sau.py | 20_phan_tich_sau_sq4.py |
