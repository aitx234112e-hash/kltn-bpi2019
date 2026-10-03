# KLTN – Process Mining và học máy cho quy trình Mua hàng – Thanh toán (BPI Challenge 2019)

Sinh viên: Trần Xuân Ái (UEL)

## Dữ liệu
Log BPI Challenge 2019 (van Dongen, 2019), DOI: 10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1.
File log không đưa lên repo do dung lượng lớn. Tải file `BPI_Challenge_2019.xes` về và đặt vào thư mục `data/`.

## Cài đặt
```
pip install -r requirements.txt
```

## Chạy
| Lệnh | Mục đích | Kết quả |
|---|---|---|
| `python src/kiem_tra_trang_thai_case.py data/BPI_Challenge_2019.xes` | Kiểm tra định nghĩa trạng thái dòng hàng (Mục 5.2): 42 hoạt động, thời điểm trích xuất, so sánh biến thể quy tắc, case biên | thư mục `ket_qua_5_2/` |
| `python src/bang3_pheu_loc.py data/BPI_Challenge_2019.xes` | Bảng 3 – phễu lọc dữ liệu và số case theo trạng thái | `bang3_pheu_loc.csv` |


## Kết quả đã chạy (03/10/2026)
- `ket_qua/tom_tat.txt`: tóm tắt đầu ra của `kiem_tra_trang_thai_case.py`
- `ket_qua/bang3_pheu_loc.csv`: Bảng 3 (251.734 → 210.887 dòng hàng; hoàn tất 164.124, còn mở 39.496, bị xóa 7.267)
