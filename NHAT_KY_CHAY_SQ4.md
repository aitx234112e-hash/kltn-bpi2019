# Nhật ký chạy SQ4

| Ngày giờ (giờ Việt Nam) | Lệnh | Thư mục kết quả | Ghi chú |
|---|---|---|---|
| 29/10/2026 | – | – | Ghi thiết kế SQ4 (Mục 3.7.4, Bảng 3.10) |
| 29/10/2026 | `sq4_mo_ta.py` | `ket_qua_sq4_mo_ta` | Phần mô tả trước khi huấn luyện; không huấn luyện mô hình nào |
| 05/11/2026 | – | – | Ghi ba điểm "Cách báo cáo" vào Bảng 3.10 (góp ý G9) |
| 05/11/2026, khoảng 19:52 (file kết quả đầu tiên lúc 19:59) | `sq4_mo_hinh.py --chi-chinh` | `ket_qua_sq4_mo_hinh` | **Lần chạy đầu tiên.** Nhãn chính và nhãn R3, 4 độ dài tiền tố (8 lần huấn luyện); mỗi độ dài tiền tố một mô hình; bootstrap rút lại theo dòng hàng |
| 05/11/2026, 20:31–21:37 | `sq4_mo_hinh.py --ra ket_qua_sq4_mo_hinh_day_du` | `ket_qua_sq4_mo_hinh_day_du` | Đủ cấu hình; máy tắt nguồn sau 34/36 lần huấn luyện; bootstrap theo dòng hàng |
| 05/11/2026, khoảng 22:00 | `sq4_mo_hinh.py --ma nhan_cong_R1 --tien-to 4 5 --bo-tu-kiem --ra ket_qua_sq4_mo_hinh_bo_sung` | `ket_qua_sq4_mo_hinh_bo_sung` | Chạy lại 2 lần còn thiếu; bootstrap theo dòng hàng |
| 06/11/2026 | – | – | Ghi hai điểm thiết kế vào Bảng 3.10 (góp ý G10): mỗi độ dài tiền tố một mô hình (đã đúng ở lần chạy đầu); bootstrap theo đơn mua hàng (điều chỉnh sau lần chạy đầu) |
| 06/11/2026, 14:26–14:46 | `sq4_mo_hinh.py --chi-chinh --ra ket_qua_sq4_mo_hinh_boot_don` | `ket_qua_sq4_mo_hinh_boot_don` | Chạy lại với bootstrap theo đơn mua hàng; ước lượng điểm trùng hoàn toàn lần chạy đầu (chênh lệch lớn nhất 0) |
| 06/11/2026, 22:34 | `sq4_phan_tich_sau.py ket_qua_sq4_mo_hinh_boot_don --log <log>` | (cùng thư mục) | Phân tích sau khi xem kết quả: Holm, tiền tố 5 chưa có Clear thanh toán, PR-AUC kiểm định chéo cạnh tập kiểm tra; không huấn luyện lại |
| 06/11/2026, 22:40–23:02 | `sq4_mo_hinh.py --ra ket_qua_sq4_mo_hinh_day_du_boot_don` | `ket_qua_sq4_mo_hinh_day_du_boot_don` | Đủ cấu hình, bootstrap theo đơn mua hàng; máy tắt nguồn sau 16/36 lần (chinh, R3, dn1, dn2); PR-AUC trùng lần chạy đầu |
| 06/11/2026, 23:24 (35 phút) | `... --ma dn3_loai_huy dn4_H120 dn5_bo_vendor_inv nhan_cong_R5 nhan_cong_R1 --so-luong 2 --bo-tu-kiem --ra ket_qua_sq4_mo_hinh_day_du_boot_don_tiep` | `ket_qua_sq4_mo_hinh_day_du_boot_don_tiep_2luong_khong_dung` | **Không dùng.** Giới hạn 2 luồng CPU làm XGBoost cho kết quả hơi khác, đổi bộ tham số ở 9/20 lần; PR-AUC của XGBoost và Mốc 1 không trùng lần chạy đầu (lệch tối đa 0,017). Chạy lại với đủ luồng |
| 07/11/2026, 00:21 | `... --ma dn3_loai_huy --bo-tu-kiem --ra ket_qua_sq4_mo_hinh_boot_don_dn3` (đủ luồng) | `ket_qua_sq4_mo_hinh_boot_don_dn3` | Chạy lại với đủ luồng; tham số được chọn và PR-AUC trùng hoàn toàn lần chạy đầu (kiểm tra 07/11/2026) |
| 07/11/2026, 00:29 | `... --ma dn4_H120 ...` | `ket_qua_sq4_mo_hinh_boot_don_dn4` | Như trên |
| 07/11/2026, 00:38 | `... --ma dn5_bo_vendor_inv ...` | `ket_qua_sq4_mo_hinh_boot_don_dn5` | Như trên |
| 07/11/2026, 00:52 | `... --ma nhan_cong_R5 ...` | `ket_qua_sq4_mo_hinh_boot_don_R5` | Như trên |
| 07/11/2026, 01:05–11:53 | `... --ma nhan_cong_R1 ...` | `ket_qua_sq4_mo_hinh_boot_don_R1` | Như trên (máy ngủ giữa chừng, chạy tiếp khi máy mở lại) |
| 07/11/2026 | ghép 6 thư mục trên | `ket_qua_sq4_mo_hinh_day_du_boot_don_tong_hop` | Bảng đủ 9 cấu hình × 4 tiền tố, bootstrap theo đơn mua hàng; Holm trong từng nhãn (20 kiểm định). Thư mục lần chạy 2 luồng đổi tên thành `..._tiep_2luong_khong_dung` |

Thời điểm của các lần chạy 05/11 lấy từ thời điểm ghi file trong thư mục kết quả. Từ ngày 06/11/2026, script tự ghi ngày giờ chạy và lệnh vào dòng đầu của file tóm tắt.
