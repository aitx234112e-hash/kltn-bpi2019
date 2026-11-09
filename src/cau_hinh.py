"""Đường dẫn dùng chung cho mọi script.

Log mặc định là du_lieu/BPI_Challenge_2019.xes trong thư mục gốc. Đặt biến môi trường KLTN_LOG để dùng file khác,
hoặc ghi đường dẫn log ngay sau tên script khi chạy.
"""
import os
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
LOG = Path(os.environ.get("KLTN_LOG", GOC / "du_lieu" / "BPI_Challenge_2019.xes"))
