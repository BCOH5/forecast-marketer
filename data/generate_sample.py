"""샘플 마케팅 데이터 CSV 생성"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.forecast_engine import generate_sample_sales

df = generate_sample_sales(start="2024-01-01", periods=550)
out_path = os.path.join(os.path.dirname(__file__), "sample_sales.csv")
df.to_csv(out_path, index=False)
print(f"Saved: {out_path} ({len(df)} rows)")
print(df.head())
print(df.tail())
