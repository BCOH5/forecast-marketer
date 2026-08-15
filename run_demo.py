#!/usr/bin/env python3
"""ForecastMarketer 통합 데모 실행"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import ForecastEngine, generate_sample_sales
from roi.channel_roi import ChannelROIPredictor, quick_demo
from agents import ForecastMarketerAgent


def main():
    print("=" * 60)
    print("  ForecastMarketer — 마케팅 포캐스팅 에이전트 데모")
    print("=" * 60)

    # 1. 매출 예측
    print("\n[1] 매출 시계열 예측 (샘플 200일 → 향후 45일)")
    df = generate_sample_sales(periods=200)
    engine = ForecastEngine(yearly_seasonality=False, country_holidays="KR")
    result = engine.run(df, periods=45)
    print(f"  MAPE: {result.mape:.1f}%")
    print(f"  최근 실제: {result.last_actual:,.0f}원")
    print(f"  45일 후 예측: {result.last_predicted:,.0f}원")
    future_sum = result.forecast_df.tail(45)["yhat"].sum()
    print(f"  향후 45일 합계: {future_sum:,.0f}원")

    # 2. 채널 ROI
    print("\n[2] 채널 ROI 예측")
    pred = ChannelROIPredictor()
    mix = pred.predict_mix({
        "Meta": 8_000_000,
        "Google": 7_000_000,
        "TikTok": 5_000_000,
        "Naver": 5_000_000,
    })
    total = mix[mix["channel"] == "TOTAL"].iloc[0]
    print(f"  총 예산: {total['budget']:,.0f}원 → 예상 매출 {total['revenue']:,.0f}원")
    print(f"  ROAS: {total['roas']:.2f} | ROI: {total['roi']:.1%}")

    # 3. 예산 최적화
    print("\n[3] 예산 최적화 (2,500만원)")
    opt = pred.optimize_budget(25_000_000, min_roas=2.0)
    print(f"  {opt['recommendation']}")

    # 4. 자연어 에이전트
    print("\n[4] 자연어 에이전트")
    agent = ForecastMarketerAgent()
    print(agent.handle("시나리오 분석해줘"))

    print("\n" + "=" * 60)
    print("대시보드 실행: streamlit run dashboard/app.py")
    print("자세한 사용법: README.md 참고")
    print("=" * 60)


if __name__ == "__main__":
    main()
