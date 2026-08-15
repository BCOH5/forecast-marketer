"""
마케팅 채널별 ROI / ROAS / CPA 예측기
Meta, Google, TikTok, Naver 등 채널별 성과 시뮬레이션 및 예산 최적화
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


@dataclass
class ChannelConfig:
    name: str
    cpc: float                  # 클릭당 비용
    ctr: float                  # 클릭률 (0~1)
    cvr: float                  # 전환율 (0~1)
    aov: float                  # 평균 주문 금액
    min_budget: float = 100_000
    max_budget: float = 50_000_000
    diminishing_factor: float = 0.85  # 예산 증가 시 효율 감소


@dataclass
class ChannelResult:
    channel: str
    budget: float
    impressions: float
    clicks: float
    conversions: float
    revenue: float
    cost: float
    roas: float
    cpa: float
    roi: float


class ChannelROIPredictor:
    """채널별 마케팅 ROI 예측 및 예산 배분 최적화"""

    DEFAULT_CHANNELS = {
        "Meta": ChannelConfig("Meta", cpc=420, ctr=0.018, cvr=0.025, aov=48_000),
        "Google": ChannelConfig("Google", cpc=680, ctr=0.032, cvr=0.035, aov=52_000),
        "TikTok": ChannelConfig("TikTok", cpc=310, ctr=0.025, cvr=0.018, aov=38_000),
        "Naver": ChannelConfig("Naver", cpc=550, ctr=0.022, cvr=0.028, aov=45_000),
        "YouTube": ChannelConfig("YouTube", cpc=480, ctr=0.015, cvr=0.022, aov=50_000),
    }

    def __init__(self, channels: Optional[Dict[str, ChannelConfig]] = None):
        self.channels = channels or self.DEFAULT_CHANNELS.copy()

    def predict_channel(
        self,
        channel_name: str,
        budget: float,
        custom: Optional[Dict[str, float]] = None,
    ) -> ChannelResult:
        """단일 채널 성과 예측 (예산 증가에 따른 체감 효율 반영)"""
        if channel_name not in self.channels:
            raise ValueError(f"알 수 없는 채널: {channel_name}")

        cfg = self.channels[channel_name]
        cpc = custom.get("cpc", cfg.cpc) if custom else cfg.cpc
        ctr = custom.get("ctr", cfg.ctr) if custom else cfg.ctr
        cvr = custom.get("cvr", cfg.cvr) if custom else cfg.cvr
        aov = custom.get("aov", cfg.aov) if custom else cfg.aov

        # 예산이 커질수록 효율이 떨어지는 간단한 diminishing returns 모델
        scale = (budget / 1_000_000) ** 0.5
        efficiency = cfg.diminishing_factor ** max(0, scale - 1)

        effective_cpc = cpc / efficiency
        clicks = budget / effective_cpc
        impressions = clicks / max(ctr, 1e-6)
        conversions = clicks * cvr
        revenue = conversions * aov
        cost = budget
        roas = revenue / cost if cost > 0 else 0
        cpa = cost / conversions if conversions > 0 else float("inf")
        roi = (revenue - cost) / cost if cost > 0 else 0

        return ChannelResult(
            channel=channel_name,
            budget=budget,
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
            revenue=revenue,
            cost=cost,
            roas=roas,
            cpa=cpa,
            roi=roi,
        )

    def predict_mix(
        self,
        budget_allocation: Dict[str, float],
        custom_params: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """여러 채널 혼합 예산 배분 예측"""
        results = []
        for name, budget in budget_allocation.items():
            custom = (custom_params or {}).get(name)
            r = self.predict_channel(name, budget, custom)
            results.append({
                "channel": r.channel,
                "budget": r.budget,
                "impressions": round(r.impressions),
                "clicks": round(r.clicks, 1),
                "conversions": round(r.conversions, 1),
                "revenue": round(r.revenue),
                "cost": round(r.cost),
                "roas": round(r.roas, 2),
                "cpa": round(r.cpa, 0) if r.cpa != float("inf") else None,
                "roi": round(r.roi, 3),
            })
        df = pd.DataFrame(results)
        # 합계 행
        total = {
            "channel": "TOTAL",
            "budget": df["budget"].sum(),
            "impressions": df["impressions"].sum(),
            "clicks": df["clicks"].sum(),
            "conversions": df["conversions"].sum(),
            "revenue": df["revenue"].sum(),
            "cost": df["cost"].sum(),
            "roas": df["revenue"].sum() / df["cost"].sum() if df["cost"].sum() > 0 else 0,
            "cpa": df["cost"].sum() / df["conversions"].sum() if df["conversions"].sum() > 0 else None,
            "roi": (df["revenue"].sum() - df["cost"].sum()) / df["cost"].sum() if df["cost"].sum() > 0 else 0,
        }
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
        return df

    def optimize_budget(
        self,
        total_budget: float,
        channels: Optional[List[str]] = None,
        min_roas: float = 2.0,
        steps: int = 20,
    ) -> Dict[str, Any]:
        """
        간단한 그리드 서치 기반 예산 최적화
        (실제 프로덕션에서는 더 정교한 최적화 알고리즘 권장)
        """
        target_channels = channels or list(self.channels.keys())
        n = len(target_channels)
        if n == 0:
            return {"error": "채널이 없습니다."}

        best_alloc = None
        best_score = -np.inf
        best_df = None

        # 균등 배분부터 시작해서 비율을 조금씩 조정하는 간단한 탐색
        ratios = np.linspace(0.05, 0.7, steps)

        # 기본 균등
        equal = {ch: total_budget / n for ch in target_channels}
        df_eq = self.predict_mix(equal)
        score_eq = self._score(df_eq, min_roas)
        best_alloc, best_score, best_df = equal, score_eq, df_eq

        # 한 채널에 더 많이 배분하는 시나리오
        for i, ch in enumerate(target_channels):
            for r in ratios:
                alloc = {c: total_budget * (1 - r) / (n - 1) for c in target_channels if c != ch}
                alloc[ch] = total_budget * r
                df = self.predict_mix(alloc)
                score = self._score(df, min_roas)
                if score > best_score:
                    best_score = score
                    best_alloc = alloc
                    best_df = df

        return {
            "total_budget": total_budget,
            "allocation": {k: round(v) for k, v in best_alloc.items()},
            "result_table": best_df,
            "score": best_score,
            "recommendation": self._make_recommendation(best_df),
        }

    def _score(self, df: pd.DataFrame, min_roas: float) -> float:
        total = df[df["channel"] == "TOTAL"].iloc[0]
        if total["roas"] < min_roas:
            return total["revenue"] * 0.3  # 페널티
        return total["revenue"] - total["cost"] * 0.2

    def _make_recommendation(self, df: pd.DataFrame) -> str:
        total = df[df["channel"] == "TOTAL"].iloc[0]
        best = df[df["channel"] != "TOTAL"].sort_values("roas", ascending=False).iloc[0]
        return (
            f"총 예산 {total['budget']:,.0f}원 기준 예상 ROAS {total['roas']:.2f}, "
            f"수익 {total['revenue']:,.0f}원. "
            f"가장 효율적인 채널은 {best['channel']} (ROAS {best['roas']:.2f})입니다."
        )


def quick_demo(total_budget: float = 30_000_000) -> None:
    predictor = ChannelROIPredictor()
    result = predictor.optimize_budget(total_budget)
    print("=== 예산 최적화 결과 ===")
    print(result["recommendation"])
    print("\n배분:")
    for k, v in result["allocation"].items():
        print(f"  {k}: {v:,}원")
    print("\n상세:")
    print(result["result_table"].to_string(index=False))
