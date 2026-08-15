"""
ForecastMarketer Core Engine
마케팅 매출/지표 시계열 예측 엔진 (Prophet + 보조 모델)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")


@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame
    mae: float
    mape: float
    rmse: float
    last_actual: float
    last_predicted: float
    horizon_days: int


class ForecastEngine:
    """마케팅 데이터에 최적화된 Prophet 기반 예측 엔진"""

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        country_holidays: Optional[str] = "KR",
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.country_holidays = country_holidays

        self.model: Optional[Prophet] = None
        self.data: Optional[pd.DataFrame] = None
        self.forecast: Optional[pd.DataFrame] = None
        self._date_col = "ds"
        self._target_col = "y"

    def load_data(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "sales",
        extra_regressors: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """데이터를 Prophet 형식으로 로드"""
        data = df.copy()
        data[date_col] = pd.to_datetime(data[date_col])
        rename_map = {date_col: "ds", target_col: "y"}
        data = data.rename(columns=rename_map)

        keep_cols = ["ds", "y"]
        if extra_regressors:
            for col in extra_regressors:
                if col in data.columns:
                    keep_cols.append(col)

        self.data = data[keep_cols].sort_values("ds").reset_index(drop=True)
        self.extra_regressors = extra_regressors or []
        return self.data

    def _build_model(self) -> Prophet:
        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        if self.country_holidays:
            try:
                model.add_country_holidays(country_name=self.country_holidays)
            except Exception:
                pass  # 일부 국가 코드 미지원 시 무시

        for reg in getattr(self, "extra_regressors", []):
            model.add_regressor(reg)
        return model

    def train(self) -> None:
        if self.data is None or len(self.data) < 14:
            raise ValueError("학습 데이터가 부족합니다. 최소 14일 이상의 데이터가 필요합니다.")
        self.model = self._build_model()
        self.model.fit(self.data)

    def predict(self, periods: int = 90, freq: str = "D") -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("먼저 train()을 호출하세요.")
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # extra regressor가 있으면 future에 채워야 함 (간단 버전: 마지막 값 유지)
        for reg in getattr(self, "extra_regressors", []):
            if reg in self.data.columns:
                last_val = self.data[reg].iloc[-1]
                future[reg] = last_val

        self.forecast = self.model.predict(future)
        return self.forecast

    def evaluate(self) -> Dict[str, float]:
        if self.forecast is None or self.data is None:
            raise RuntimeError("예측 결과가 없습니다.")
        train_pred = self.forecast.iloc[: len(self.data)]
        y_true = self.data["y"].values
        y_pred = train_pred["yhat"].values

        mae = float(mean_absolute_error(y_true, y_pred))
        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
        except Exception:
            mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        return {"mae": mae, "mape": mape, "rmse": rmse}

    def run(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "sales",
        periods: int = 90,
        extra_regressors: Optional[List[str]] = None,
    ) -> ForecastResult:
        """원스톱 실행: 로드 → 학습 → 예측 → 평가"""
        self.load_data(df, date_col, target_col, extra_regressors)
        self.train()
        forecast_df = self.predict(periods=periods)
        metrics = self.evaluate()

        last_actual = float(self.data["y"].iloc[-1])
        last_predicted = float(forecast_df["yhat"].iloc[-1])

        return ForecastResult(
            forecast_df=forecast_df,
            mae=metrics["mae"],
            mape=metrics["mape"],
            rmse=metrics["rmse"],
            last_actual=last_actual,
            last_predicted=last_predicted,
            horizon_days=periods,
        )

    def scenario(
        self,
        base_forecast: pd.DataFrame,
        growth_rate: float = 0.0,
        cost_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """시나리오 분석 (성장률 / 비용 배수)"""
        future_only = base_forecast.iloc[-90:].copy() if len(base_forecast) > 90 else base_forecast.copy()
        adjusted = future_only["yhat"] * (1 + growth_rate)
        total = float(adjusted.sum())
        avg = float(adjusted.mean())
        return {
            "growth_rate": growth_rate,
            "cost_multiplier": cost_multiplier,
            "total_forecast": total,
            "avg_daily": avg,
            "adjusted_series": adjusted,
        }

    def get_components(self) -> Optional[pd.DataFrame]:
        """트렌드 / 계절성 컴포넌트 반환"""
        if self.model is None or self.forecast is None:
            return None
        return self.forecast[["ds", "trend", "yearly", "weekly", "yhat", "yhat_lower", "yhat_upper"]]


def generate_sample_sales(
    start: str = "2024-01-01",
    periods: int = 500,
    base: float = 3_500_000,
    trend: float = 800,
    noise: float = 250_000,
    seed: int = 42,
) -> pd.DataFrame:
    """마케팅 매출 샘플 데이터 생성"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="D")
    t = np.arange(periods)

    # 트렌드 + 주간/연간 계절성 + 노이즈
    weekly = 180_000 * np.sin(2 * np.pi * t / 7)
    yearly = 400_000 * np.sin(2 * np.pi * t / 365.25)
    sales = base + trend * t + weekly + yearly + rng.normal(0, noise, periods)
    sales = np.maximum(sales, 500_000)  # 음수 방지

    # 마케팅비 (대략 매출의 15~25%)
    ad_spend = sales * rng.uniform(0.15, 0.25, periods)

    return pd.DataFrame(
        {
            "date": dates,
            "sales": sales.round(0).astype(int),
            "ad_spend": ad_spend.round(0).astype(int),
            "orders": (sales / 45_000 + rng.normal(0, 5, periods)).clip(10).round(0).astype(int),
        }
    )
