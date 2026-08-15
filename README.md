# ForecastMarketer

마케팅에서 바로 쓸 수 있는 **포캐스팅 에이전트 AI** 통합 패키지입니다.

포함 기능:
1. **매출/지표 시계열 예측** (Prophet 기반)
2. **채널별 ROI / ROAS / CPA 예측**
3. **예산 자동 최적화**
4. **자연어 에이전트** (키워드 라우팅 + LangChain 옵션)
5. **Streamlit 웹 대시보드**
6. **Google Sheets 연동 템플릿**

---

## 빠른 시작

```bash
cd forecast_marketer
pip install -r requirements.txt

# 샘플 데이터 이미 생성됨: data/sample_sales.csv

# 1) 코어 엔진 테스트
python -c "
from core import ForecastEngine, generate_sample_sales
df = generate_sample_sales()
r = ForecastEngine().run(df, periods=60)
print(f'MAPE: {r.mape:.1f}% | 60일 후 예측: {r.last_predicted:,.0f}')
"

# 2) 자연어 에이전트
python agents/langchain_agent.py

# 3) 채널 ROI 데모
python -c "from roi.channel_roi import quick_demo; quick_demo(30_000_000)"

# 4) Streamlit 대시보드
streamlit run dashboard/app.py
```

---

## 폴더 구조

```
forecast_marketer/
├── core/
│   └── forecast_engine.py    # Prophet 예측 엔진
├── roi/
│   └── channel_roi.py        # 채널 ROI & 예산 최적화
├── agents/
│   └── langchain_agent.py    # 자연어 에이전트 (+ LangChain 옵션)
├── dashboard/
│   └── app.py                # Streamlit 대시보드
├── sheets/
│   └── google_sheets_sync.py # Google Sheets 연동 템플릿
├── data/
│   ├── sample_sales.csv
│   └── generate_sample.py
├── requirements.txt
└── README.md
```

---

## 사용 예시

### 매출 예측
```python
from core import ForecastEngine
import pandas as pd

df = pd.read_csv("data/sample_sales.csv")
engine = ForecastEngine(country_holidays="KR")
result = engine.run(df, date_col="date", target_col="sales", periods=90)

print(result.mape, result.last_predicted)
# result.forecast_df 에 전체 예측값
```

### 채널 ROI
```python
from roi import ChannelROIPredictor

pred = ChannelROIPredictor()
df = pred.predict_mix({
    "Meta": 10_000_000,
    "Google": 8_000_000,
    "TikTok": 7_000_000,
})
print(df)

# 예산 최적화
opt = pred.optimize_budget(30_000_000, min_roas=2.0)
print(opt["recommendation"])
print(opt["allocation"])
```

### 자연어 에이전트
```python
from agents import ForecastMarketerAgent

agent = ForecastMarketerAgent()
print(agent.handle("다음 90일 매출 예측해줘"))
print(agent.handle("30000000 예산 최적화해줘"))
print(agent.handle("시나리오 분석해줘"))
```

### LangChain + OpenAI (선택)
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
from agents import create_langchain_agent

executor = create_langchain_agent()
executor.invoke({"input": "다음 60일 매출 예측하고, 2천만원으로 예산 최적화해줘"})
```

### Google Sheets
1. Google Cloud에서 서비스 계정 생성 → JSON 키 다운로드
2. Sheets API 활성화
3. 스프레드시트를 서비스 계정 이메일에 공유
4. `sheets/google_sheets_sync.py` 참고

---

## 주의사항

- Prophet은 처음 설치 시 시간이 걸릴 수 있습니다.
- 실제 광고 API(Meta/Google) 연동은 별도 토큰이 필요합니다. 현재는 시뮬레이션 모델입니다.
- Google Sheets는 `credentials.json`이 있어야 동작합니다.
- Streamlit은 로컬에서 `streamlit run dashboard/app.py`로 실행하세요.

---

## 라이선스

개인/팀 마케팅 실무 사용 가능. 자유롭게 수정하세요.
