"""
Google Sheets 연동 템플릿
실제 사용 시:
1. Google Cloud Console에서 서비스 계정 생성
2. Sheets API 활성화
3. JSON 키 파일 다운로드
4. 스프레드시트를 서비스 계정 이메일과 공유
5. credentials.json 경로를 환경변수 또는 인자로 전달
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class GoogleSheetsClient:
    def __init__(self, credentials_path: Optional[str] = None):
        if not GSPREAD_AVAILABLE:
            raise ImportError("gspread와 google-auth를 설치하세요: pip install gspread google-auth")

        path = credentials_path or os.getenv("GOOGLE_SHEETS_CREDENTIALS", "credentials.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"인증 파일이 없습니다: {path}\n"
                "Google Cloud 서비스 계정 JSON 키를 준비하세요."
            )
        creds = Credentials.from_service_account_file(path, scopes=SCOPES)
        self.client = gspread.authorize(creds)

    def read_sheet(
        self,
        spreadsheet_id: str,
        worksheet_name: str = "Sheet1",
        range_name: Optional[str] = None,
    ) -> pd.DataFrame:
        sh = self.client.open_by_key(spreadsheet_id)
        ws = sh.worksheet(worksheet_name)
        if range_name:
            data = ws.get(range_name)
        else:
            data = ws.get_all_values()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df

    def write_dataframe(
        self,
        df: pd.DataFrame,
        spreadsheet_id: str,
        worksheet_name: str = "Forecast",
        start_cell: str = "A1",
        clear_first: bool = True,
    ) -> None:
        sh = self.client.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=20)

        if clear_first:
            ws.clear()

        # 헤더 + 데이터
        values = [df.columns.tolist()] + df.astype(str).values.tolist()
        ws.update(start_cell, values)

    def append_rows(
        self,
        rows: List[List[Any]],
        spreadsheet_id: str,
        worksheet_name: str = "Sheet1",
    ) -> None:
        sh = self.client.open_by_key(spreadsheet_id)
        ws = sh.worksheet(worksheet_name)
        ws.append_rows(rows)


def sync_forecast_to_sheets(
    forecast_df: pd.DataFrame,
    spreadsheet_id: str,
    credentials_path: Optional[str] = None,
    worksheet_name: str = "ForecastResult",
) -> str:
    """예측 결과를 시트에 저장"""
    client = GoogleSheetsClient(credentials_path)
    # 필요한 컬럼만
    cols = [c for c in ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"] if c in forecast_df.columns]
    out = forecast_df[cols].copy()
    out["ds"] = out["ds"].astype(str)
    client.write_dataframe(out, spreadsheet_id, worksheet_name)
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"


# ---------- 사용 예시 (주석) ----------
"""
from sheets.google_sheets_sync import GoogleSheetsClient, sync_forecast_to_sheets
from core.forecast_engine import ForecastEngine, generate_sample_sales

# 1) 시트에서 데이터 읽기
client = GoogleSheetsClient("path/to/credentials.json")
df = client.read_sheet("YOUR_SPREADSHEET_ID", "RawData")

# 2) 예측
engine = ForecastEngine()
result = engine.run(df, date_col="date", target_col="sales", periods=90)

# 3) 결과 쓰기
url = sync_forecast_to_sheets(result.forecast_df, "YOUR_SPREADSHEET_ID")
print("결과 시트:", url)
"""
