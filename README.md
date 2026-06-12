# money-back-my-home

**蒙地卡羅美股策略分析 v4(產品化版)**

輸入美股代碼與目標利潤,工具比較「六種進場點位 × 四種出場規則」共 24 種組合,
以**機會成本入帳的帳戶年化**與「單純持有」正面對決,並用三層驗證(全期實測、
walk-forward 樣本外、熊市壓力測試)告訴你結果可不可信 — 包括誠實告訴你
「這檔股票其實不該擇時」。

v4 起支援多使用者(Supabase Auth)、PostgreSQL、Docker 一鍵部署,
月營運成本約 $4(VPS)+ 網域年費;所有外部資料呼叫集中於每日排程,
使用者請求只讀資料庫。

## 本機開發(零設定)

```
pip install -r requirements.txt
python app.py          # 或直接雙擊 start.bat
```

開瀏覽器 → http://127.0.0.1:5000 。預設 `AUTH_MODE=none`(單人模式,不需登入)、
SQLite(`local.db` 自動建立)。網址帶參數:`/?ticker=NVDA&target=20`、
多檔掃描 `/?ticker=AAPL,NVDA&target=15`、探索 `/?discover=1`、追蹤紀錄 `/?track=1`。

## 生產部署(多使用者)

事前準備(皆免費):
1. **Supabase**:建立專案 → 記下 `Settings → API` 的 Project URL、anon key、JWT Secret,
   與 `Settings → Database` 的連線字串(Session pooler)
2. **Tiingo**:到 tiingo.com 免費註冊取得 API key(含息調整資料,主要資料源)
3. **網域**:DNS A 記錄指向你的 VPS(建議掛 Cloudflare)

部署(VPS 上):
```
git clone <repo> && cd <repo>
cp .env.example .env     # 填入上面三項 + AUTH_MODE=supabase + DOMAIN
docker compose up -d     # web(gunicorn ×2)+ scheduler(每日排程)+ caddy(自動 HTTPS)
docker compose exec scheduler python daily_job.py   # 首次手動跑一次,灌入價格資料
```

每日排程於 UTC 22:30(美股收盤後)自動:更新全部標的價格 → 批次結算所有使用者的
追蹤紀錄 → 跑預設參數全市場掃描存檔。白天所有請求只讀 DB,一台 $4 VPS 可服務上千人。

## 架構

| 模組 | 職責 |
|---|---|
| `engine.py` | 蒙地卡羅模擬 + 四種出場 + 條件化 + 信賴區間 + 三層驗證(CLI:`python engine.py NVDA 20`) |
| `rotation.py` | 組合輪動回測(事件驅動,多檔資金再部署) |
| `datasource.py` | 多源資料層:DB 快取 → Tiingo → yfinance → Stooq,容錯 + 節流 + 交叉驗證 |
| `db.py` | SQLAlchemy:SQLite(開發)/ Postgres(生產)雙棲;價格、快取、掃描結果、任務狀態 |
| `tracker.py` | 建議追蹤與自動對答案(per-user,批次結算) |
| `auth.py` | `AUTH_MODE=none / supabase`;Supabase JWT(HS256 或 JWKS)驗證 |
| `app.py` | Flask API;狀態全落 DB,多 gunicorn worker 行為正確;per-user 限流 |
| `daily_job.py` | 每日排程(`--loop` 常駐模式) |
| `universe.py` | 內建約 110 檔掃描股票池(請自行增刪) |
| `static/index.html` | 前端儀表板(Chart.js + supabase-js) |

資料分層原則:使用者紀錄(`recommendations`)是唯一 per-user 的表;
價格(`price_eod`)、分析快取、掃描結果皆全域共享 — 資料源呼叫量 O(標的數)/天,
與使用者數無關。

## 功能總覽

- 六種進場(現價或等回檔 3/5/8/10/15% 掛限價)× 四種出場(固定目標、+停損、
  移動停利、移停+停損),一次全算、前端即時切換
- **帳戶年化**:閒置資金計息的幾何年化,與單純持有同窗口可比;輸基準的組合標灰示警
- **條件化模擬**:依當前市場狀態(200 日均線趨勢 × 21 日波動)抽樣,條件作用前一季後
  回歸無條件;walk-forward 以「逐起點查表」驗證實際使用情境
- **68% 信賴區間**(二階 bootstrap):點估計的誠實模糊度
- **三層驗證**:in-sample 全期實測、walk-forward 樣本外、熊市壓力測試(自動定位歷史最差一年)
- **市場探索**:掃描股票池分三類(擇時有優勢/適合買入持有/避開),入圍者自動精審
  + 信心分級;相關群組偵測(>0.8 視為同一注);¼ Kelly 倉位
- **組合輪動回測**:出場後資金立即輪動的組合級年化(解除單檔的閒置保守假設)
- **建議追蹤(成績單)**:每筆建議自動以收盤資料對答案;跨裝置同步(登入後)
- 厚尾增強、股息預扣稅(預設 30%)、資料健檢警告

## 已知限制(誠實聲明)

bootstrap 假設未來重演過去的分佈,對倖存者偏差無解;回測樣本重疊,有效獨立樣本
遠少於表面數量;walk-forward 偏差(約 6~15pp)才是接近真實的預測誤差;
免費資料源(Tiingo 免費層/yfinance)僅限內部計算用途,正式營運請評估付費資料授權
(計畫:有營收後切換 Polygon/Tiingo 商用方案,改動範圍僅 `datasource.py`)。

## 免責聲明

本工具為歷史資料統計模擬與教育用途,所有數字皆為估計,不代表未來表現,
不構成投資建議;對外營運前請評估當地投資顧問相關法規。
Kelly 倉位為理論上限的四分之一,仍可能高於你的風險承受度。
