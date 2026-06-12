# -*- coding: utf-8 -*-
"""內建掃描股票池:2026 年中常見的美股大型股與指數 ETF。

這份清單只是起點,不是推薦;成分會隨時間過時,請自行增刪。
格式:代碼 -> 顯示名稱(粗掃階段批量下載拿不到公司名,精審時會以 Yahoo 名稱覆蓋)。
"""

UNIVERSE = {
    # 科技
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "AVGO": "Broadcom", "ADBE": "Adobe",
    "CRM": "Salesforce", "ORCL": "Oracle", "IBM": "IBM", "CSCO": "Cisco",
    "ACN": "Accenture", "NOW": "ServiceNow", "INTC": "Intel", "AMD": "AMD",
    "QCOM": "Qualcomm", "TXN": "Texas Instruments", "MU": "Micron",
    "AMAT": "Applied Materials", "INTU": "Intuit", "UBER": "Uber",
    "SHOP": "Shopify", "PLTR": "Palantir", "PANW": "Palo Alto Networks",
    "ANET": "Arista Networks",
    # 通訊與媒體
    "NFLX": "Netflix", "DIS": "Disney", "TMUS": "T-Mobile",
    "T": "AT&T", "VZ": "Verizon", "CMCSA": "Comcast",
    # 金融
    "JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "C": "Citigroup",
    "SCHW": "Charles Schwab", "BLK": "BlackRock", "AXP": "American Express",
    "V": "Visa", "MA": "Mastercard", "PYPL": "PayPal", "COIN": "Coinbase",
    "BRK-B": "Berkshire Hathaway",
    # 醫療
    "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "LLY": "Eli Lilly",
    "PFE": "Pfizer", "MRK": "Merck", "ABBV": "AbbVie", "TMO": "Thermo Fisher",
    "ABT": "Abbott", "DHR": "Danaher", "BMY": "Bristol-Myers",
    "AMGN": "Amgen", "GILD": "Gilead", "ISRG": "Intuitive Surgical",
    "MDT": "Medtronic", "CVS": "CVS Health",
    # 消費
    "WMT": "Walmart", "PG": "Procter & Gamble", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "COST": "Costco", "HD": "Home Depot", "MCD": "McDonald's",
    "NKE": "Nike", "SBUX": "Starbucks", "TGT": "Target", "LOW": "Lowe's",
    "BKNG": "Booking", "MAR": "Marriott", "MO": "Altria", "PM": "Philip Morris",
    "CL": "Colgate",
    # 工業與能源
    "XOM": "Exxon Mobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "BA": "Boeing", "CAT": "Caterpillar", "DE": "Deere", "GE": "GE Aerospace",
    "HON": "Honeywell", "LMT": "Lockheed Martin", "RTX": "RTX",
    "UPS": "UPS", "UNP": "Union Pacific", "FDX": "FedEx",
    "NEE": "NextEra", "DUK": "Duke Energy", "SO": "Southern",
    "LIN": "Linde", "FCX": "Freeport-McMoRan", "NEM": "Newmont",
    # 汽車
    "TSLA": "Tesla", "F": "Ford", "GM": "GM",
    # 指數與資產 ETF
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "DIA": "道瓊 ETF",
    "IWM": "羅素 2000 ETF", "VTI": "美國全市場 ETF", "SMH": "半導體 ETF",
    "XLE": "能源類股 ETF", "XLF": "金融類股 ETF", "XLK": "科技類股 ETF",
    "GLD": "黃金 ETF", "SLV": "白銀 ETF", "TLT": "20 年期美債 ETF",
}
