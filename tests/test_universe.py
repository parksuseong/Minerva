from minerva.strategy import universe


def test_kospi_skips_etf_and_preferred():
    stocks = [
        {"itemCode": "005930", "stockName": "삼성전자", "stockEndType": "stock"},
        {"itemCode": "005935", "stockName": "삼성전자우", "stockEndType": "stock"},
        {"itemCode": "069500", "stockName": "KODEX 200", "stockEndType": "etf"},
        {"itemCode": "000660", "stockName": "SK하이닉스", "stockEndType": "stock"},
    ]
    assert universe.pick_kospi(stocks, 5) == [
        ("005930.KS", "삼성전자"),
        ("000660.KS", "SK하이닉스"),
    ]


def test_nasdaq_dedupes_share_classes():
    stocks = [
        {"symbolCode": "GOOGL", "stockName": "알파벳 Class A", "stockEndType": "stock"},
        {"symbolCode": "GOOG", "stockName": "알파벳 Class C", "stockEndType": "stock"},
        {"symbolCode": "MSFT", "stockName": "마이크로소프트", "stockEndType": "stock"},
    ]
    assert universe.pick_nasdaq(stocks, 2) == [("GOOGL", "알파벳"), ("MSFT", "마이크로소프트")]


def test_falls_back_when_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "CACHE_DIR", tmp_path)

    def boom(url):
        raise OSError("offline")

    monkeypatch.setattr(universe, "_get_json", boom)
    assert universe.kospi_top(50) == universe.FALLBACK_KOSPI


def test_kospi_top_reads_several_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "CACHE_DIR", tmp_path)
    pages = {
        1: [
            {"itemCode": "000001", "stockName": "A", "stockEndType": "stock"},
            {"itemCode": "069500", "stockName": "KODEX 200", "stockEndType": "etf"},
        ],
        2: [{"itemCode": "000002", "stockName": "B", "stockEndType": "stock"}],
        3: [{"itemCode": "000003", "stockName": "C", "stockEndType": "stock"}],
    }
    seen = []

    def fake(url):
        page = int(url.rsplit("=", 1)[1])
        seen.append(page)
        return {"stocks": pages.get(page, [])}

    monkeypatch.setattr(universe, "_get_json", fake)
    assert universe.kospi_top(2) == [("000001.KS", "A"), ("000002.KS", "B")]
    assert seen == [1, 2]  # 2종목이 차면 더 받지 않는다


def test_kospi_dedupes_stock_seen_on_two_pages():
    s = {"itemCode": "000001", "stockName": "A", "stockEndType": "stock"}
    b = {"itemCode": "000002", "stockName": "B", "stockEndType": "stock"}
    assert universe.pick_kospi([s, s, b], 5) == [("000001.KS", "A"), ("000002.KS", "B")]
