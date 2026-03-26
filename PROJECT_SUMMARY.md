# AI Trading Simulator - Kisa Ozet

## Ne Yapiyor?
- Brent Petrol (`BZ=F`) fiyatini izleyip **paper trading** (hayali alim/satim) yapar.
- Islem kararini haber analizi + teknik sinyaller ile verir.
- Tum cuzdan ve islem gecmisini `wallet_state.json` dosyasinda tutar.

## Neye Bakiyor?
- **Fiyat:** `yfinance` ile Brent fiyat verisi.
- **Haber:** RSS/Google News sorgulari (Iran, OPEC, ABD stok verileri, Orta Dogu gerilimleri, uluslararasi enerji haberleri).
- **Sinyal:** Haber duygu puani (`AL / SAT / TUT`) + EMA/RSI/volatilite temelli teknik zamanlama.

## Ozelikleri
- Haberleri kaynak guvenine gore agirliklandirir (daha guvenilir kaynak daha etkili).
- Ayni haber birden fazla kaynaktan gelirse ekstra dogrulama puani verir.
- Haber tazeligine (recency) gore puani artirir/azaltir.
- Agresif ama kuralli alim-satim (trailing stop, max loss, trend kirilimi cikis).
- Her islemde Telegram'a neden aldi/satti + bakiye + kar/zarar ozeti gonderir.
- Acilista otomatik backtest calistirip en iyi parametreleri yukler (`best_params.json`).

## Calistirma
- Tek komut: `python main.py`
