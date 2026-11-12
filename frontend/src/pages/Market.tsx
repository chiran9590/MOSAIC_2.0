import { useEffect, useMemo, useState } from 'react'
import { getMarketData } from '../api/dashboard'
import type { MarketResponse } from '../types/api'

const formatPrice = (value: number) =>
  `₹${value.toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`

const formatVolume = (value: number) =>
  value.toLocaleString('en-IN')

const formatTime = (timestamp: string) => {
  const date = new Date(timestamp.replace(' ', 'T'))
  return date.toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

function Market() {
  const [market, setMarket] = useState<MarketResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [retryCount, setRetryCount] = useState(0)
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null)

  useEffect(() => {
    let cancelled = false

    async function loadMarketData() {
      try {
        setLoading(true)
        setError(null)

        const response = await getMarketData()

        if (!response.success) {
          throw new Error('The market API returned an unsuccessful response.')
        }

        if (!cancelled) {
          setMarket(response)
          setLastRefreshed(new Date())
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : 'Unable to load market data.',
          )
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadMarketData()

    return () => {
      cancelled = true
    }
  }, [retryCount])

  const chartData = useMemo(
    () => market?.candles.slice(-60) ?? [],
    [market],
  )

  const chart = useMemo(() => {
    if (chartData.length === 0) return null

    const width = 1100
    const height = 430
    const left = 72
    const right = 24
    const top = 24
    const priceBottom = 290
    const volumeTop = 325
    const volumeBottom = 390
    const plotWidth = width - left - right

    const highest = Math.max(...chartData.map((candle) => candle.high))
    const lowest = Math.min(...chartData.map((candle) => candle.low))
    const padding = (highest - lowest) * 0.08 || highest * 0.001
    const minPrice = lowest - padding
    const maxPrice = highest + padding
    const maxVolume = Math.max(...chartData.map((candle) => candle.volume), 1)
    const slotWidth = plotWidth / chartData.length

    const priceY = (price: number) =>
      priceBottom -
      ((price - minPrice) / (maxPrice - minPrice)) *
        (priceBottom - top)

    const volumeY = (volume: number) =>
      volumeBottom -
      (volume / maxVolume) * (volumeBottom - volumeTop)

    const candles = chartData.map((candle, index) => {
      const x = left + index * slotWidth + slotWidth / 2
      const rising = candle.close >= candle.open
      const bodyTop = priceY(Math.max(candle.open, candle.close))
      const bodyBottom = priceY(Math.min(candle.open, candle.close))

      return {
        ...candle,
        x,
        rising,
        color: rising ? '#20c997' : '#f16b78',
        wickTop: priceY(candle.high),
        wickBottom: priceY(candle.low),
        bodyTop,
        bodyHeight: Math.max(1.5, bodyBottom - bodyTop),
        volumeY: volumeY(candle.volume),
        volumeHeight:
          volumeBottom - volumeY(candle.volume),
      }
    })

    const priceTicks = Array.from({ length: 5 }, (_, index) => {
      const value = maxPrice - (index / 4) * (maxPrice - minPrice)
      return { value, y: priceY(value) }
    })

    const labelIndices = [
      0,
      Math.floor((chartData.length - 1) / 3),
      Math.floor(((chartData.length - 1) * 2) / 3),
      chartData.length - 1,
    ]

    return {
      width,
      height,
      left,
      right,
      top,
      priceBottom,
      volumeTop,
      volumeBottom,
      plotWidth,
      candles,
      priceTicks,
      labels: labelIndices.map((index) => ({
        x: left + index * slotWidth + slotWidth / 2,
        label: formatTime(chartData[index].timestamp),
      })),
      minPrice,
      maxPrice,
    }
  }, [chartData])

  const latest = market?.candles[market.candles.length - 1]
  const previous = market && market.candles.length > 1
    ? market.candles[market.candles.length - 2]
    : null

  const priceChange = latest && previous
    ? latest.close - previous.close
    : 0

  const priceChangePct = previous && previous.close !== 0
    ? (priceChange / previous.close) * 100
    : 0

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-4 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Market Overview
          </h1>
          <p className="mt-1 text-sm text-[#8492a6]">
            Price action, OHLC candles and trading volume.
          </p>
        </div>

        <div className="rounded-lg border border-[#174b43] bg-[#102b29] px-3 py-2 text-xs font-semibold text-[#20c997]">
          {market?.data_source === 'SAMCO'
            ? 'SAMCO · REAL MARKET DATA'
            : 'MARKET DATA'}
        </div>
      </div>

      {market && (
        <div className={`rounded-xl border p-4 ${
          market.market_status === 'SESSION_HOURS'
            ? 'border-[#174b43] bg-[#102b29]'
            : 'border-[#493e25] bg-[#29251a]'
        }`}>
          <div className={`text-sm font-semibold ${
            market.market_status === 'SESSION_HOURS'
              ? 'text-[#20c997]'
              : 'text-amber-300'
          }`}>
            {market.market_status === 'SESSION_HOURS'
              ? 'NSE market session'
              : 'Market closed · Latest available session'}
          </div>
          <p className="mt-1 text-xs leading-5 text-[#c2ccda]">
            Source: Samco · {market.trading_date} ·
            {' '}{market.total_candles} completed five-minute candles.
            {market.is_current_session
              ? ' Data belongs to today’s trading session.'
              : ' Displaying the most recent available historical session.'}
            {' '}The experimental model is not a validated trading signal.
          </p>
          {market.last_candle_timestamp && (
            <p className="mt-2 text-xs text-[#8492a6]">
              Last candle: {market.last_candle_timestamp} IST
            </p>
          )}
        </div>
      )}

      {loading && (
        <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-8 text-center text-sm text-[#8492a6]">
          Loading market data...
        </div>
      )}

      {error && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#713b42] bg-[#2b1920] p-4">
          <p className="text-sm text-[#f4a0a8]">
            Market API error: {error}. Check that the FastAPI backend is running.
          </p>
          <button
            type="button"
            onClick={() => setRetryCount((count) => count + 1)}
            disabled={loading}
            className="rounded-lg border border-[#713b42] px-3 py-2 text-xs font-semibold text-white transition hover:bg-[#49232b] disabled:cursor-not-allowed disabled:opacity-50"
          >
            Retry
          </button>
        </div>
      )}

      {market && latest && (
        <>
          <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <div className="flex flex-wrap items-center gap-3">
                  <h2 className="text-xl font-semibold text-white">
                    {market.symbol}
                  </h2>
                  <span className="rounded-md border border-[#174b43] bg-[#102b29] px-2 py-1 text-[10px] font-semibold text-[#20c997]">
                    {market.data_source}
                  </span>
                </div>
                <p className="mt-1 text-xs text-[#8492a6]">
                  NSE · {market.timeframe} · {market.trading_date}
                </p>
              </div>

              <div className="text-right">
                <div className="text-2xl font-semibold tabular-nums text-white">
                  {formatPrice(latest.close)}
                </div>
                <div
                  className={`mt-1 text-xs font-medium ${
                    priceChange >= 0 ? 'text-[#20c997]' : 'text-[#f16b78]'
                  }`}
                >
                  {priceChange >= 0 ? '+' : ''}
                  {formatPrice(priceChange)} ({priceChangePct.toFixed(2)}%)
                  <span className="ml-1 text-[#8492a6]">
                    vs previous candle
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-4 border-t border-[#202d40] pt-5 md:grid-cols-4">
              {[
                { label: 'OPEN', value: latest.open },
                { label: 'HIGH', value: latest.high },
                { label: 'LOW', value: latest.low },
                { label: 'CLOSE', value: latest.close },
              ].map((item) => (
                <div key={item.label}>
                  <div className="text-[10px] font-medium tracking-wider text-[#8492a6]">
                    {item.label}
                  </div>
                  <div className="mt-2 text-lg font-semibold tabular-nums text-slate-200">
                    {formatPrice(item.value)}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-4 md:p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-white">
                  RELIANCE · Candlestick Chart
                </h2>
                <p className="mt-1 text-xs text-[#8492a6]">
                  Last {chartData.length} available candles · 5-minute interval
                </p>
                {lastRefreshed && (
                  <p className="mt-1 text-[11px] text-[#617087]">
                    Last refreshed: {lastRefreshed.toLocaleTimeString('en-IN', {
                      timeZone: 'Asia/Kolkata',
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                      hour12: true,
                    })} IST
                  </p>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() => setRetryCount((count) => count + 1)}
                  disabled={loading}
                  className="flex items-center gap-2 rounded-lg border border-[#304158] bg-[#142033] px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-[#1b2b42] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <span className={loading ? 'animate-spin' : ''}>↻</span>
                  {loading ? 'Refreshing...' : 'Refresh'}
                </button>

                <div className="flex items-center gap-4 text-[11px]">
                <span className="flex items-center gap-2 text-[#20c997]">
                  <span className="h-2 w-2 rounded-full bg-[#20c997]" />
                  Bullish
                </span>
                <span className="flex items-center gap-2 text-[#f16b78]">
                  <span className="h-2 w-2 rounded-full bg-[#f16b78]" />
                  Bearish
                </span>
                </div>
              </div>
            </div>

            {chart && (
              <div className="mt-5 w-full overflow-x-auto">
                <svg
                  viewBox={`0 0 ${chart.width} ${chart.height}`}
                  className="block min-w-[680px] w-full"
                  role="img"
                  aria-label="RELIANCE five-minute candlestick and volume chart"
                >
                  {chart.priceTicks.map((tick) => (
                    <g key={tick.value}>
                      <line
                        x1={chart.left}
                        y1={tick.y}
                        x2={chart.width - chart.right}
                        y2={tick.y}
                        stroke="#243246"
                        strokeDasharray="4 5"
                      />
                      <text
                        x={chart.left - 10}
                        y={tick.y + 4}
                        textAnchor="end"
                        fill="#8492a6"
                        fontSize="12"
                      >
                        {tick.value.toFixed(2)}
                      </text>
                    </g>
                  ))}

                  <line
                    x1={chart.left}
                    y1={chart.priceBottom + 16}
                    x2={chart.width - chart.right}
                    y2={chart.priceBottom + 16}
                    stroke="#304158"
                  />

                  <text
                    x={chart.left}
                    y={chart.volumeTop - 10}
                    fill="#8492a6"
                    fontSize="11"
                  >
                    VOLUME
                  </text>

                  {chart.candles.map((candle) => (
                    <g key={candle.timestamp}>
                      <title>
                        {`${candle.timestamp} | O: ${candle.open} | H: ${candle.high} | L: ${candle.low} | C: ${candle.close} | Volume: ${candle.volume}`}
                      </title>
                      <line
                        x1={candle.x}
                        y1={candle.wickTop}
                        x2={candle.x}
                        y2={candle.wickBottom}
                        stroke={candle.color}
                        strokeWidth="1.5"
                      />
                      <rect
                        x={candle.x - Math.max(1.5, (chart.plotWidth / chart.candles.length) * 0.3)}
                        y={candle.bodyTop}
                        width={Math.max(3, (chart.plotWidth / chart.candles.length) * 0.6)}
                        height={candle.bodyHeight}
                        fill={candle.rising ? candle.color : candle.color}
                        rx="1"
                      />
                      <rect
                        x={candle.x - Math.max(1.5, (chart.plotWidth / chart.candles.length) * 0.3)}
                        y={candle.volumeY}
                        width={Math.max(3, (chart.plotWidth / chart.candles.length) * 0.6)}
                        height={candle.volumeHeight}
                        fill={candle.color}
                        opacity="0.55"
                      />
                    </g>
                  ))}

                  {chart.labels.map((item) => (
                    <text
                      key={item.x}
                      x={item.x}
                      y={chart.height - 12}
                      textAnchor="middle"
                      fill="#8492a6"
                      fontSize="12"
                    >
                      {item.label}
                    </text>
                  ))}
                </svg>
              </div>
            )}

            <div className="mt-3 flex flex-wrap justify-between gap-2 border-t border-[#202d40] pt-4 text-[11px] text-[#8492a6]">
              <span>Price scale: INR</span>
              <span>Volume: shares</span>
              <span>Chart displays the latest 60 candles</span>
            </div>
          </section>

          <section className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
              <div className="text-xs text-[#8492a6]">Available candles</div>
              <div className="mt-2 text-2xl font-semibold text-white">
                {market.total_candles}
              </div>
            </div>
            <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
              <div className="text-xs text-[#8492a6]">Latest candle volume</div>
              <div className="mt-2 text-2xl font-semibold text-white">
                {formatVolume(latest.volume)}
              </div>
            </div>
            <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
              <div className="text-xs text-[#8492a6]">Data source</div>
              <div className="mt-2 text-2xl font-semibold text-amber-300">
                {market.data_source}
              </div>
            </div>
          </section>

          <footer className="pb-3 text-[10px] text-[#617087]">
            MOSAIC 2.0 · EXPERIMENTAL MARKET RESEARCH · SAMCO MARKET DATA · PAPER TRADING ONLY
          </footer>
        </>
      )}
    </div>
  )
}

export default Market
