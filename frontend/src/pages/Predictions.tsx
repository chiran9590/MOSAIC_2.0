import { useCallback, useEffect, useState } from 'react'
import {
  Activity,
  RefreshCw,
  TrendingDown,
  TrendingUp,
  Minus,
  AlertTriangle,
  Radio,
  Database,
  Clock,
} from 'lucide-react'

import {
  getMockPrediction,
  getLiveStrategy,
} from '../api/dashboard'

import type {
  PredictionResponse,
  LiveStrategyResponse,
} from '../types/api'

type PredictionMode = 'live' | 'mock'

function formatProbability(value: number) {
  return `${(value * 100).toFixed(2)}%`
}

function Predictions() {
  const [mode, setMode] = useState<PredictionMode>('live')
  const [data, setData] = useState<
    PredictionResponse | LiveStrategyResponse | null
  >(null)
  const [strategyData, setStrategyData] =
    useState<LiveStrategyResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadPrediction = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      if (mode === 'live') {
        const response = await getLiveStrategy()

        if (!response.success) {
          throw new Error('Live strategy request was unsuccessful.')
        }

        setData(response)
        setStrategyData(response)
      } else {
        const response = await getMockPrediction()

        if (!response.success) {
          throw new Error('Mock prediction request was unsuccessful.')
        }

        setData(response)
        setStrategyData(null)
      }
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail

      setError(
        detail?.includes('No intraday candles returned')
          ? 'Market is closed today. Live predictions will resume when the market reopens.'
          : err instanceof Error
            ? err.message
            : 'Unable to load prediction.',
      )
    } finally {
      setLoading(false)
    }
  }, [mode])

  useEffect(() => {
    void Promise.resolve().then(() => loadPrediction())
  }, [loadPrediction])

  const result = data?.prediction_result

  const predictionStyle =
    result?.prediction === 'UP'
      ? {
          color: 'text-[#20c997]',
          background: 'bg-[#102a27]',
          border: 'border-[#275043]',
          Icon: TrendingUp,
        }
      : result?.prediction === 'DOWN'
        ? {
            color: 'text-[#f16b78]',
            background: 'bg-[#2b1920]',
            border: 'border-[#713b42]',
            Icon: TrendingDown,
          }
        : {
            color: 'text-amber-400',
            background: 'bg-[#29251a]',
            border: 'border-[#493e25]',
            Icon: Minus,
          }

  const probabilities = result
    ? [
        { label: 'UP', value: result.probabilities.UP, color: 'bg-[#20c997]' },
        { label: 'DOWN', value: result.probabilities.DOWN, color: 'bg-[#f16b78]' },
        { label: 'SIDEWAYS', value: result.probabilities.SIDEWAYS, color: 'bg-amber-400' },
      ]
    : []

  const isLive = mode === 'live'
  const isFresh = result?.is_latest_candle_prediction
  const predictionAge = result?.prediction_age_candles

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-4 md:p-8">

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Market Predictions
          </h1>
          <p className="mt-1 text-sm text-[#8492a6]">
            RELIANCE predictions from the MOSAIC experimental model.
          </p>
        </div>

        <button
          onClick={() => void loadPrediction()}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-[#304158] bg-[#142033] px-4 py-2.5 text-xs font-medium text-slate-200 transition hover:bg-[#1b2b42] disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          Refresh Prediction
        </button>
      </div>

      {/* Data source selector */}
      <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-4">
        <p className="mb-3 text-xs font-semibold tracking-wide text-[#8492a6]">
          PREDICTION DATA SOURCE
        </p>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => setMode('live')}
            disabled={loading && mode === 'live'}
            className={`flex items-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition ${
              isLive
                ? 'border-[#275043] bg-[#102a27] text-[#20c997]'
                : 'border-[#304158] bg-[#142033] text-slate-300 hover:bg-[#1b2b42]'
            }`}
          >
            <Radio size={16} />
            Live Samco
          </button>

          <button
            onClick={() => setMode('mock')}
            disabled={loading && mode === 'mock'}
            className={`flex items-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition ${
              !isLive
                ? 'border-[#493e25] bg-[#29251a] text-amber-400'
                : 'border-[#304158] bg-[#142033] text-slate-300 hover:bg-[#1b2b42]'
            }`}
          >
            <Database size={16} />
            Mock Simulation
          </button>
        </div>
      </section>

      {/* Data source warning */}
      <div className={`rounded-xl border p-4 ${
        isLive
          ? 'border-[#275043] bg-[#102a27]'
          : 'border-[#493e25] bg-[#29251a]'
      }`}>
        <div className="flex items-start gap-3">
          <AlertTriangle
            size={18}
            className={`mt-0.5 shrink-0 ${
              isLive ? 'text-[#20c997]' : 'text-amber-400'
            }`}
          />

          <div>
            <p className={`text-xs font-semibold ${
              isLive ? 'text-[#20c997]' : 'text-amber-300'
            }`}>
              {isLive ? 'Live market data · Experimental model' : 'Synthetic market simulation'}
            </p>

            <p className="mt-1 text-xs leading-5 text-[#b5a98c]">
              {isLive
                ? 'Market candles are retrieved from Samco. Model predictions remain experimental and are not validated trading signals or financial advice.'
                : 'This mode uses synthetic market data for testing. Results do not represent actual market conditions.'}
              {' '}No real broker orders are placed.
            </p>
          </div>
        </div>
      </div>

      {/* API error */}
      {error && (
        <div className="rounded-lg border border-[#713b42] bg-[#2b1920] p-4 text-sm text-[#f4a0a8]">
          Prediction API error: {error}

          <button
            onClick={() => void loadPrediction()}
            className="ml-3 underline"
          >
            Try again
          </button>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-8 text-center text-sm text-[#8492a6]">
          <RefreshCw size={22} className="mx-auto mb-3 animate-spin" />
          Running {isLive ? 'live Samco' : 'mock'} prediction model...
        </div>
      )}

      {/* Prediction results */}
      {!loading && result && (
        <>
          {/* Prediction freshness */}
          {isLive && (
            <section className={`rounded-xl border p-4 ${
              isFresh
                ? 'border-[#275043] bg-[#102a27]'
                : 'border-[#493e25] bg-[#29251a]'
            }`}>
              <div className="flex items-start gap-3">
                <Clock
                  size={19}
                  className={isFresh ? 'text-[#20c997]' : 'text-amber-400'}
                />

                <div>
                  <p className={`text-sm font-semibold ${
                    isFresh ? 'text-[#20c997]' : 'text-amber-300'
                  }`}>
                    {isFresh
                      ? 'Prediction uses the latest market candle'
                      : `Prediction is ${predictionAge ?? '—'} candle(s) behind`}
                  </p>

                  <p className="mt-1 text-xs leading-5 text-[#b5a98c]">
                    Prediction candle: {result.candle_timestamp}
                  </p>

                  <p className="text-xs leading-5 text-[#b5a98c]">
                    Latest market candle: {result.latest_market_candle_timestamp ?? data?.last_candle_timestamp ?? '—'}
                  </p>

                  {!isFresh && (
                    <p className="mt-2 text-xs text-amber-300">
                      The latest candle has unavailable model features.
                      This prediction uses the most recent valid feature row.
                    </p>
                  )}
                </div>
              </div>
            </section>
          )}

          {/* Live strategy assessment */}
          {isLive && strategyData && (
            <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5 md:p-6">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold text-white">
                    Strategy Assessment
                  </h2>
                  <p className="mt-1 text-xs text-[#8492a6]">
                    Decision and risk evaluation from the live strategy engine.
                  </p>
                </div>

                <span className="rounded-lg border border-[#493e25] bg-[#29251a] px-3 py-2 text-xs font-semibold text-amber-300">
                  {strategyData.strategy_result.decision}
                </span>
              </div>

              <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
                <div>
                  <p className="text-xs text-[#8492a6]">Model approval status</p>
                  <p className="mt-2 break-words text-sm font-medium text-amber-300">
                    {strategyData.strategy_result.model_status}
                  </p>
                </div>

                <div>
                  <p className="text-xs text-[#8492a6]">Direction</p>
                  <p className="mt-2 text-sm font-medium text-white">
                    {strategyData.strategy_result.direction ?? 'None'}
                  </p>
                </div>

                <div>
                  <p className="text-xs text-[#8492a6]">Risk accepted</p>
                  <p className="mt-2 text-sm font-medium text-white">
                    {strategyData.strategy_result.risk_accepted ? 'Yes' : 'No'}
                  </p>
                </div>

                <div>
                  <p className="text-xs text-[#8492a6]">Entry price</p>
                  <p className="mt-2 text-sm font-medium text-white">
                    {strategyData.strategy_result.entry_price !== null
                      ? `₹${strategyData.strategy_result.entry_price.toFixed(2)}`
                      : '—'}
                  </p>
                </div>

                <div>
                  <p className="text-xs text-[#8492a6]">Stop loss</p>
                  <p className="mt-2 text-sm font-medium text-white">
                    {strategyData.strategy_result.stop_loss !== null
                      ? `₹${strategyData.strategy_result.stop_loss.toFixed(2)}`
                      : '—'}
                  </p>
                </div>

                <div>
                  <p className="text-xs text-[#8492a6]">Position quantity</p>
                  <p className="mt-2 text-sm font-medium text-white">
                    {strategyData.strategy_result.quantity}
                  </p>
                </div>
              </div>

              <div className="mt-5 rounded-lg border border-[#304158] bg-[#142033] p-4">
                <p className="text-xs text-[#8492a6]">Decision explanation</p>
                <p className="mt-2 text-sm leading-6 text-slate-200">
                  {strategyData.strategy_result.reason}
                </p>

                <p className="mt-3 text-xs leading-5 text-amber-300">
                  Risk assessment: {strategyData.strategy_result.risk_reason}
                </p>
              </div>

              <div className="mt-5 flex flex-wrap gap-3 text-xs">
                <span className="rounded-md border border-[#713b42] bg-[#2b1920] px-3 py-2 text-[#f4a0a8]">
                  Paper trade eligible: {strategyData.paper_trade_eligible ? 'Yes' : 'No'}
                </span>

                <span className="rounded-md border border-[#304158] bg-[#142033] px-3 py-2 text-slate-300">
                  Real order placed: {strategyData.execution.real_order_placed ? 'Yes' : 'No'}
                </span>
              </div>

              <p className="mt-4 text-xs leading-5 text-[#8492a6]">
                {strategyData.warning}
              </p>
            </section>
          )}

          {/* Summary cards */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">

            <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
              <div className="flex items-center gap-2 text-xs text-[#8492a6]">
                <Activity size={16} />
                INSTRUMENT
              </div>

              <div className="mt-4 text-2xl font-semibold text-white">
                {result.symbol}
              </div>

              <p className="mt-2 text-xs text-[#8492a6]">
                NSE equity · RELIANCE
              </p>
            </section>

            <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
              <div className="text-xs text-[#8492a6]">
                MODEL PREDICTION
              </div>

              <div className={`mt-3 inline-flex items-center gap-2 rounded-lg border px-4 py-2 ${predictionStyle.background} ${predictionStyle.border} ${predictionStyle.color}`}>
                <predictionStyle.Icon size={21} />
                <span className="text-xl font-bold">
                  {result.prediction}
                </span>
              </div>

              <p className="mt-3 text-xs text-[#8492a6]">
                {result.prediction_horizon_minutes}-minute prediction horizon
              </p>
            </section>

            <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
              <div className="text-xs text-[#8492a6]">
                MODEL STATUS
              </div>

              <div className="mt-4 text-xl font-semibold text-amber-400">
                {result.status}
              </div>

              <p className="mt-2 break-all text-xs text-[#8492a6]">
                {result.model_version}
              </p>
            </section>
          </div>

          {/* Probabilities */}
          <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5 md:p-6">
            <div>
              <h2 className="text-sm font-semibold text-white">
                Prediction Probability Distribution
              </h2>

              <p className="mt-1 text-xs text-[#8492a6]">
                Model-estimated probabilities for each classification.
              </p>
            </div>

            <div className="mt-7 space-y-6">
              {probabilities.map((item) => (
                <div key={item.label}>
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-sm font-medium text-slate-300">
                      {item.label}
                    </span>

                    <span className="text-sm font-semibold tabular-nums text-white">
                      {formatProbability(item.value)}
                    </span>
                  </div>

                  <div className="h-3 overflow-hidden rounded-full bg-[#1a2739]">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${item.color}`}
                      style={{
                        width: `${Math.max(0, Math.min(100, item.value * 100))}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Run details */}
          <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
            <h2 className="text-sm font-semibold text-white">
              Prediction Run Details
            </h2>

            <div className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
              {[
                ['Data source', data?.data_source ?? '—'],
                ['Trading date', data?.trading_date ?? '—'],
                ['Input 1-minute candles', String(data?.input_candles ?? '—')],
                ['Completed 5-minute candles', String(data?.completed_5min_candles ?? '—')],
                ['Prediction candle', result.candle_timestamp],
                ['Latest market candle', result.latest_market_candle_timestamp ?? data?.last_candle_timestamp ?? '—'],
              ].map(([label, value]) => (
                <div key={label}>
                  <p className="text-xs text-[#8492a6]">{label}</p>
                  <p className="mt-2 break-words text-sm font-medium text-slate-200">
                    {value}
                  </p>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  )
}

export default Predictions
