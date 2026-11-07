import { useCallback, useEffect, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Clock3,
  Database,
  Minus,
  RefreshCw,
  ShieldCheck,
  Wallet,
} from 'lucide-react'

import { getLiveStrategy } from '../api/dashboard'
import type { LiveStrategyResponse } from '../types/api'

function currency(value: number | null) {
  if (value === null || !Number.isFinite(value)) return '—'

  return `₹${value.toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

function dateTime(value?: string) {
  if (!value) return '—'

  const date = new Date(value)

  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString('en-IN', {
        dateStyle: 'medium',
        timeStyle: 'short',
      })
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string
  value: string
  detail?: string
}) {
  return (
    <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
      <p className="text-xs text-[#8492a6]">{label}</p>
      <p className="mt-3 break-words text-xl font-semibold text-white">
        {value}
      </p>
      {detail && (
        <p className="mt-2 text-xs leading-5 text-[#8492a6]">{detail}</p>
      )}
    </div>
  )
}

export default function Strategy() {
  const [data, setData] = useState<LiveStrategyResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadStrategy = useCallback(async (manual = false) => {
    try {
      if (manual) setRefreshing(true)
      else setLoading(true)

      setError(null)

      const response = await getLiveStrategy()

      if (!response.success) {
        throw new Error('The strategy API returned an unsuccessful response.')
      }

      setData(response)
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail

      setError(
        detail?.includes('No intraday candles returned')
          ? 'Market is closed today. Live strategy analysis will resume when the market reopens.'
          : err instanceof Error
            ? err.message
            : 'Unable to load the live strategy assessment.',
      )
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void Promise.resolve().then(() => loadStrategy())
  }, [loadStrategy])

  const strategy = data?.strategy_result
  const prediction = data?.prediction_result

  const decisionStyle =
    strategy?.decision === 'PAPER_TRADE_CANDIDATE'
      ? 'border-[#275043] bg-[#102a27] text-[#53d6b0]'
      : strategy?.decision === 'REJECTED_BY_RISK'
        ? 'border-[#713b42] bg-[#2b1920] text-[#f4a0a8]'
        : 'border-[#493e25] bg-[#29251a] text-amber-300'

  const probabilities = prediction
    ? [
        { label: 'UP', value: prediction.probabilities.UP, color: 'bg-[#20c997]' },
        { label: 'DOWN', value: prediction.probabilities.DOWN, color: 'bg-[#f16b78]' },
        { label: 'SIDEWAYS', value: prediction.probabilities.SIDEWAYS, color: 'bg-amber-400' },
      ]
    : []

  const isFresh = prediction?.is_latest_candle_prediction

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-4 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Strategy
          </h1>
          <p className="mt-1 text-sm text-[#8492a6]">
            Live prediction assessment, trade direction and risk management.
          </p>
        </div>

        <button
          type="button"
          onClick={() => void loadStrategy(true)}
          disabled={loading || refreshing}
          className="flex items-center gap-2 rounded-lg border border-[#304158] bg-[#142033] px-4 py-2.5 text-sm font-medium text-slate-200 transition hover:bg-[#1b2b40] disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RefreshCw
            size={15}
            className={refreshing ? 'animate-spin' : ''}
          />
          {refreshing ? 'Refreshing...' : 'Refresh Strategy'}
        </button>
      </div>

      <div className="flex items-start gap-3 rounded-xl border border-[#493e25] bg-[#29251a] p-4">
        <AlertTriangle size={19} className="mt-0.5 shrink-0 text-amber-300" />
        <div>
          <p className="text-sm font-semibold text-amber-300">
            Experimental research environment
          </p>
          <p className="mt-1 text-xs leading-5 text-[#b5a98c]">
            Strategy assessments are experimental and for paper-trading
            research only. The model is not production-approved. No real
            broker orders are placed.
          </p>
        </div>
      </div>

      {error && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[#713b42] bg-[#2b1920] p-4">
          <p className="text-sm text-[#f4a0a8]">
            Strategy API error: {error}
          </p>
          <button
            type="button"
            onClick={() => void loadStrategy(true)}
            disabled={refreshing}
            className="rounded-lg border border-[#713b42] px-3 py-2 text-xs font-semibold text-white hover:bg-[#49232b] disabled:opacity-50"
          >
            Retry
          </button>
        </div>
      )}

      {loading && (
        <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-10 text-center">
          <RefreshCw size={24} className="mx-auto mb-3 animate-spin text-[#70a7f8]" />
          <p className="text-sm text-[#8492a6]">
            Fetching market data and evaluating strategy...
          </p>
        </div>
      )}

      {!loading && !error && data && strategy && prediction && (
        <>
          <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5 md:p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-2 text-xs text-[#8492a6]">
                  <ShieldCheck size={16} />
                  STRATEGY ENGINE DECISION
                </div>
                <h2 className="mt-3 text-2xl font-bold text-white">
                  {strategy.direction ?? 'NO DIRECTION'}
                </h2>
                <p className="mt-1 text-sm text-[#8492a6]">
                  {strategy.symbol} · {strategy.prediction} prediction
                </p>
              </div>

              <span className={`rounded-lg border px-3 py-2 text-xs font-semibold ${decisionStyle}`}>
                {strategy.decision}
              </span>
            </div>

            <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Metric
                label="Model approval status"
                value={strategy.model_status}
                detail={strategy.model_version}
              />
              <Metric
                label="Risk accepted"
                value={strategy.risk_accepted ? 'Yes' : 'No'}
                detail={strategy.risk_reason}
              />
              <Metric
                label="Paper-trade eligibility"
                value={data.paper_trade_eligible ? 'Yes' : 'No'}
                detail="Strategy-level eligibility only"
              />
            </div>

            <div className="mt-5 rounded-lg border border-[#304158] bg-[#142033] p-4">
              <p className="text-xs text-[#8492a6]">Strategy explanation</p>
              <p className="mt-2 text-sm leading-6 text-slate-200">
                {strategy.reason}
              </p>
            </div>
          </section>

          <section>
            <div className="mb-4">
              <h2 className="text-sm font-semibold text-white">
                Risk Management
              </h2>
              <p className="mt-1 text-xs text-[#8492a6]">
                Values calculated by the backend risk manager.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
              <Metric
                label="Indicative entry price"
                value={currency(strategy.entry_price)}
              />
              <Metric
                label="Stop loss"
                value={currency(strategy.stop_loss)}
              />
              <Metric
                label="Position quantity"
                value={strategy.quantity.toLocaleString('en-IN')}
              />
              <Metric
                label="Position value"
                value={currency(strategy.position_value)}
              />
              <Metric
                label="Estimated total risk"
                value={currency(strategy.estimated_total_risk)}
              />
              <Metric
                label="Risk assessment"
                value={strategy.risk_accepted ? 'Accepted' : 'Not accepted'}
                detail={strategy.risk_reason}
              />
            </div>
          </section>

          <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5 md:p-6">
            <div className="flex items-center gap-2">
              <Activity size={17} className="text-[#70a7f8]" />
              <h2 className="text-sm font-semibold text-white">
                Prediction Probabilities
              </h2>
            </div>

            <p className="mt-1 text-xs text-[#8492a6]">
              Model-estimated classification probabilities.
            </p>

            <div className="mt-6 space-y-5">
              {probabilities.map((item) => (
                <div key={item.label}>
                  <div className="mb-2 flex justify-between text-sm">
                    <span className="text-slate-300">{item.label}</span>
                    <span className="font-semibold tabular-nums text-white">
                      {(item.value * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="h-2.5 overflow-hidden rounded-full bg-[#1a2739]">
                    <div
                      className={`h-full rounded-full ${item.color}`}
                      style={{
                        width: `${Math.max(0, Math.min(100, item.value * 100))}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
            <div className="flex items-center gap-2">
              <Database size={17} className="text-[#70a7f8]" />
              <h2 className="text-sm font-semibold text-white">
                Market Data and Execution
              </h2>
            </div>

            <div className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
              <Metric
                label="Data source"
                value={data.data_source}
                detail={data.synthetic_data ? 'Synthetic data' : 'Non-synthetic market data'}
              />
              <Metric
                label="Timeframe"
                value={data.timeframe}
                detail={`${data.completed_5min_candles} completed 5-minute candles`}
              />
              <Metric
                label="Prediction horizon"
                value={`${prediction.prediction_horizon_minutes} minutes`}
              />
              <Metric
                label="Prediction candle"
                value={dateTime(prediction.candle_timestamp)}
              />
              <Metric
                label="Latest market candle"
                value={dateTime(data.last_candle_timestamp)}
              />
              <Metric
                label="Prediction freshness"
                value={
                  isFresh === undefined
                    ? 'Not reported'
                    : isFresh
                      ? 'Latest candle'
                      : `${prediction.prediction_age_candles ?? '—'} candle(s) behind`
                }
              />
            </div>

            <div className="mt-5 flex flex-wrap gap-3 text-xs">
              <span className="flex items-center gap-2 rounded-md border border-[#304158] bg-[#142033] px-3 py-2 text-slate-300">
                <Wallet size={14} />
                {data.execution.mode}
              </span>
              <span className="flex items-center gap-2 rounded-md border border-[#275043] bg-[#102a27] px-3 py-2 text-[#53d6b0]">
                <ShieldCheck size={14} />
                Real orders: {data.execution.real_order_placed ? 'Yes' : 'No'}
              </span>
              <span className="flex items-center gap-2 rounded-md border border-[#304158] bg-[#142033] px-3 py-2 text-slate-300">
                <Clock3 size={14} />
                Broker connected: {data.execution.broker_connected ? 'Yes' : 'No'}
              </span>
            </div>

            <div className="mt-5 rounded-lg border border-[#713b42] bg-[#2b1920] p-4">
              <p className="text-sm font-semibold text-[#f4a0a8]">
                Paper-trade creation remains disabled
              </p>
              <p className="mt-2 text-xs leading-5 text-[#d6a4aa]">
                Even if the strategy reports paper-trade eligibility, the
                separate paper-trade creation endpoint remains disabled until
                production approval. This page cannot place or create trades.
              </p>
            </div>

            <p className="mt-4 text-xs leading-5 text-[#8492a6]">
              {data.warning}
            </p>
          </section>

          <div className="flex flex-wrap items-center gap-2 pb-3 text-[10px] text-[#617087]">
            <Minus size={13} />
            MOSAIC 2.0 · EXPERIMENTAL STRATEGY RESEARCH · PAPER ONLY
          </div>
        </>
      )}
    </div>
  )
}
