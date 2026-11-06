import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  getPaperTrades,
  getPerformance,
} from '../api/dashboard'
import type {
  PaperTrade,
  PerformanceResponse,
} from '../types/api'
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  RefreshCw,
  Wallet,
} from 'lucide-react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function currency(value: number) {
  return `₹${Math.abs(value).toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

function signedCurrency(value: number) {
  const sign = value > 0 ? '+' : value < 0 ? '-' : ''
  return `${sign}${currency(value)}`
}

function dateTime(value: string) {
  if (!value) return '—'

  const date = new Date(value)

  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString('en-IN', {
        dateStyle: 'medium',
        timeStyle: 'short',
      })
}

function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
  accent = 'blue',
}: {
  label: string
  value: string
  detail: string
  icon: typeof Wallet
  accent?: 'blue' | 'green' | 'red' | 'neutral'
}) {
  const accents = {
    blue: 'bg-[#18263a] text-[#70a7f8]',
    green: 'bg-[#15332e] text-[#53d6b0]',
    red: 'bg-[#35202a] text-[#f16b78]',
    neutral: 'bg-[#202d40] text-[#a7b4c7]',
  }

  return (
    <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[#8492a6]">{label}</span>
        <span className={`rounded-lg p-2 ${accents[accent]}`}>
          <Icon size={17} />
        </span>
      </div>
      <div className="mt-4 text-2xl font-semibold text-white">{value}</div>
      <p className="mt-2 text-xs text-[#8492a6]">{detail}</p>
    </div>
  )
}

export default function Analytics() {
  const [performance, setPerformance] =
    useState<PerformanceResponse | null>(null)
  const [trades, setTrades] = useState<PaperTrade[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadData = useCallback(async (manual = false) => {
    try {
      if (manual) setRefreshing(true)
      else setLoading(true)

      setError(null)

      const [performanceResponse, tradesResponse] = await Promise.all([
        getPerformance(),
        getPaperTrades(),
      ])

      if (!performanceResponse.success || !tradesResponse.success) {
        throw new Error('The backend returned an unsuccessful response.')
      }

      setPerformance(performanceResponse)
      setTrades(tradesResponse.trades)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to load analytics data.',
      )
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void Promise.resolve().then(() => loadData())
  }, [loadData])

  const equityData = useMemo(() => {
    if (!performance) return []

    return [
      {
        label: 'Start',
        equity: performance.starting_capital,
      },
      ...performance.equity_curve.map((point, index) => ({
        label: `Trade ${index + 1}`,
        equity: point.equity,
      })),
    ]
  }, [performance])

  const closedTrades = useMemo(
    () =>
      trades
        .filter(
          (trade) =>
            trade.status === 'CLOSED' &&
            trade.net_pnl !== null &&
            trade.exit_time !== null,
        )
        .sort(
          (a, b) =>
            new Date(b.exit_time!).getTime() -
            new Date(a.exit_time!).getTime(),
        ),
    [trades],
  )

  const hasPerformance = performance !== null
  const totalClosed = performance?.closed_trades ?? 0
  const wins = performance?.winning_trades ?? 0
  const losses = performance?.losing_trades ?? 0
  const breakeven = performance?.breakeven_trades ?? 0
  const distributionTotal = wins + losses + breakeven

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-4 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="mb-2 text-xs text-[#8492a6]">
            MOSAIC / WORKSPACE / ANALYTICS
          </div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold text-white">Analytics</h1>
            <span className="rounded-md border border-[#275043] bg-[#102a27] px-2 py-1 text-[10px] font-semibold tracking-wider text-[#53d6b0]">
              PAPER DATA
            </span>
          </div>
          <p className="mt-2 text-sm text-[#8492a6]">
            Review realized performance, equity progression and simulated trade outcomes.
          </p>
        </div>

        <button
          onClick={() => void loadData(true)}
          disabled={loading || refreshing}
          className="flex items-center gap-2 rounded-lg border border-[#26364a] bg-[#111d2d] px-4 py-2.5 text-sm text-slate-200 transition hover:bg-[#18263a] disabled:opacity-50"
        >
          <RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      <div className="flex items-start gap-3 rounded-lg border border-[#493e25] bg-[#29251a] px-4 py-3 text-xs leading-5 text-amber-300">
        <Activity size={17} className="mt-0.5 shrink-0" />
        <p>
          Simulation only. These figures are calculated from recorded paper
          trades and estimated transaction costs. They do not establish live
          trading profitability. No real broker orders are placed.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-[#6b3038] bg-[#321b23] p-4 text-sm text-[#ff9ba4]">
          {error}
          <button
            onClick={() => void loadData(true)}
            className="ml-3 underline"
          >
            Retry
          </button>
        </div>
      )}

      {hasPerformance && totalClosed < 30 && (
        <div className="rounded-lg border border-[#493e25] bg-[#211f1a] px-4 py-3 text-xs leading-5 text-amber-300">
          Limited sample size: only {totalClosed} closed trade(s) are available.
          Performance percentages and win rate are descriptive only and should
          not be treated as reliable evidence of strategy performance.
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <MetricCard
          label="Net Realized P&L"
          value={
            loading || !performance
              ? '—'
              : signedCurrency(performance.net_pnl)
          }
          detail="After estimated transaction costs"
          icon={performance && performance.net_pnl < 0 ? ArrowDownRight : ArrowUpRight}
          accent={
            !performance || performance.net_pnl === 0
              ? 'neutral'
              : performance.net_pnl > 0
                ? 'green'
                : 'red'
          }
        />
        <MetricCard
          label="Gross P&L"
          value={loading || !performance ? '—' : signedCurrency(performance.gross_pnl)}
          detail="Before estimated costs"
          icon={BarChart3}
          accent="blue"
        />
        <MetricCard
          label="Estimated Costs"
          value={loading || !performance ? '—' : currency(performance.estimated_cost)}
          detail="Recorded closed-trade cost estimates"
          icon={Activity}
          accent="neutral"
        />
        <MetricCard
          label="Win Rate"
          value={loading || !performance ? '—' : `${performance.win_rate_pct.toFixed(2)}%`}
          detail={`${wins} wins from ${totalClosed} closed trades`}
          icon={ArrowUpRight}
          accent="blue"
        />
        <MetricCard
          label="Maximum Drawdown"
          value={loading || !performance ? '—' : currency(performance.max_drawdown)}
          detail={performance ? `${performance.max_drawdown_pct.toFixed(2)}% from peak equity` : 'Realized equity only'}
          icon={ArrowDownRight}
          accent={performance && performance.max_drawdown > 0 ? 'red' : 'neutral'}
        />
        <MetricCard
          label="Current Simulated Equity"
          value={loading || !performance ? '—' : currency(performance.current_equity)}
          detail={performance ? `Starting capital ${currency(performance.starting_capital)}` : 'Paper portfolio'}
          icon={Wallet}
          accent="blue"
        />
      </div>

      <div className="grid gap-5 xl:grid-cols-3">
        <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5 xl:col-span-2">
          <div>
            <h2 className="text-sm font-semibold text-white">Equity Curve</h2>
            <p className="mt-1 text-xs text-[#8492a6]">
              Starting capital plus chronological realized closed-trade P&L
            </p>
          </div>

          <div className="mt-6 h-[300px] w-full">
            {loading ? (
              <div className="flex h-full items-center justify-center text-xs text-[#8492a6]">
                Loading equity history...
              </div>
            ) : equityData.length <= 1 ? (
              <div className="flex h-full items-center justify-center text-xs text-[#8492a6]">
                No closed trades yet. The equity curve will appear after a paper trade is closed.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equityData}>
                  <defs>
                    <linearGradient id="analyticsEquityGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#20c997" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#20c997" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#243246" strokeDasharray="3 5" vertical={false} />
                  <XAxis
                    dataKey="label"
                    tick={{ fill: '#8492a6', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    domain={['auto', 'auto']}
                    tick={{ fill: '#8492a6', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(value: number) => `₹${(value / 1000).toFixed(1)}k`}
                    width={65}
                  />
                  <Tooltip
                    contentStyle={{
                      background: '#111d2d',
                      border: '1px solid #304158',
                      borderRadius: 8,
                      color: '#e2e8f0',
                    }}
                    formatter={(value) => [currency(Number(value)), 'Equity']}
                  />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke="#20c997"
                    strokeWidth={2}
                    fill="url(#analyticsEquityGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </section>

        <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
          <h2 className="text-sm font-semibold text-white">Trade Outcomes</h2>
          <p className="mt-1 text-xs text-[#8492a6]">
            Closed paper trades only
          </p>

          {loading ? (
            <div className="py-12 text-center text-xs text-[#8492a6]">
              Loading trade outcomes...
            </div>
          ) : distributionTotal === 0 ? (
            <div className="py-12 text-center text-xs text-[#8492a6]">
              No closed trades to analyze yet.
            </div>
          ) : (
            <>
              <div className="mt-8 space-y-5">
                {[
                  { label: 'Winning trades', count: wins, color: 'bg-[#20c997]', text: 'text-[#53d6b0]' },
                  { label: 'Losing trades', count: losses, color: 'bg-[#f16b78]', text: 'text-[#f16b78]' },
                  { label: 'Breakeven trades', count: breakeven, color: 'bg-[#8492a6]', text: 'text-[#a7b4c7]' },
                ].map((item) => (
                  <div key={item.label}>
                    <div className="mb-2 flex justify-between text-xs">
                      <span className="text-[#a7b4c7]">{item.label}</span>
                      <span className={item.text}>
                        {item.count} ({((item.count / distributionTotal) * 100).toFixed(1)}%)
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-[#202d40]">
                      <div
                        className={`h-full rounded-full ${item.color}`}
                        style={{ width: `${(item.count / distributionTotal) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-8 border-t border-[#202d40] pt-4 text-xs text-[#8492a6]">
                {distributionTotal} closed trades analyzed
              </div>
            </>
          )}
        </section>
      </div>

      <section className="overflow-hidden rounded-xl border border-[#202d40] bg-[#0e1625]">
        <div className="border-b border-[#202d40] p-5">
          <h2 className="text-sm font-semibold text-white">Closed Trade History</h2>
          <p className="mt-1 text-xs text-[#8492a6]">
            Realized results with entry, exit and estimated costs
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] text-left text-xs">
            <thead className="bg-[#111d2d] text-[#8492a6]">
              <tr>
                {['Trade', 'Symbol', 'Direction', 'Entry', 'Exit', 'Gross P&L', 'Costs', 'Net P&L', 'Exit Time'].map((heading) => (
                  <th key={heading} className="px-4 py-4 font-medium">
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {closedTrades.map((trade) => (
                <tr key={trade.id} className="border-t border-[#202d40] hover:bg-[#111d2d]/60">
                  <td className="px-4 py-4 text-[#8492a6]">#{trade.id}</td>
                  <td className="px-4 py-4 font-semibold text-white">{trade.symbol}</td>
                  <td className={`px-4 py-4 font-semibold ${trade.direction === 'LONG' ? 'text-[#53d6b0]' : 'text-[#f16b78]'}`}>
                    {trade.direction}
                  </td>
                  <td className="px-4 py-4 text-slate-300">{currency(trade.entry_price)}</td>
                  <td className="px-4 py-4 text-slate-300">
                    {trade.exit_price === null ? '—' : currency(trade.exit_price)}
                  </td>
                  <td className="px-4 py-4 text-slate-300">
                    {trade.gross_pnl === null ? '—' : signedCurrency(trade.gross_pnl)}
                  </td>
                  <td className="px-4 py-4 text-slate-300">
                    {trade.estimated_cost === null ? '—' : currency(trade.estimated_cost)}
                  </td>
                  <td className={`px-4 py-4 font-semibold ${trade.net_pnl === null ? 'text-[#8492a6]' : trade.net_pnl >= 0 ? 'text-[#53d6b0]' : 'text-[#f16b78]'}`}>
                    {trade.net_pnl === null ? '—' : signedCurrency(trade.net_pnl)}
                  </td>
                  <td className="px-4 py-4 text-[#8492a6]">{dateTime(trade.exit_time ?? '')}</td>
                </tr>
              ))}

              {!loading && closedTrades.length === 0 && (
                <tr>
                  <td colSpan={9} className="px-5 py-12 text-center text-[#8492a6]">
                    No closed paper trades available yet.
                  </td>
                </tr>
              )}

              {loading && (
                <tr>
                  <td colSpan={9} className="px-5 py-12 text-center text-[#8492a6]">
                    Loading closed trades...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="border-t border-[#202d40] px-5 py-3 text-[10px] text-[#8492a6]">
          Showing {closedTrades.length} closed trade(s). Open positions are excluded from realized P&L and the equity curve.
        </div>
      </section>

      <footer className="flex flex-wrap items-center justify-between gap-2 pb-3 text-[10px] text-[#617087]">
        <span>MOSAIC 2.0 · EXPERIMENTAL ANALYTICS</span>
        <span>SIMULATION ONLY · NO REAL BROKER ORDERS</span>
      </footer>
    </div>
  )
}
