
import { useCallback, useEffect, useMemo, useState } from 'react'
import { apiClient } from '../api/client'
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
  RefreshCw,
  Wallet,
  X,
} from 'lucide-react'

type TradeFilter = 'ALL' | 'OPEN' | 'CLOSED'

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

function SummaryCard({
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
  const accentClasses = {
    blue: 'text-[#70a7f8] bg-[#18263a]',
    green: 'text-[#53d6b0] bg-[#15332e]',
    red: 'text-[#f16b78] bg-[#35202a]',
    neutral: 'text-[#a7b4c7] bg-[#202d40]',
  }

  return (
    <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[#8492a6]">{label}</span>
        <span className={`rounded-lg p-2 ${accentClasses[accent]}`}>
          <Icon size={17} />
        </span>
      </div>
      <div className="mt-4 text-2xl font-semibold text-white">{value}</div>
      <p className="mt-2 text-xs text-[#8492a6]">{detail}</p>
    </div>
  )
}

export default function PaperTrading() {
  const [trades, setTrades] = useState<PaperTrade[]>([])
  const [performance, setPerformance] =
    useState<PerformanceResponse | null>(null)
  const [filter, setFilter] = useState<TradeFilter>('ALL')
  const [symbolFilter, setSymbolFilter] = useState('ALL')
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [selectedTrade, setSelectedTrade] =
    useState<PaperTrade | null>(null)
  const [exitPrice, setExitPrice] = useState('')
  const [costPct, setCostPct] = useState('0.20')
  const [closing, setClosing] = useState(false)
  const [actionMessage, setActionMessage] = useState<string | null>(null)

  const loadData = useCallback(async (manual = false) => {
    try {
      if (manual) setRefreshing(true)
      else setLoading(true)

      setError(null)

      const [tradeResponse, performanceResponse] = await Promise.all([
        getPaperTrades(),
        getPerformance(),
      ])

      if (!tradeResponse.success || !performanceResponse.success) {
        throw new Error('The backend returned an unsuccessful response.')
      }

      setTrades(tradeResponse.trades)
      setPerformance(performanceResponse)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unable to load paper trading data.',
      )
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void Promise.resolve().then(() => loadData())
  }, [loadData])

  const symbols = useMemo(
    () => [...new Set(trades.map((trade) => trade.symbol))].sort(),
    [trades],
  )

  const filteredTrades = useMemo(() => {
    return trades.filter((trade) => {
      const matchesStatus =
        filter === 'ALL' || trade.status === filter
      const matchesSymbol =
        symbolFilter === 'ALL' || trade.symbol === symbolFilter

      return matchesStatus && matchesSymbol
    })
  }, [trades, filter, symbolFilter])

  async function handleCloseTrade() {
    if (!selectedTrade) return

    const price = Number(exitPrice)
    const cost = Number(costPct)

    if (!Number.isFinite(price) || price <= 0) {
      setActionMessage('Enter a valid exit price greater than zero.')
      return
    }

    if (!Number.isFinite(cost) || cost < 0 || cost > 100) {
      setActionMessage('Transaction cost must be between 0 and 100%.')
      return
    }

    try {
      setClosing(true)
      setActionMessage(null)

      const response = await apiClient.post(
        `/api/v1/paper-trades/${selectedTrade.id}/close`,
        {
          exit_price: price,
          estimated_round_trip_cost_pct: cost,
        },
      )

      if (!response.data.success) {
        throw new Error('The backend could not close this trade.')
      }

      setSelectedTrade(null)
      setExitPrice('')
      setActionMessage(
        `Trade #${selectedTrade.id} closed successfully. Net P&L: ${signedCurrency(Number(response.data.net_pnl))}`,
      )

      await loadData(true)
    } catch (err) {
      setActionMessage(
        err instanceof Error
          ? err.message
          : 'Unable to close this paper trade.',
      )
    } finally {
      setClosing(false)
    }
  }

  const openTrades = trades.filter((trade) => trade.status === 'OPEN').length
  const closedTrades = trades.filter((trade) => trade.status === 'CLOSED').length

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-4 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="mb-2 text-xs text-[#8492a6]">
            MOSAIC / WORKSPACE / PAPER TRADING
          </div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold text-white">
              Paper Trading
            </h1>
            <span className="rounded-md border border-[#275043] bg-[#102a27] px-2 py-1 text-[10px] font-semibold tracking-wider text-[#53d6b0]">
              SIMULATION
            </span>
          </div>
          <p className="mt-2 text-sm text-[#8492a6]">
            Monitor simulated positions, review trade history and manage exits.
          </p>
        </div>

        <button
          onClick={() => void loadData(true)}
          disabled={refreshing || loading}
          className="flex items-center gap-2 rounded-lg border border-[#26364a] bg-[#111d2d] px-4 py-2.5 text-sm text-slate-200 transition hover:bg-[#18263a] disabled:opacity-50"
        >
          <RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      <div className="flex items-center gap-2 rounded-lg border border-[#493e25] bg-[#29251a] px-4 py-3 text-xs text-amber-300">
        <Activity size={16} />
        New paper trades are disabled until model production approval.
        Existing simulated trades can still be reviewed and closed. No real
        broker orders are placed.
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

      {actionMessage && (
        <div className="flex items-start justify-between gap-3 rounded-lg border border-[#26364a] bg-[#111d2d] p-4 text-sm text-slate-200">
          <span>{actionMessage}</span>
          <button
            onClick={() => setActionMessage(null)}
            aria-label="Dismiss message"
          >
            <X size={16} />
          </button>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <SummaryCard
          label="Total Trades"
          value={loading ? '—' : String(performance?.total_trades ?? trades.length)}
          detail="All recorded simulations"
          icon={Activity}
        />
        <SummaryCard
          label="Open Positions"
          value={loading ? '—' : String(openTrades)}
          detail="Currently active simulations"
          icon={Wallet}
          accent="blue"
        />
        <SummaryCard
          label="Closed Trades"
          value={loading ? '—' : String(closedTrades)}
          detail="Completed simulations"
          icon={Activity}
          accent="neutral"
        />
        <SummaryCard
          label="Net Realized P&L"
          value={
            loading || !performance
              ? '—'
              : signedCurrency(performance.net_pnl)
          }
          detail="After estimated costs"
          icon={
            performance && performance.net_pnl < 0
              ? ArrowDownRight
              : ArrowUpRight
          }
          accent={
            !performance || performance.net_pnl === 0
              ? 'neutral'
              : performance.net_pnl > 0
                ? 'green'
                : 'red'
          }
        />
      </div>

      {selectedTrade && (
        <section className="rounded-xl border border-[#275043] bg-[#0e1b25] p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="font-semibold text-white">
                Close simulated position · {selectedTrade.symbol}
              </h2>
              <p className="mt-1 text-xs text-[#8492a6]">
                Trade #{selectedTrade.id} · {selectedTrade.direction} · Entry{' '}
                {currency(selectedTrade.entry_price)}
              </p>
            </div>
            <button
              onClick={() => setSelectedTrade(null)}
              aria-label="Cancel closing trade"
              className="text-[#8492a6] hover:text-white"
            >
              <X size={18} />
            </button>
          </div>

          <div className="mt-5 grid gap-4 sm:grid-cols-2">
            <label className="text-xs text-[#8492a6]">
              Exit price (₹)
              <input
                type="number"
                min="0.01"
                step="0.01"
                value={exitPrice}
                onChange={(event) => setExitPrice(event.target.value)}
                placeholder="Enter simulated exit price"
                className="mt-2 w-full rounded-lg border border-[#26364a] bg-[#080d17] px-3 py-3 text-sm text-white outline-none focus:border-[#53d6b0]"
              />
            </label>
            <label className="text-xs text-[#8492a6]">
              Estimated round-trip cost (%)
              <input
                type="number"
                min="0"
                max="100"
                step="0.01"
                value={costPct}
                onChange={(event) => setCostPct(event.target.value)}
                className="mt-2 w-full rounded-lg border border-[#26364a] bg-[#080d17] px-3 py-3 text-sm text-white outline-none focus:border-[#53d6b0]"
              />
            </label>
          </div>

          <div className="mt-4 flex flex-wrap gap-3">
            <button
              onClick={() => void handleCloseTrade()}
              disabled={closing}
              className="rounded-lg bg-[#176b56] px-4 py-2.5 text-sm font-semibold text-white hover:bg-[#1c8067] disabled:opacity-50"
            >
              {closing ? 'Closing...' : 'Confirm simulated close'}
            </button>
            <button
              onClick={() => setSelectedTrade(null)}
              disabled={closing}
              className="rounded-lg border border-[#26364a] px-4 py-2.5 text-sm text-slate-300 hover:bg-[#18263a]"
            >
              Cancel
            </button>
          </div>
          <p className="mt-3 text-[11px] text-[#8492a6]">
            This records a simulated exit and calculates illustrative P&L. It
            does not send an order to a broker.
          </p>
        </section>
      )}

      <section className="overflow-hidden rounded-xl border border-[#202d40] bg-[#0e1625]">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#202d40] p-5">
          <div>
            <h2 className="text-sm font-semibold text-white">
              Trade Activity
            </h2>
            <p className="mt-1 text-xs text-[#8492a6]">
              Open positions and completed paper trades
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            {(['ALL', 'OPEN', 'CLOSED'] as TradeFilter[]).map((item) => (
              <button
                key={item}
                onClick={() => setFilter(item)}
                className={`rounded-lg px-3 py-2 text-xs transition ${
                  filter === item
                    ? 'border border-[#275043] bg-[#153238] text-[#53d6b0]'
                    : 'border border-[#26364a] bg-[#111d2d] text-[#8492a6] hover:text-white'
                }`}
              >
                {item === 'ALL' ? 'All trades' : item}
              </button>
            ))}
            <select
              value={symbolFilter}
              onChange={(event) => setSymbolFilter(event.target.value)}
              className="rounded-lg border border-[#26364a] bg-[#111d2d] px-3 py-2 text-xs text-slate-200 outline-none"
            >
              <option value="ALL">All symbols</option>
              {symbols.map((symbol) => (
                <option key={symbol} value={symbol}>
                  {symbol}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[1050px] text-left text-xs">
            <thead className="bg-[#111d2d] text-[#8492a6]">
              <tr>
                {[
                  'Trade',
                  'Symbol',
                  'Direction',
                  'Quantity',
                  'Entry Price',
                  'Exit Price',
                  'Stop Loss',
                  'Net P&L',
                  'Status',
                  'Action',
                ].map((heading) => (
                  <th key={heading} className="px-4 py-4 font-medium">
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredTrades.map((trade) => (
                <tr
                  key={trade.id}
                  className="border-t border-[#202d40] hover:bg-[#111d2d]/60"
                >
                  <td className="px-4 py-4 text-[#8492a6]">#{trade.id}</td>
                  <td className="px-4 py-4 font-semibold text-white">
                    {trade.symbol}
                    <div className="mt-1 text-[10px] font-normal text-[#617087]">
                      {dateTime(trade.entry_time)}
                    </div>
                  </td>
                  <td
                    className={`px-4 py-4 font-semibold ${
                      trade.direction === 'LONG'
                        ? 'text-[#53d6b0]'
                        : 'text-[#f16b78]'
                    }`}
                  >
                    {trade.direction}
                  </td>
                  <td className="px-4 py-4 text-slate-300">
                    {trade.quantity}
                  </td>
                  <td className="px-4 py-4 text-slate-300">
                    {currency(trade.entry_price)}
                  </td>
                  <td className="px-4 py-4 text-slate-300">
                    {trade.exit_price === null
                      ? '—'
                      : currency(trade.exit_price)}
                  </td>
                  <td className="px-4 py-4 text-slate-300">
                    {currency(trade.stop_loss)}
                  </td>
                  <td
                    className={`px-4 py-4 font-semibold ${
                      trade.net_pnl === null
                        ? 'text-[#8492a6]'
                        : trade.net_pnl >= 0
                          ? 'text-[#53d6b0]'
                          : 'text-[#f16b78]'
                    }`}
                  >
                    {trade.net_pnl === null
                      ? '—'
                      : signedCurrency(trade.net_pnl)}
                  </td>
                  <td className="px-4 py-4">
                    <span
                      className={`rounded-md border px-2 py-1 text-[10px] ${
                        trade.status === 'CLOSED'
                          ? 'border-[#275043] bg-[#102a27] text-[#53d6b0]'
                          : 'border-[#493e25] bg-[#29251a] text-amber-300'
                      }`}
                    >
                      {trade.status}
                    </span>
                  </td>
                  <td className="px-4 py-4">
                    {trade.status === 'OPEN' ? (
                      <button
                        onClick={() => {
                          setSelectedTrade(trade)
                          setExitPrice('')
                          setActionMessage(null)
                        }}
                        className="rounded-md border border-[#275043] bg-[#102a27] px-3 py-1.5 text-[11px] font-medium text-[#53d6b0] hover:bg-[#153b34]"
                      >
                        Close
                      </button>
                    ) : (
                      <span className="text-[#617087]">Completed</span>
                    )}
                  </td>
                </tr>
              ))}

              {!loading && filteredTrades.length === 0 && (
                <tr>
                  <td
                    colSpan={10}
                    className="px-5 py-12 text-center text-[#8492a6]"
                  >
                    No trades match the selected filters.
                  </td>
                </tr>
              )}

              {loading && (
                <tr>
                  <td
                    colSpan={10}
                    className="px-5 py-12 text-center text-[#8492a6]"
                  >
                    Loading paper trades...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="border-t border-[#202d40] px-5 py-3 text-[10px] text-[#8492a6]">
          Showing {filteredTrades.length} of {trades.length} recorded trades.
          P&L is calculated from simulated entries and exits.
        </div>
      </section>

      <footer className="flex flex-wrap items-center justify-between gap-2 pb-3 text-[10px] text-[#617087]">
        <span>MOSAIC 2.0 · PAPER TRADING ENVIRONMENT</span>
        <span>SIMULATION ONLY · NO REAL BROKER ORDERS</span>
      </footer>
    </div>
  )
}
