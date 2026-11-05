import { useEffect, useState } from 'react'
import { getPaperTrades, getPerformance } from './api/dashboard'
import type { PaperTrade, PerformanceResponse } from './types/api'
import {
  Activity,
  BarChart3,
  Bell,
  CandlestickChart,
  ChevronDown,
  CircleDollarSign,
  Clock3,
  Command,
  LayoutDashboard,
  Menu,
  Radio,
  Settings,
  ShieldCheck,
  TrendingUp,
  Wallet,
  X,
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

const navigation = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Market', icon: CandlestickChart },
  { label: 'Predictions', icon: Activity },
  { label: 'Strategy', icon: ShieldCheck },
  { label: 'Paper Trading', icon: Wallet },
  { label: 'Analytics', icon: BarChart3 },
]

function formatCurrency(value: number) {
  return `₹${Math.abs(value).toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

function formatSignedCurrency(value: number) {
  const sign = value > 0 ? '+' : value < 0 ? '-' : ''
  return `${sign}${formatCurrency(value)}`
}

function MetricCard({
  label,
  value,
  change,
  icon: Icon,
  positive,
}: {
  label: string
  value: string
  change: string
  icon: typeof Wallet
  positive?: boolean
}) {
  return (
    <div className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
      <div className="flex items-center justify-between">
        <span className="text-sm text-[#8492a6]">{label}</span>
        <div className="rounded-lg bg-[#18263a] p-2">
          <Icon size={17} className="text-[#70a7f8]" />
        </div>
      </div>
      <div className="mt-4 text-2xl font-semibold tracking-tight text-slate-100">
        {value}
      </div>
      <div className={`mt-2 text-xs ${positive ? 'text-[#20c997]' : 'text-[#8492a6]'}`}>
        {change}
      </div>
    </div>
  )
}

function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [performance, setPerformance] = useState<PerformanceResponse | null>(null)
  const [trades, setTrades] = useState<PaperTrade[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function loadDashboard() {
      try {
        setLoading(true)
        setError(null)

        const [performanceData, tradesData] = await Promise.all([
          getPerformance(),
          getPaperTrades(),
        ])

        if (!performanceData.success || !tradesData.success) {
          throw new Error('The backend returned an unsuccessful response.')
        }

        if (!cancelled) {
          setPerformance(performanceData)
          setTrades(tradesData.trades)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unable to load dashboard data.')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void loadDashboard()

    return () => {
      cancelled = true
    }
  }, [])

  const equityData = performance
    ? [
        { time: 'Start', equity: performance.starting_capital },
        ...performance.equity_curve.map((point) => ({
          time: new Date(point.exit_time).toLocaleTimeString('en-IN', {
            hour: '2-digit',
            minute: '2-digit',
          }),
          equity: point.equity,
        })),
      ]
    : []

  return (
    <div className="min-h-screen bg-[#080d17] text-slate-200">
      <div className="flex min-h-screen">
        {sidebarOpen && (
          <button
            aria-label="Close sidebar"
            className="fixed inset-0 z-30 bg-black/60 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        <aside
          className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-[#202d40] bg-[#0b1321] transition-transform lg:static lg:translate-x-0 ${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          <div className="flex h-[76px] items-center justify-between border-b border-[#202d40] px-6">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#153b3c]">
                <Activity size={21} className="text-[#20c997]" />
              </div>
              <div>
                <div className="text-lg font-bold tracking-[0.12em] text-white">
                  MOSAIC
                </div>
                <div className="text-[10px] tracking-[0.2em] text-[#8492a6]">
                  TRADING SYSTEM 2.0
                </div>
              </div>
            </div>
            <button
              aria-label="Close menu"
              className="text-slate-400 lg:hidden"
              onClick={() => setSidebarOpen(false)}
            >
              <X size={19} />
            </button>
          </div>

          <div className="px-4 pt-7">
            <div className="mb-3 px-3 text-[10px] font-semibold tracking-[0.18em] text-[#617087]">
              WORKSPACE
            </div>
            <nav className="space-y-1">
              {navigation.map(({ label, icon: Icon }) => (
                <button
                  key={label}
                  onClick={() => {
                    setActivePage(label)
                    setSidebarOpen(false)
                  }}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-sm transition ${
                    activePage === label
                      ? 'border border-[#214a4b] bg-[#153238] font-medium text-[#53d6b0]'
                      : 'text-[#94a3b8] hover:bg-[#142033] hover:text-white'
                  }`}
                >
                  <Icon size={18} />
                  {label}
                  {label === 'Paper Trading' && (
                    <span className="ml-auto rounded bg-[#20352f] px-2 py-0.5 text-[10px] text-[#53d6b0]">
                      SIM
                    </span>
                  )}
                </button>
              ))}
            </nav>
          </div>

          <div className="mt-auto border-t border-[#202d40] p-4">
            <button
              onClick={() => setActivePage('Settings')}
              className="flex w-full items-center gap-3 rounded-lg px-3 py-3 text-sm text-[#94a3b8] hover:bg-[#142033]"
            >
              <Settings size={18} />
              Settings
            </button>
            <div className="mt-4 rounded-xl border border-[#26364a] bg-[#111d2d] p-3">
              <div className="flex items-center gap-2 text-xs font-medium text-[#53d6b0]">
                <span className="h-2 w-2 rounded-full bg-[#20c997]" />
                PAPER MODE ACTIVE
              </div>
              <p className="mt-2 text-[11px] leading-5 text-[#8492a6]">
                Simulation only. No real broker orders are placed.
              </p>
            </div>
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <header className="sticky top-0 z-20 flex h-[76px] items-center justify-between border-b border-[#202d40] bg-[#0b1321]/95 px-4 backdrop-blur md:px-8">
            <div className="flex items-center gap-3">
              <button
                aria-label="Open navigation"
                className="text-slate-400 lg:hidden"
                onClick={() => setSidebarOpen(true)}
              >
                <Menu size={21} />
              </button>
              <div>
                <div className="text-sm font-semibold text-white">{activePage}</div>
                <div className="mt-1 text-[11px] text-[#8492a6]">
                  MOSAIC / Workspace / {activePage}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 md:gap-5">
              <div className="hidden items-center gap-2 text-xs text-[#94a3b8] sm:flex">
                <Clock3 size={14} />
                NSE MARKET
              </div>
              <div className="hidden items-center gap-2 rounded-md border border-[#275043] bg-[#102a27] px-3 py-2 text-[11px] font-semibold text-[#53d6b0] sm:flex">
                <Radio size={14} />
                PAPER TRADING
              </div>
              <button aria-label="Notifications" className="relative text-[#8492a6] hover:text-white">
                <Bell size={19} />
                <span className="absolute -right-1 -top-1 h-2 w-2 rounded-full bg-[#5295f5]" />
              </button>
              <div className="flex items-center gap-2 border-l border-[#26364a] pl-3 md:pl-5">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#233750] text-xs font-semibold text-blue-200">
                  M
                </div>
                <span className="hidden text-xs text-slate-300 sm:block">MOSAIC User</span>
                <ChevronDown size={14} className="text-[#8492a6]" />
              </div>
            </div>
          </header>

          <div className="mx-auto max-w-[1600px] space-y-6 p-4 md:p-8">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h1 className="text-2xl font-semibold tracking-tight text-white">
                  Trading Overview
                </h1>
                <p className="mt-1 text-sm text-[#8492a6]">
                  Monitor your portfolio, market and strategy performance.
                </p>
              </div>
              <div className="flex items-center gap-2 rounded-lg border border-[#275043] bg-[#102a27] px-3 py-2 text-xs text-[#53d6b0]">
                <span className="h-2 w-2 rounded-full bg-[#20c997]" />
                SIMULATION ENVIRONMENT
              </div>
            </div>

            {error && (
              <div className="rounded-lg border border-[#713b42] bg-[#2b1920] px-4 py-3 text-xs text-[#f4a0a8]">
                Dashboard API error: {error}. Check that the FastAPI backend is running on port 8001.
              </div>
            )}
            {loading && (
              <div className="text-xs text-[#8492a6]">
                Loading dashboard data from MOSAIC backend...
              </div>
            )}

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label="Portfolio Equity"
                value={performance ? formatCurrency(performance.current_equity) : '—'}
                change={performance ? `Starting capital ${formatCurrency(performance.starting_capital)}` : 'Loading...'}
                icon={Wallet}
              />
              <MetricCard
                label="Net P&L"
                value={performance ? formatSignedCurrency(performance.net_pnl) : '—'}
                change={performance ? `After estimated costs · ${performance.closed_trades} closed trades` : 'Recorded paper-trading results'}
                icon={TrendingUp}
                positive={performance ? performance.net_pnl >= 0 : false}
              />
              <MetricCard
                label="Win Rate"
                value={performance ? `${performance.win_rate_pct.toFixed(2)}%` : '—'}
                change={performance ? `${performance.winning_trades} wins · ${performance.closed_trades} closed trades` : 'Based on closed paper trades'}
                icon={CircleDollarSign}
                positive={performance ? performance.win_rate_pct >= 50 : false}
              />
              <MetricCard
                label="Maximum Drawdown"
                value={performance ? `${performance.max_drawdown_pct.toFixed(2)}%` : '—'}
                change="Realized closed-trade equity only"
                icon={BarChart3}
              />
            </div>

            <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
              <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5 xl:col-span-2">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="text-sm font-semibold text-white">Portfolio Performance</h2>
                    <p className="mt-1 text-xs text-[#8492a6]">Realized equity · Closed paper trades · MOCK data</p>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-[#53d6b0]">
                    <span className="h-2 w-2 rounded-full bg-[#20c997]" />
                    Equity
                  </div>
                </div>
                <div className="mt-6 h-[260px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={equityData}>
                      <defs>
                        <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#20c997" stopOpacity={0.25} />
                          <stop offset="95%" stopColor="#20c997" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#243246" strokeDasharray="3 5" vertical={false} />
                      <XAxis
                        dataKey="time"
                        tick={{ fill: '#8492a6', fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        domain={['dataMin - 20', 'dataMax + 20']}
                        tick={{ fill: '#8492a6', fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(value: number) => `₹${(value / 1000).toFixed(1)}k`}
                        width={58}
                      />
                      <Tooltip
                        contentStyle={{
                          background: '#111d2d',
                          border: '1px solid #304158',
                          borderRadius: 8,
                          color: '#e2e8f0',
                        }}
                        formatter={(value) => [`₹${Number(value).toLocaleString('en-IN')}`, 'Equity']}
                      />
                      <Area
                        type="monotone"
                        dataKey="equity"
                        stroke="#20c997"
                        strokeWidth={2}
                        fill="url(#equityGradient)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="rounded-xl border border-[#202d40] bg-[#0e1625] p-5">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-white">System Status</h2>
                  <Command size={17} className="text-[#8492a6]" />
                </div>
                <div className="mt-5 space-y-4">
                  {[
                    ['Execution Mode', 'Paper only', 'good'],
                    ['Market Data', 'MOCK source', 'warn'],
                    ['Prediction Engine', 'Experimental', 'warn'],
                    ['Risk Manager', 'Enabled', 'good'],
                    ['Real Orders', 'Disabled', 'good'],
                  ].map(([label, value, status]) => (
                    <div key={label} className="flex items-center justify-between gap-2">
                      <span className="text-xs text-[#94a3b8]">{label}</span>
                      <span
                        className={`flex items-center gap-2 text-xs ${
                          status === 'good' ? 'text-[#53d6b0]' : 'text-amber-400'
                        }`}
                      >
                        <span
                          className={`h-1.5 w-1.5 rounded-full ${
                            status === 'good' ? 'bg-[#20c997]' : 'bg-amber-400'
                          }`}
                        />
                        {value}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-6 rounded-lg border border-[#493e25] bg-[#29251a] p-3">
                  <div className="text-xs font-semibold text-amber-300">Research environment</div>
                  <p className="mt-2 text-[11px] leading-5 text-[#b5a98c]">
                    Predictions and performance are experimental. Historical or simulated
                    results do not establish live profitability.
                  </p>
                </div>
              </section>
            </div>

            <section className="overflow-hidden rounded-xl border border-[#202d40] bg-[#0e1625]">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#202d40] p-5">
                <div>
                  <h2 className="text-sm font-semibold text-white">Recent Paper Trades</h2>
                  <p className="mt-1 text-xs text-[#8492a6]">Recorded simulation activity</p>
                </div>
                <button
                  onClick={() => setActivePage('Paper Trading')}
                  className="text-xs text-[#70a7f8] hover:text-blue-300"
                >
                  View all trades →
                </button>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[700px] text-left text-xs">
                  <thead className="bg-[#111d2d] text-[#8492a6]">
                    <tr>
                      {['Symbol', 'Direction', 'Quantity', 'Entry Price', 'Exit Price', 'Net P&L', 'Status'].map(
                        (heading) => (
                          <th key={heading} className="px-5 py-4 font-medium">{heading}</th>
                        ),
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {trades.length > 0 ? (
                      trades.map((trade) => (
                        <tr key={trade.id} className="border-t border-[#202d40]">
                          <td className="px-5 py-4 font-semibold text-white">{trade.symbol}</td>
                          <td className={`px-5 py-4 ${trade.direction === 'LONG' ? 'text-[#20c997]' : 'text-[#f16b78]'}`}>
                            {trade.direction}
                          </td>
                          <td className="px-5 py-4 text-slate-300">{trade.quantity}</td>
                          <td className="px-5 py-4 text-slate-300">{formatCurrency(trade.entry_price)}</td>
                          <td className="px-5 py-4 text-slate-300">
                            {trade.exit_price === null ? '—' : formatCurrency(trade.exit_price)}
                          </td>
                          <td className={`px-5 py-4 font-semibold ${
                            trade.net_pnl === null ? 'text-[#8492a6]' :
                            trade.net_pnl >= 0 ? 'text-[#20c997]' : 'text-[#f16b78]'
                          }`}>
                            {trade.net_pnl === null ? '—' : formatSignedCurrency(trade.net_pnl)}
                          </td>
                          <td className="px-5 py-4">
                            <span className={`rounded-md border px-2 py-1 text-[10px] ${
                              trade.status === 'CLOSED'
                                ? 'border-[#275043] bg-[#102a27] text-[#53d6b0]'
                                : 'border-[#493e25] bg-[#29251a] text-amber-300'
                            }`}>
                              {trade.status}
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={7} className="px-5 py-8 text-center text-[#8492a6]">
                          {loading ? 'Loading trades...' : 'No paper trades recorded yet.'}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div className="border-t border-[#202d40] px-5 py-3 text-[10px] text-[#8492a6]">
                Showing {trades.length} recorded paper trade(s). MOCK data is synthetic and is not live market data.
              </div>
            </section>

            <footer className="flex flex-wrap items-center justify-between gap-2 pb-3 text-[10px] text-[#617087]">
              <span>MOSAIC 2.0 · EXPERIMENTAL MARKET RESEARCH PLATFORM</span>
              <span>NO REAL BROKER ORDERS · PAPER TRADING ONLY</span>
            </footer>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
