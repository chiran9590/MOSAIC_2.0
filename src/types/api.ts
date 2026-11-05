export interface EquityPoint {
  trade_id: number
  exit_time: string
  net_pnl: number
  equity: number
  peak_equity: number
  drawdown: number
  drawdown_pct: number
}

export interface PerformanceResponse {
  success: boolean
  total_trades: number
  open_trades: number
  closed_trades: number
  winning_trades: number
  losing_trades: number
  breakeven_trades: number
  gross_pnl: number
  estimated_cost: number
  net_pnl: number
  win_rate_pct: number
  average_net_pnl: number
  starting_capital: number
  current_equity: number
  cumulative_pnl: number
  max_drawdown: number
  max_drawdown_pct: number
  equity_curve: EquityPoint[]
  execution_mode: string
  real_order_placed: boolean
  warning: string
}

export interface PaperTrade {
  id: number
  symbol: string
  direction: string
  prediction: string
  candle_timestamp: string
  entry_time: string
  entry_price: number
  stop_loss: number
  quantity: number
  position_value: number
  estimated_risk: number
  probabilities: string
  model_version: string
  data_source: string
  synthetic_data: number
  status: string
  exit_price: number | null
  exit_time: string | null
  gross_pnl: number | null
  estimated_cost: number | null
  net_pnl: number | null
  created_at: string
}

export interface PaperTradesResponse {
  success: boolean
  total: number
  trades: PaperTrade[]
}

export interface MarketCandle {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface MarketResponse {
  success: boolean
  symbol: string
  timeframe: string
  data_source: string
  synthetic_data: boolean
  trading_date: string
  total_candles: number
  candles: MarketCandle[]
  warning: string
}
