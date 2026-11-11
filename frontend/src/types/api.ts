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
  data_status?: string
  market_status?: string
  is_current_session?: boolean
  input_candles?: number
  last_candle_timestamp?: string
}

export interface PredictionResult {
  symbol: string
  prediction: 'DOWN' | 'SIDEWAYS' | 'UP'
  probabilities: {
    DOWN: number
    SIDEWAYS: number
    UP: number
  }
  candle_timestamp: string
  latest_market_candle_timestamp?: string
  prediction_age_candles?: number
  is_latest_candle_prediction?: boolean
  prediction_horizon_minutes: number
  model_version: string
  status: string
}

export interface PredictionResponse {
  success: boolean
  symbol?: string
  data_source: string
  synthetic_data: boolean
  trading_date?: string
  timeframe?: string
  input_candles: number
  completed_5min_candles: number
  last_candle_timestamp?: string
  prediction_result: PredictionResult
  execution?: {
    mode: string
    real_order_placed: boolean
    broker_order_execution_enabled: boolean
  }
  warning: string
}

export interface StrategyResult {
  symbol: string
  decision: string
  direction: string | null
  prediction: 'DOWN' | 'SIDEWAYS' | 'UP'
  probabilities: {
    DOWN: number
    SIDEWAYS: number
    UP: number
  }
  candle_timestamp: string
  prediction_horizon_minutes: number
  model_version: string
  model_status: string
  risk_accepted: boolean
  entry_price: number | null
  stop_loss: number | null
  quantity: number
  position_value: number
  estimated_total_risk: number
  risk_reason: string
  paper_trading_only: boolean
  real_order_placed: boolean
  reason: string
}

export interface LiveStrategyResponse {
  success: boolean
  symbol: string
  data_source: string
  synthetic_data: boolean
  trading_date: string
  timeframe: string
  input_candles: number
  completed_5min_candles: number
  last_candle_timestamp: string
  prediction_result: PredictionResult
  strategy_result: StrategyResult
  paper_trade_eligible: boolean
  execution: {
    mode: string
    real_order_placed: boolean
    broker_connected: boolean
  }
  warning: string
}
