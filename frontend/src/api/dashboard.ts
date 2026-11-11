import { apiClient } from './client'

import type {
  PerformanceResponse,
  PaperTradesResponse,
  MarketResponse,
  PredictionResponse,
  LiveStrategyResponse,
} from '../types/api'

export const getPerformance = async () => {
  const response = await apiClient.get<PerformanceResponse>(
    '/api/v1/paper-trades/performance'
  )
  return response.data
}

export const getPaperTrades = async () => {
  const response = await apiClient.get<PaperTradesResponse>(
    '/api/v1/paper-trades'
  )
  return response.data
}

export const getOpenPaperTrades = async () => {
  const response = await apiClient.get<PaperTradesResponse>(
    '/api/v1/paper-trades/open'
  )
  return response.data
}

export const getMarketData = async () => {
  const response = await apiClient.get<MarketResponse>(
    '/api/v1/market/live'
  )
  return response.data
}

export const getMockPrediction = async () => {
  const response = await apiClient.post<PredictionResponse>(
    '/api/v1/predict/mock'
  )

  return response.data
}

export const getLivePrediction = async () => {
  const response = await apiClient.post<PredictionResponse>(
    '/api/v1/predict/live',
    { symbol: 'RELIANCE' }
  )

  return response.data
}

export const getLiveStrategy = async () => {
  const response = await apiClient.post<LiveStrategyResponse>(
    '/api/v1/strategy/live',
    { symbol: 'RELIANCE' }
  )

  return response.data
}
