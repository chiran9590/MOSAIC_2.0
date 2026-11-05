import { apiClient } from './client'

import type {
  PerformanceResponse,
  PaperTradesResponse,
  MarketResponse,
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
    '/api/v1/market/mock'
  )
  return response.data
}
