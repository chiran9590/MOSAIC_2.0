import axios from 'axios'

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/'

export const apiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 10000,
  headers: {
    Accept: 'application/json',
  },
})
