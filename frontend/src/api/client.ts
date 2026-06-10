import axios from 'axios'
import { getApiBaseUrl } from '../lib/apiBaseUrl'

const client = axios.create({
  baseURL: getApiBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
  },
})

export default client
