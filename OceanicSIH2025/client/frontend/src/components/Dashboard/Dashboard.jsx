import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../services/api'
import StatsPanel from './StatsPanel'
import ProviderButtons from './ProviderButtons'
import IndobisDashboard from '../Ai-Model/IndobisDashboard'

// Recharts
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Legend,
  CartesianGrid
} from 'recharts'
import InsightsUI from '../../InsightsUI'

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#845EC2', '#D65DB1', '#FF6F91']

export default function Dashboard() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [records, setRecords] = useState([])

  // Derived provider-specific data
  const fisheries = useMemo(
    () => records.filter(r => r.source === 'data.gov.in' && r.year),
    [records]
  )

  // Chart Data
  const chartData = useMemo(() => {
    const sourcesMap = {}
    records.forEach(r => {
      const key = r.source || 'Unknown'
      sourcesMap[key] = (sourcesMap[key] || 0) + 1
    })
    return Object.entries(sourcesMap).map(([name, value]) => ({ name, value }))
  }, [records])

  const speciesData = useMemo(() => {
    const speciesMap = {}
    records.forEach(r => {
      if (r.species) {
        speciesMap[r.species] = (speciesMap[r.species] || 0) + 1
      }
    })
    return Object.entries(speciesMap).map(([name, count]) => ({ name, count }))
  }, [records])

  // KPIs
  const kpis = useMemo(() => {
    const total = records.length
    const providers = new Set(records.map(r => (r.source || '').split('/')[0]))
    const speciesCount = new Set(records.filter(r => r.species).map(r => r.species)).size
    const latestYear = fisheries.reduce(
      (acc, v) => Math.max(acc, Number(String(v.year).slice(0, 4)) || 0),
      0
    )
    return { total, providers: providers.size, speciesCount, latestYear }
  }, [records, fisheries])

  // Fetch + Ingest
  async function fetchData() {
    setLoading(true)
    setError('')
    try {
      const data = await api.getData()
      setRecords(data)
    } catch (e) {
      console.error(e)
      setError('Failed to load data. Try ingesting providers below.')
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { fetchData() }, [])

  const providers = [
    { key: 'fisheries', label: 'Fisheries' },
    { key: 'obis', label: 'OBIS' },
    { key: 'worms', label: 'WoRMS' },
    { key: 'bold', label: 'BOLD' },
    { key: 'cmfri', label: 'CMFRI PDFs' },
    { key: 'open-meteo', label: 'Open-Meteo' },
    { key: 'noaa', label: 'NOAA' },
  ]
  function getLastSync(providerKey) {
    const v = localStorage.getItem(`lastSync:${providerKey}`)
    if (!v) return 'Never'
    try { return new Date(Number(v)).toLocaleString() } catch { return '—' }
  }

  // Custom Pie Chart Label
  const renderPieLabel = ({ name, value }) => `${name}: ${value}`

  return (
    <div className="container">
      {/* Charts at the top */}
      <div className="grid grid-2" style={{ gap: 24, marginBottom: 32 }}>
        <div className="card">
          <div className="card-body">
            <h2 className="card-title">Records by Provider</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={chartData}
                  dataKey="value"
                  nameKey="name"
                  cx="40%"
                  cy="50%"
                  outerRadius={100}
                  labelLine={true} // enables lines connecting slices
                  label={renderPieLabel}
                  isAnimationActive={true}
                  animationDuration={500}
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend layout="horizontal" verticalAlign="bottom" align="center" />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <h2 className="card-title">Species Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={speciesData} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{fontSize:11, fill: '#FAF7F0' }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#82ca9d" isAnimationActive={true} animationDuration={500} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Stats Panel */}
      <StatsPanel kpis={kpis} onTotalClick={() => navigate('/dashboard/records')} />

      <div className="grid grid-3" style={{ gap: 24 }}>
        {/* Left Column */}
        <div className="card" style={{ gridColumn: 'span 2' }}>
          <div className="card-body">
            <h2 className="card-title">Data Providers</h2>
            {error && (
              <div style={{ background: '#3b1d2a', color: '#ffb4c8', padding: 10, borderRadius: 10, border: '1px solid rgba(255,255,255,0.06)', marginBottom: 8 }}>{error}</div>
            )}
            <div className="grid grid-2" style={{ gap: 16 }}>
              {providers.map(p => (
                <div key={p.key} className="p-3 rounded-md border border-slate-800 bg-slate-900/40 flex items-center justify-between">
                  <div>
                    <div className="text-slate-100 font-semibold">{p.label}</div>
                    <div className="text-xs text-slate-400">Last sync: {getLastSync(p.key)}</div>
                  </div>
                  <span className="badge bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 rounded-full px-2 py-1 text-xs">Ready</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="card">
          <div className="card-body">
            <h2 className="card-title">Ingest Providers</h2>
            <ProviderButtons onAfterIngest={fetchData} />
          </div>
        </div>
        <div className="card">
          <div className="card-body">
            <h2 className="card-title">Ingest Providers</h2>
            <IndobisDashboard />
          </div>
        </div>
        <div className="card">
          <div className="card-body">
            <h2 className="card-title">Ingest Providers</h2>
            <InsightsUI />
          </div>
        </div>
      </div>
    </div>
  )
}


