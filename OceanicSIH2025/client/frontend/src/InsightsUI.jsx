// import { useEffect, useState } from "react"
// import { Card, CardContent } from "./components/ui/card"
// import { Button } from "./components/ui/button"
// import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select"
// import { motion } from "framer-motion"
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts"

// const cachedInsights = {
//   "Katsuwonus pelamis": {
//     sst_points: [27.8, 28.3, 29.1],
//     insight: "Found mostly in tropical waters (27–29°C). Highly sensitive to SST rise; may impact Indian Ocean tuna fisheries."
//   },
//   "Sardinella longiceps": {
//     sst_points: [27.9, 28.5],
//     insight: "Strong presence along India’s west coast; critical for small-scale fisheries. Distribution overlaps with warming hotspots."
//   },
//   "Thunnus thynnus": {
//     sst_points: [21.4, 22.1],
//     insight: "Prefers temperate waters (21–22°C). Rising SST may shift habitats northward, affecting Atlantic fisheries."
//   }
// }

// export default function InsightsUI() {
//   const [species, setSpecies] = useState("Katsuwonus pelamis")
//   const [loading, setLoading] = useState(false)
//   const [insightData, setInsightData] = useState({})

//   const fetchInsights = async () => {
//     setLoading(true)

//     // Simulate fetching from backend
//     setTimeout(() => {
//       setInsightData(cachedInsights[species] || {})
//       setLoading(false)
//     }, 1500)
//   }

//   useEffect(() => {
//     fetchInsights()
//   }, [species])

//   return (
//     <div className="p-6 space-y-6 max-w-5xl mx-auto">
//       <motion.h1 className="text-3xl font-bold text-center">
//         🌊 Marine Species Insights Dashboard
//       </motion.h1>

//       {/* Species Selector */}
//       <div className="flex justify-center">
//         <Select value={species} onValueChange={(val) => setSpecies(val)}>
//           <SelectValue placeholder="Select a Species" />
//           <SelectContent>
//             {Object.keys(cachedInsights).map((sp) => (
//               <SelectItem key={sp} value={sp}>{sp}</SelectItem>
//             ))}
//           </SelectContent>
//         </Select>
//       </div>

//       {/* Insights Section */}
//       <Card className="shadow-xl rounded-2xl">
//         <CardContent className="p-6">
//           {loading ? (
//             <motion.div
//               className="text-center text-blue-500"
//               animate={{ opacity: [0.5, 1, 0.5] }}
//               transition={{ repeat: Infinity, duration: 1.5 }}
//             >
//               Fetching insights for <b>{species}</b>...
//             </motion.div>
//           ) : (
//             <>
//               <h2 className="text-xl font-semibold mb-4">Environmental Correlation</h2>

//               {insightData.sst_points ? (
//                 <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
//                   {/* Aggregated stats */}
//                   <div className="space-y-2">
//                     <p><b>Average SST:</b> {(
//                       insightData.sst_points.reduce((a, b) => a + b, 0) /
//                       insightData.sst_points.length
//                     ).toFixed(2)} °C</p>
//                     <p><b>Min SST:</b> {Math.min(...insightData.sst_points)} °C</p>
//                     <p><b>Max SST:</b> {Math.max(...insightData.sst_points)} °C</p>
//                     <p className="text-blue-700 mt-3"><b>Policy Insight:</b> {insightData.insight}</p>
//                   </div>

//                   {/* Chart */}
//                   <BarChart width={400} height={250} data={insightData.sst_points.map((v, i) => ({ idx: i + 1, value: v }))}>
//                     <CartesianGrid strokeDasharray="3 3" />
//                     <XAxis dataKey="idx" label={{ value: "Sample Points", position: "insideBottom", offset: -5 }} />
//                     <YAxis label={{ value: "SST (°C)", angle: -90, position: "insideLeft" }} />
//                     <Tooltip />
//                     <Bar dataKey="value" fill="#2563eb" />
//                   </BarChart>
//                 </div>
//               ) : (
//                 <p className="text-gray-500">No SST data available for this species.</p>
//               )}
//             </>
//           )}
//         </CardContent>
//       </Card>

//       {/* Refetch Button */}
//       <div className="flex justify-center">
//         <Button onClick={fetchInsights}>🔄 Refresh Insights</Button>
//       </div>
//     </div>
//   )
// }
import { useState, useEffect } from "react"
import { Card, CardHeader, CardTitle, CardContent } from "./components/ui/card"
import { Button } from "./components/ui/button"
import { Loader2, Thermometer, MapPin, AlertTriangle, ShieldCheck } from "lucide-react"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts"

// Mock cache (static fallback data for prototype)
// Mock cache (static fallback data for prototype)
const staticInsights = {
  "Thunnus albacares": {
    sstRange: "23°C – 29°C",
    regions: ["Tropical Pacific", "Indian Ocean"],
    risk: "High (Overfishing)",
    recommendation: "Strengthen tuna stock monitoring and enforce quotas.",
    chartData: [
      { temp: 22, density: 20 },
      { temp: 24, density: 45 },
      { temp: 26, density: 60 },
      { temp: 28, density: 40 },
      { temp: 30, density: 15 },
    ],
  },
  "Cyanea capillata": {
    sstRange: "2°C – 12°C",
    regions: ["North Atlantic", "Arctic waters"],
    risk: "Low",
    recommendation: "Monitor blooms; climate shifts may expand range.",
    chartData: [
      { temp: 0, density: 10 },
      { temp: 4, density: 35 },
      { temp: 8, density: 50 },
      { temp: 12, density: 25 },
      { temp: 16, density: 5 },
    ],
  },
  "Sardinella longiceps": {
    sstRange: "22°C – 28°C",
    regions: ["Arabian Sea", "Bay of Bengal"],
    risk: "Medium (Climate-sensitive)",
    recommendation: "Implement adaptive seasonal fishing bans.",
    chartData: [
      { temp: 20, density: 15 },
      { temp: 22, density: 40 },
      { temp: 24, density: 55 },
      { temp: 26, density: 50 },
      { temp: 28, density: 30 },
    ],
  },
  "Katsuwonus pelamis": {
    sstRange: "20°C – 28°C",
    regions: ["Pacific Ocean", "Indian Ocean", "Atlantic Tropics"],
    risk: "High (Climate-driven migration)",
    recommendation: "Adopt dynamic spatial fisheries management.",
    chartData: [
      { temp: 18, density: 10 },
      { temp: 22, density: 50 },
      { temp: 24, density: 65 },
      { temp: 26, density: 55 },
      { temp: 28, density: 25 },
    ],
  },
  "Engraulis encrasicolus": {
    sstRange: "15°C – 24°C",
    regions: ["Mediterranean", "Eastern Atlantic"],
    risk: "Medium",
    recommendation: "Improve monitoring of spawning grounds in warming waters.",
    chartData: [
      { temp: 14, density: 15 },
      { temp: 16, density: 35 },
      { temp: 18, density: 55 },
      { temp: 20, density: 60 },
      { temp: 22, density: 40 },
    ],
  },
  "Thunnus thynnus": {
    sstRange: "10°C – 26°C",
    regions: ["North Atlantic", "Mediterranean"],
    risk: "High (Critically Endangered)",
    recommendation: "Strict protection of spawning grounds; ban IUU fishing.",
    chartData: [
      { temp: 10, density: 20 },
      { temp: 14, density: 40 },
      { temp: 18, density: 60 },
      { temp: 22, density: 45 },
      { temp: 26, density: 20 },
    ],
  },
  "Lutjanus campechanus": {
    sstRange: "18°C – 28°C",
    regions: ["Gulf of Mexico", "Caribbean"],
    risk: "Medium (Habitat degradation)",
    recommendation: "Expand marine protected areas and restore coral reefs.",
    chartData: [
      { temp: 18, density: 20 },
      { temp: 20, density: 35 },
      { temp: 24, density: 55 },
      { temp: 26, density: 50 },
      { temp: 28, density: 30 },
    ],
  },
  "Sepia officinalis": {
    sstRange: "8°C – 22°C",
    regions: ["Eastern Atlantic", "Mediterranean"],
    risk: "Low",
    recommendation: "Promote sustainable fisheries with seasonal closures.",
    chartData: [
      { temp: 8, density: 15 },
      { temp: 12, density: 35 },
      { temp: 16, density: 50 },
      { temp: 20, density: 40 },
      { temp: 22, density: 20 },
    ],
  },
}


export default function InsightsUI() {
  const [species, setSpecies] = useState("")
  const [loading, setLoading] = useState(false)
  const [insight, setInsight] = useState(null)

  const handleFetchInsights = () => {
    if (!species) return
    setLoading(true)

    // Simulated API call with fallback
    setTimeout(() => {
      const fetched = staticInsights[species]
      setInsight(fetched || { error: "No insights available" })
      setLoading(false)
    }, 1200)
  }

  return (
    <div className="w-full max-w-5xl mx-auto p-6 space-y-6">
      {/* Species Selector */}
      <Card className="shadow-lg rounded-2xl p-4">
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Select a Species</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col md:flex-row items-center gap-4">
          <select
            value={species}
            onChange={(e) => setSpecies(e.target.value)}
            className="border rounded-lg p-2 w-full md:w-1/2"
          >
            <option value="">-- Choose Species --</option>
            {Object.keys(staticInsights).map((sp) => (
              <option key={sp} value={sp}>
                {sp}
              </option>
            ))}
          </select>
          <Button onClick={handleFetchInsights} disabled={!species || loading}>
            {loading ? <Loader2 className="animate-spin w-4 h-4 mr-2" /> : null}
            Get Insights
          </Button>
        </CardContent>
      </Card>

      {/* Insights Panel */}
      {insight && (
        <Card className="shadow-xl rounded-2xl p-4">
          <CardHeader>
            <CardTitle className="text-xl font-bold flex justify-between items-center">
              AI-Driven Insights for <span className="text-blue-600">{species}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {insight.error ? (
              <p className="text-red-500">{insight.error}</p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-3 bg-blue-50 rounded-xl">
                  <Thermometer className="inline mr-2 text-blue-600" />
                  <strong>Temperature Range:</strong>
                  <p>{insight.sstRange}</p>
                </div>

                <div className="p-3 bg-green-50 rounded-xl">
                  <MapPin className="inline mr-2 text-green-600" />
                  <strong>Regions:</strong>
                  <p>{insight.regions.join(", ")}</p>
                </div>

                <div className="p-3 bg-red-50 rounded-xl">
                  <AlertTriangle className="inline mr-2 text-red-600" />
                  <strong>Risk Level:</strong>
                  <p>{insight.risk}</p>
                </div>

                <div className="p-3 bg-yellow-50 rounded-xl">
                  <ShieldCheck className="inline mr-2 text-yellow-600" />
                  <strong>Recommendation:</strong>
                  <p>{insight.recommendation}</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Chart Section */}
      {insight && insight.chartData && (
        <Card className="shadow-xl rounded-2xl p-4">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">
              Habitat Suitability vs Temperature
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={insight.chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="temp" label={{ value: "Temperature (°C)", position: "insideBottom", offset: -5 }} />
                <YAxis label={{ value: "Occurrence Density", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Line type="monotone" dataKey="density" stroke="#2563eb" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
