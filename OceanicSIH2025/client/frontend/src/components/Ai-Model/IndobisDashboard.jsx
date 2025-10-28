import React, { useState, useEffect, useMemo } from "react";
import axios from "axios";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";


const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

/* simple color ramp - low->high */
function getColorForProb(p) {
  if (p === null || p === undefined || !isFinite(p)) return "#ffffff00";
  // interpolate roughly: blue -> cyan -> yellow -> orange -> red
  if (p <= 0.1) return "#f7fbff";
  if (p <= 0.25) return "#c6dbef";
  if (p <= 0.45) return "#74a9cf";
  if (p <= 0.6) return "#fdae6b";
  if (p <= 0.8) return "#f16913";
  return "#a50f15";
}

/* map zoom-to-bbox helper (custom hook) */
function FitBounds({ bbox }) {
  const map = useMap();
  useEffect(() => {
    if (!bbox || bbox.length !== 4) return;
    const [[minLon, minLat, maxLon, maxLat]] = [[bbox]];
    try {
      map.fitBounds(
        [
          [minLat, minLon],
          [maxLat, maxLon],
        ],
        { padding: [40, 40] }
      );
    } catch (e) {}
  }, [bbox, map]);
  return null;
}

function safeNum(x, fallback = 0) {
  if (x === null || x === undefined) return fallback;
  if (typeof x !== "number" || !isFinite(x)) return fallback;
  return x;
}


export default function IndobisDashboard() {
  const [speciesInput, setSpeciesInput] = useState("");
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [gridResult, setGridResult] = useState(null);

    async function fetchStatus() {
    try {
        // First get API status
        const resp = await axios.get(`${API_BASE}/status`);
        let ms = resp.data.models || [];

        // Then merge with disk-based models
        const listResp = await fetch(`${API_BASE}/list_models`);
        const listData = await listResp.json();
        if (listData.models) {
        // Merge without duplicates
        const merged = [...ms];
        listData.models.forEach((lm) => {
            if (!merged.find((m) => m.scientific_name === lm.scientific_name)) {
            merged.push(lm);
            }
        });
        ms = merged;
        }

        // Sort by trained_at desc
        ms.sort((a, b) => {
        const ta = a.trained_at || "";
        const tb = b.trained_at || "";
        return tb.localeCompare(ta);
        });

        setModels(ms);

        // Auto-select latest trained if none selected
        if (ms.length > 0 && !selectedModel) {
        setSelectedModel(ms[0].scientific_name);
        }
        } catch (err) {
        console.error("fetchStatus", err);
    }
    }

    useEffect(() => {
    fetch("http://localhost:8000/list_models") // replace with your backend URL
        .then((res) => res.json())
        .then((data) => {
        if (data.models) setModels(data.models);
        });
    }, []);
   


    const handleTrain = async () => {
        if (!speciesInput.trim()) {
            alert("Please enter scientific name.");
            return;
        }
        setLoading(true);
        setStatusMsg(`Training ${speciesInput} ...`);
        try {
            await axios.post(`${API_BASE}/train_batch`, {
            species: [speciesInput.trim()],
            max_records: 3000,
            test_size: 0.2,
            });
            setStatusMsg(`Trained: ${speciesInput}`);
            setSpeciesInput("");

            // Refresh models + auto-select latest
            await fetchStatus();
            setSelectedModel(speciesInput.trim());
        } catch (err) {
            console.error(err);
            alert("Training failed: " + (err?.response?.data?.detail || err.message));
            setStatusMsg("Training failed.");
        } finally {
            setLoading(false);
        }
    };


  // Remove model
  const handleRemoveModel = async (name) => {
    if (!window.confirm(`Remove model ${name}?`)) return;
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/remove_model`, { scientific_name: name });
      setStatusMsg(`Removed model ${name}`);
      await fetchStatus();
      if (selectedModel === name) setSelectedModel("");
    } catch (err) {
      console.error(err);
      alert("Remove failed: " + (err?.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Point prediction
  const handlePredictPoint = async () => {
    if (!selectedModel) {
      alert("Select a trained model first.");
      return;
    }
    const lat = prompt("Latitude (decimal):", "10.0");
    const lon = prompt("Longitude (decimal):", "72.0");
    if (!lat || !lon) return;
    setLoading(true);
    setStatusMsg("Running point prediction...");
    try {
      const resp = await axios.post(`${API_BASE}/predict`, {
        scientific_name: selectedModel,
        latitude: parseFloat(lat),
        longitude: parseFloat(lon),
      });
      setPrediction(resp.data);
      setGridResult(null);
      setStatusMsg("Point prediction done.");
    } catch (err) {
      console.error(err);
      alert("Prediction failed: " + (err?.response?.data?.detail || err.message));
      setStatusMsg("Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  // Grid prediction
  const handlePredictGrid = async () => {
    if (!selectedModel) {
      alert("Select a trained model first.");
      return;
    }
    // Optionally we could show a modal to capture bbox & resolution. For simplicity use default and confirm:
    const ok = window.confirm(
      "Run grid prediction for selected model over default bbox (60,-20,100,20) at 0.5° resolution? (May take some seconds)"
    );
    if (!ok) return;
    setLoading(true);
    setStatusMsg("Running grid prediction...");
    try {
      const resp = await axios.post(`${API_BASE}/predict_grid`, {
        scientific_name: selectedModel,
        bbox: [60, -20, 100, 20],
        grid_resolution: 0.5,
      });
      setGridResult(resp.data);
      setPrediction(null);
      setStatusMsg("Grid prediction finished.");
    } catch (err) {
      console.error(err);
      alert("Grid prediction failed: " + (err?.response?.data?.detail || err.message));
      setStatusMsg("Grid prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  // Download hotspots as CSV
  const downloadHotspotsCSV = () => {
    if (!gridResult || !gridResult.hotspot_centroids) return;
    const rows = gridResult.hotspot_centroids;
    if (!rows.length) {
      alert("No hotspots to download.");
      return;
    }
    const csv = [
      ["lat", "lon", "prob"],
      ...rows.map((r) => [r.lat.toFixed(6), r.lon.toFixed(6), r.prob.toFixed(4)]),
    ]
      .map((r) => r.join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedModel.replace(/\s+/g, "_")}_hotspots.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Memoized style function for GeoJSON features
  const geoStyle = (feature) => {
    const p = safeNum(feature.properties?.prob, null);
    return {
        fillColor: p == null ? "#ffffff00" : getColorForProb(p),
        fillOpacity: p != null ? Math.min(0.95, 0.35 + p * 0.6) : 0,
        weight: 0.25,
        color: p != null && p > 0.7 ? "#800026" : "#666",
    };
    };


  // onEachFeature for tooltip/popups
  const onEachFeature = (feature, layer) => {
    const p = feature.properties?.prob;
    const label = p != null ? `${(p * 100).toFixed(1)}%` : "N/A";
    layer.bindTooltip(`Prob: ${label}`, { sticky: true });
    layer.on({
      click: () => {
        layer.openTooltip();
      },
    });
  };

  // legend bands from bin_edges if available
  const legendItems = useMemo(() => {
    if (!gridResult?.prob_breaks) {
      // default bands
      return [
        { label: "0-10%", color: getColorForProb(0.05) },
        { label: "10-25%", color: getColorForProb(0.18) },
        { label: "25-45%", color: getColorForProb(0.35) },
        { label: "45-65%", color: getColorForProb(0.55) },
        { label: "65-100%", color: getColorForProb(0.85) },
      ];
    }
    const edges = gridResult.prob_breaks.bin_edges; // length 6 for 5 bins
    const items = [];
    for (let i = 0; i < edges.length - 1; i++) {
      const lo = edges[i], hi = edges[i + 1];
      const mid = (lo + hi) / 2.0;
      items.push({
        label: `${Math.round(lo * 100)} - ${Math.round(hi * 100)}%`,
        color: getColorForProb(mid),
      });
    }
    return items;
  }, [gridResult]);

  return (
    <div className="max-w-6xl mx-auto p-4 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">🌊 IndOBIS SDM — Dashboard</h1>
          <p className="text-sm text-gray-600">Train, manage and visualize species models</p>
        </div>
        <div>
          <div className="text-sm text-gray-700">Status: {statusMsg || "idle"}</div>
        </div>
      </header>

      {/* Training */}
      <section className="bg-white p-4 rounded shadow space-y-3">
        <h2 className="font-semibold">1. Train a species model</h2>
        <div className="flex gap-2 items-center">
          <input
            value={speciesInput}
            onChange={(e) => setSpeciesInput(e.target.value)}
            placeholder="Thunnus albacares"
            className="border px-3 py-2 rounded w-80"
          />
          <button onClick={handleTrain} className="px-4 py-2 bg-blue-600 text-white rounded">Train</button>
          <button onClick={fetchStatus} className="px-3 py-2 bg-gray-200 rounded">Refresh models</button>
        </div>
        <div className="text-xs text-gray-500">
          Tip: train models one at a time. Models are cached on the server and will appear in the list below.
        </div>
      </section>

      {/* Manage models */}
      <section className="bg-white p-4 rounded shadow space-y-3">
        <h2 className="font-semibold">2. Manage & choose a trained model</h2>
        {models.length === 0 ? (
          <div className="text-gray-600">No trained models available.</div>
        ) : (
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div className="flex items-center gap-3">
              <label className="text-sm">Pick model</label>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="p-2 border rounded">
                <option value="">-- select model --</option>
                {models.map((m) => (
                  <option key={m.scientific_name} value={m.scientific_name}>
                    {m.scientific_name} {m.auc_test ? `(AUC ${m.auc_test.toFixed(2)})` : ""}
                  </option>
                ))}
              </select>
              {models.length > 0 && (
                <div className="text-sm text-gray-500">Latest: <b>{models[0].scientific_name}</b></div>
              )}
            </div>

            <div className="flex gap-2">
              <button onClick={() => selectedModel && handlePredictPoint()} disabled={!selectedModel} className="px-3 py-2 rounded bg-green-600 text-white">Point predict</button>
              <button onClick={() => selectedModel && handlePredictGrid()} disabled={!selectedModel} className="px-3 py-2 rounded bg-purple-600 text-white">Grid predict</button>
              {selectedModel && <button onClick={() => handleRemoveModel(selectedModel)} className="px-3 py-2 rounded bg-red-600 text-white">Remove selected</button>}
            </div>
          </div>
        )}

        {/* Model cards */}
        <div className="grid md:grid-cols-2 gap-3 mt-3">
          {models.map((m) => (
            <div key={m.scientific_name} className="p-3 border rounded flex justify-between items-start">
              <div>
                <div className="font-semibold">{m.scientific_name}</div>
                <div className="text-sm text-gray-600">AUC: {m.auc_test ? m.auc_test.toFixed(2) : "N/A"} • {m.n_presence} presences</div>
                <div className="text-xs mt-1 text-gray-500">Trained: {m.trained_at || "unknown"}</div>
              </div>
              <div className="flex flex-col gap-2">
                <button onClick={() => setSelectedModel(m.scientific_name)} className="px-2 py-1 bg-blue-600 text-white rounded text-sm">Use</button>
                <button onClick={() => handleRemoveModel(m.scientific_name)} className="px-2 py-1 bg-red-600 text-white rounded text-sm">Delete</button>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Results & map */}
      <section className="bg-white p-4 rounded shadow space-y-4">
        <h2 className="font-semibold">3. Results & Visualization</h2>

        {/* Point result */}
        {prediction && (
          <div className="p-3 border rounded">
            <h3 className="font-medium">Point prediction</h3>
            <div className="mt-2">
              <div><b>{prediction.scientific_name}</b></div>
              <div>Probability: {(safeNum(prediction.probability) * 100).toFixed(1)}% — <span className="capitalize">{prediction.interpretation}</span></div>
              <div className="text-xs text-gray-500">Query: {prediction.query.latitude}, {prediction.query.longitude}</div>
            </div>
            <div className="mt-3" style={{ height: 250 }}>
              <MapContainer center={[prediction.query.latitude, prediction.query.longitude]} zoom={4} style={{ height: "100%" }}>
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                <Marker position={[prediction.query.latitude, prediction.query.longitude]}>
                  <Popup>{prediction.scientific_name} — {(prediction.probability * 100).toFixed(1)}%</Popup>
                </Marker>
              </MapContainer>
            </div>
          </div>
        )}

        {/* Grid result */}
        {gridResult && (
          <div className="p-3 border rounded">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-medium">Grid prediction — {gridResult.scientific_name}</h3>
                <div className="text-sm text-gray-600">
                    Avg prob: {(safeNum(gridResult.summary.average_probability) * 100).toFixed(1)}% •
                    Hotspots: {safeNum(gridResult.summary.hotspots_cells)} •
                    Cells: {safeNum(gridResult.summary.total_cells)}
                </div>
            </div>
            <div className="flex gap-2">
                <button onClick={downloadHotspotsCSV} className="px-3 py-1 bg-indigo-600 text-white rounded text-sm">Download hotspots</button>
              </div>
            </div>

            <div className="mt-3 md:flex md:gap-4">
              <div className="md:w-2/3" style={{ minHeight: 400 }}>
                <MapContainer center={[10, 80]} zoom={3} style={{ height: 400 }}>
                  <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  <GeoJSON data={gridResult.geojson} style={geoStyle} onEachFeature={onEachFeature} />
                  {/* hotspot markers */}
                  {gridResult.hotspot_centroids && gridResult.hotspot_centroids.map((h, idx) => (
                    <Marker key={idx} position={[h.lat, h.lon]}>
                      <Popup>Hotspot: {(h.prob * 100).toFixed(1)}%</Popup>
                    </Marker>
                  ))}
                </MapContainer>
              </div>

              <aside className="md:w-1/3 mt-4 md:mt-0">
                <div className="p-2 border rounded">
                  <h4 className="font-semibold">Legend</h4>
                  <div className="mt-2 space-y-1">
                    {legendItems.map((li, i) => (
                      <div key={i} className="flex items-center gap-2 text-sm">
                        <div style={{ width: 28, height: 12, background: li.color, borderRadius: 2 }} />
                        <div>{li.label}</div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-3">
                    <h5 className="font-medium text-sm">Prob bins</h5>
                    {gridResult.prob_breaks && (
                      <div className="text-xs text-gray-600 mt-1">
                        {gridResult.prob_breaks.bin_edges.map((e, i) => (
                          <div key={i}>{(e * 100).toFixed(0)}{i < gridResult.prob_breaks.bin_edges.length - 1 ? " - " : ""}</div>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="mt-3">
                    <h5 className="font-medium text-sm">Interpretation</h5>
                    <div className="text-xs text-gray-600">
                      Prob &lt; {gridResult.prob_breaks?.coldspot_threshold ? (gridResult.prob_breaks.coldspot_threshold * 100).toFixed(0) : 30}% → unlikely<br />
                      Between thresholds → possible<br />
                      &gt; {gridResult.prob_breaks?.hotspot_threshold ? (gridResult.prob_breaks.hotspot_threshold * 100).toFixed(0) : 70}% → likely (hotspot)
                    </div>
                  </div>
                </div>
              </aside>
            </div>
          </div>
        )}
      </section>

      {/* Loading overlay */}
      {loading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="bg-white p-6 rounded shadow">
            <div className="animate-spin h-8 w-8 border-4 border-blue-600 border-t-transparent rounded-full mx-auto" />
            <div className="mt-3 text-sm">Processing — please wait...</div>
          </div>
        </div>
      )}
    </div>
  );
}
