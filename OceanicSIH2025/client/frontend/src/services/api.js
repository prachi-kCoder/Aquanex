
const API_URL = "http://localhost:8000"; // change to deployed backend later

// 🔹 Generic: get all stored records
async function getData() {
  const res = await fetch(`${API_URL}/data/`);
  if (!res.ok) throw new Error("Failed to fetch data");
  return await res.json();
}

// 🔹 Generic: ingest any provider
async function ingest(provider, payload) {
  const res = await fetch(`${API_URL}/ingest/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, payload }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Ingest failed");
  }
  return await res.json();
}

// 🔹 NOAA: dedicated function (singular product)
async function getNOAARecord({ station, product, begin_date, end_date }) {
  if (!station || !product) {
    throw new Error("Station and product are required");
  }

  const res = await fetch(`${API_URL}/providers/noaa`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      station,
      product,
      begin_date,
      end_date,
    }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to fetch NOAA data");
  }

  return await res.json();
}
async function getNOAARecords(station, products, begin_date, end_date) {
  const calls = products.map(product =>
    getNOAARecord({ station, product, begin_date, end_date })
  );
  const results = await Promise.all(calls);
  return results.flat(); // flatten array of arrays
}
export async function ingestNOAA(station, product, begin_date, end_date) {
  const res = await fetch(`${API_URL}/ingest/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: "noaa",
      payload: {
        station,
        product,
        begin_date,
        end_date,
      },
    }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "NOAA ingestion failed");
  }

  const data = await res.json();
  return data.records || [];
}
// 🔹 WoRMS: dedicated function
async function ingestWorms({ scientificname, AphiaID }) {
  if (!scientificname && !AphiaID) {
    throw new Error("Either scientificname or AphiaID is required");
  }

  const payload = scientificname
    ? { endpoint: "AphiaRecordsByName", params: { scientificname } }
    : { endpoint: "AphiaRecordByAphiaID", params: { AphiaID } };

  return await ingest("worms", payload);
}
async function ingestObis({ scientificname, size = 20 }) {
  if (!scientificname) {
    throw new Error("Scientific name is required for OBIS ingestion");
  }

  const payload = {
    endpoint: "occurrence",
    params: { scientificname, size }
  };

  return await ingest("obis", payload);
}

 async function ingestOpenMeteo(payload) {
  const reqPayload = {
    ...payload,
    limit_hours: payload.limit_hours || 6, // default 48 hours
  };

  const res = await fetch("http://localhost:8000/ingest/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider: "open-meteo", payload: reqPayload }),
  });
  if (!res.ok) throw new Error("Failed to fetch Open-Meteo data");
  return res.json();
}

export default { getData, ingest, ingestOpenMeteo ,ingestNOAA, getNOAARecord ,getNOAARecords, ingestWorms, ingestObis };


