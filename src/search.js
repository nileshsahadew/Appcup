// query-qdrant.js
import fs from "fs";
import path from "path";
import dotenv from "dotenv";
dotenv.config();

import { GoogleGenerativeAI } from "@google/generative-ai";
import { QdrantClient } from "@qdrant/js-client-rest";

const KEY = process.env.GEMINI_API_KEY;
if (!KEY) {
  console.error("❌ Please set GEMINI_API_KEY in .env");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" });

const QDRANT_URL = process.env.QDRANT_URL || "http://localhost:6333";
const COLLECTION_NAME = process.env.QDRANT_COLLECTION_NAME || "mauritius_knowledge";

const client = new QdrantClient({ url: QDRANT_URL });

/**
 * Query Qdrant with embeddings
 */
async function queryCollection(queryText) {
  console.log(`\n🔍 Searching for: "${queryText}"`);
  console.log("-".repeat(50));

  try {
    // 1) Generate embeddings
    const embedResp = await embeddingModel.embedContent(queryText);
    const queryVector = embedResp.embedding.values;

    // 2) Vector search
    const results = await client.search(COLLECTION_NAME, {
      vector: queryVector,
      limit: 5,
      with_payload: true,
    });

    if (!results || results.length === 0) {
      console.log("❌ No dense results. Trying payload text filter...\n");
      const filtered = await textFilterSearch(queryText, 10);
      if (!filtered || filtered.length === 0) {
        console.log("❌ No results from text filter. Falling back to local embeddings...\n");
        await queryLocalEmbeddings(queryText);
      }
      return;
    }

    console.log(`✅ Found ${results.length} vector results.\n`);
    results.forEach((hit, i) => {
      console.log(`\n${"🎯".repeat(i + 1)} RESULT ${i + 1} ${"🎯".repeat(i + 1)}`);
      console.log(`📊 Score: ${(hit.score * 100).toFixed(1)}%`);
      const payload = hit.payload || {};
      if (payload.title) console.log(`🧭 Title: ${payload.title}`);
      console.log(`🏷️ Type: ${payload?.type || "Unknown"}`);
      console.log(`📁 Source: ${payload?.source_file || "Unknown"}`);
      if (payload?.region) console.log(`🗺️ Region: ${payload.region}`);
      console.log("\n📄 ATTRACTION DETAILS:");
      console.log("─".repeat(40));
      console.log(payload?.content || JSON.stringify(payload, null, 2));
      console.log("─".repeat(40));
    });
  } catch (err) {
    console.error("❌ Error during query:", err.message || err);
  }
}

/**
 * Ensure collection exists
 */
async function ensureCollectionExists() {
  try {
    await client.getCollection(COLLECTION_NAME);
    return true;
  } catch (err) {
    console.error(`❌ Collection "${COLLECTION_NAME}" not found at ${QDRANT_URL}.`);
    console.error("   Run ingestion first: npm run ingest-qdrant");
    return false;
  }
}

/**
 * Get total number of points
 */
async function getPointCount() {
  try {
    const { count } = await client.count(COLLECTION_NAME, { exact: true });
    return Number(count) || 0;
  } catch (err) {
    console.log("ℹ️ Could not retrieve point count:", err.message || err);
    return 0;
  }
}

/**
 * Qdrant REST request helper (for text filter)
 */
async function qdrantRequest(method, pathSuffix, body) {
  const url = new URL(pathSuffix, QDRANT_URL).toString();
  const res = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`Qdrant ${method} ${url} failed: ${res.status} ${txt}`);
  }
  return res.json().catch(() => undefined);
}

/**
 * Ensure text index on a field
 */
async function ensureTextIndex(fieldName = "content") {
  try {
    await qdrantRequest("PUT", `/collections/${COLLECTION_NAME}/index`, {
      field_name: fieldName,
      field_schema: { type: "text", tokenizer: "word", lowercase: true },
    });
    console.log(`🔠 Text index ensured on field '${fieldName}'.`);
  } catch (e) {
    console.log(`ℹ️ Text index ensure on '${fieldName}': ${e.message}`);
  }
}

/**
 * Text filter search fallback
 */
async function textFilterSearch(queryText, limit = 10) {
  try {
    await ensureTextIndex("content");
    const resp = await qdrantRequest(
      "POST",
      `/collections/${COLLECTION_NAME}/points/scroll`,
      {
        filter: {
          must: [{ key: "content", match: { text: queryText } }],
        },
        limit,
        with_payload: true,
      }
    );

    const points = resp?.result?.points || [];
    if (!points.length) return [];

    console.log(`✅ Found ${points.length} results via text filter.\n`);
    points.forEach((p, i) => {
      const payload = p.payload || {};
      console.log(`\n${"🧾".repeat(i + 1)} FILTER RESULT ${i + 1} ${"🧾".repeat(i + 1)}`);
      if (payload?.title) console.log(`🧭 Title: ${payload.title}`);
      if (payload?.type) console.log(`🏷️ Type: ${payload.type}`);
      if (payload?.source_file) console.log(`📁 Source: ${payload.source_file}`);
      if (payload?.region) console.log(`🗺️ Region: ${payload.region}`);
      console.log("─".repeat(40));
      console.log(payload?.content || JSON.stringify(payload, null, 2));
      console.log("─".repeat(40));
    });

    return points;
  } catch (err) {
    console.log("❌ Text filter search error:", err.message || err);
    return [];
  }
}

/**
 * Local embeddings fallback
 */
function loadLocalEmbeddings() {
  try {
    const embeddingsDir = path.join(process.cwd(), "embeddings");
    const allEmbeddingsPath = path.join(embeddingsDir, "all_embeddings.json");
    if (!fs.existsSync(allEmbeddingsPath)) return null;
    const data = JSON.parse(fs.readFileSync(allEmbeddingsPath, "utf-8"));
    return Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

function cosineSimilarity(vecA, vecB) {
  const dot = vecA.reduce((sum, a, i) => sum + a * (vecB[i] ?? 0), 0);
  const magA = Math.sqrt(vecA.reduce((s, a) => s + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((s, b) => s + b * b, 0));
  if (!magA || !magB) return 0;
  return dot / (magA * magB);
}

async function queryLocalEmbeddings(queryText, topK = 5) {
  const local = loadLocalEmbeddings();
  if (!local) {
    console.log("ℹ️ No local embeddings found at embeddings/all_embeddings.json. Skipping local fallback.");
    return;
  }

  const embedResp = await embeddingModel.embedContent(queryText);
  const queryVector = embedResp.embedding.values;

  const ranked = local
    .map((e, i) => ({ i, score: cosineSimilarity(queryVector, e.embedding), e }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  if (!ranked.length) {
    console.log("❌ No results found in local embeddings.");
    return;
  }

  console.log(`✅ Found ${ranked.length} local results.\n`);
  ranked.forEach((r, idx) => {
    console.log(`\n${"📄".repeat(idx + 1)} LOCAL RESULT ${idx + 1} ${"📄".repeat(idx + 1)}`);
    console.log(`📊 Similarity: ${(r.score * 100).toFixed(1)}%`);
    console.log("─".repeat(40));
    console.log(r.e.document);
    console.log("─".repeat(40));
    if (r.e.metadata) console.log("📝 Metadata:", JSON.stringify(r.e.metadata));
  });
}

/**
 * Main runner
 */
async function main() {
  const argQuery = process.argv.slice(2).join(" ").trim();

  console.log("🔄 Ready to query Qdrant...");
  console.log("🌐 URL:", QDRANT_URL);
  console.log("📚 Collection:", COLLECTION_NAME);

  const exists = await ensureCollectionExists();
  if (!exists) process.exit(1);

  const count = await getPointCount();
  console.log(`📦 Points in collection: ${count}`);
  if (count === 0) {
    console.log("❗ Collection is empty. Run: npm run ingest-qdrant, then retry.");
  }

  if (argQuery) {
    await queryCollection(argQuery);
  } else {
    console.log("\n" + "=".repeat(60));
    console.log("🔍 TESTING QUERIES");
    console.log("=".repeat(60));
    await queryCollection("best beaches in Mauritius");
    await queryCollection("water sports and snorkeling");
    await queryCollection("Black River Gorges National Park");
    await queryCollection("historical places and museums");
    await queryCollection("family friendly attractions");
  }

  process.exit(0);
}

main();
