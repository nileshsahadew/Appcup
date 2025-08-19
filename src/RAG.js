import dotenv from "dotenv";
dotenv.config();

import { GoogleGenerativeAI } from "@google/generative-ai";
import { Qdrant } from "qdrant";

const KEY = process.env.GEMINI_API_KEY;

if (!KEY) {
  console.error("❌ Please set GEMINI_API_KEY in .env");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" });

// Initialize Qdrant client
const QDRANT_URL = process.env.QDRANT_URL || "http://localhost:6333/";
const client = new Qdrant(QDRANT_URL);

// Use the same collection name as other files
const COLLECTION_NAME = "mauritius_knowledge";

// Ingest helpers removed — this file only performs queries now.

/**
 * Queries the Qdrant collection with a given text prompt.
 * @param {string} queryText The text to search for.
 */
async function queryCollection(queryText) {
  console.log(`\n🔍 Searching for: "${queryText}"`);
  console.log("-".repeat(50));

  try {
    // Generate an embedding for the query text
    const embedResp = await embeddingModel.embedContent(queryText);
    const queryVector = embedResp.embedding.values;

    // Prefer the options signature to request payloads explicitly
    let normalized;
    try {
      const res = await client.search_collection(COLLECTION_NAME, {
        vector: queryVector,
        limit: 5,
        with_payload: true,
        with_vector: false,
      });
      normalized = res?.response ?? res; // some clients wrap under .response
    } catch (_e) {
      // Fallback to legacy positional signature
      const { err, response } = await client.search_collection(
        COLLECTION_NAME,
        queryVector,
        5
      );
      if (err) {
        throw new Error(typeof err === "string" ? err : JSON.stringify(err));
      }
      normalized = response;
    }

    const hits = Array.isArray(normalized?.result) ? normalized.result : (Array.isArray(normalized) ? normalized : []);
    console.log(`✅ Found ${hits.length} results.\n`);
    
    if (hits.length === 0) {
      console.log("❌ No results found for this query.");
      return;
    }
    
    hits.forEach((result, index) => {
      console.log(`\n${"🎯".repeat(index + 1)} RESULT ${index + 1} ${"🎯".repeat(index + 1)}`);
      console.log(`📊 Relevance Score: ${(result.score * 100).toFixed(1)}%`);
      const payload =
        result?.payload ??
        result?.payloads ??
        result?.point?.payload ??
        result?.data?.payload ??
        result?.record?.payload ??
        {};
      if (payload?.title) console.log(`🧭 Title: ${payload.title}`);
      console.log(`🏷️ Type: ${payload?.type || 'Unknown'}`);
      console.log(`📁 Source: ${payload?.source_file || 'Unknown'}`);
      if (payload?.region) {
        console.log(`🗺️ Region: ${payload.region}`);
      }
      console.log(`\n📄 ATTRACTION DETAILS:`);
      console.log("─".repeat(40));
      const details = payload?.content ?? payload?.document ?? payload?.text ?? payload?.body;
      console.log(details || JSON.stringify(payload, null, 2) || 'No content available');
      console.log("─".repeat(40));
    });
  } catch (error) {
    console.error("❌ Error during query:", error?.message || error);
  }
}

async function ensureCollectionExists() {
  try {
    await client.get_collection(COLLECTION_NAME);
    return true;
  } catch (error) {
    console.error(`❌ Collection "${COLLECTION_NAME}" not found at ${QDRANT_URL}.`);
    console.error("   Run ingestion first: npm run ingest-qdrant");
    return false;
  }
}

/**
 * Main function to ingest data and perform queries.
 */
async function main() {
  try {
    const argQuery = process.argv.slice(2).join(" ").trim();

    console.log("🔄 Ready to query Qdrant...");
    console.log("🌐 URL:", QDRANT_URL);
    console.log("📚 Collection:", COLLECTION_NAME);

    const exists = await ensureCollectionExists();
    if (!exists) {
      process.exit(1);
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
  } catch (error) {
    console.error("❌ Error:", error);
    process.exit(1);
  }
}

main();
