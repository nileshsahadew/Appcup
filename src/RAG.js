import fs from "fs";
import path from "path";
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
const client = new Qdrant("http://localhost:6333/");

// Use the same collection name as other files
const COLLECTION_NAME = "mauritius_knowledge";
const VECTOR_SIZE = 768; // Gemini embedding dimension

/**
 * Utility function to clean text
 */
function cleanText(text) {
  return text.replace(/\s+/g, " ").trim();
}

/**
 * Utility function to chunk text
 */
function chunkText(text, chunkSize = 300, overlap = 50) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, Math.min(start + chunkSize, text.length)));
    start += chunkSize - overlap;
  }
  return chunks;
}

/**
 * Build attraction text from attraction object
 */
function buildAttractionText(attraction) {
  const parts = [];
  
  // Basic information
  if (attraction.name) parts.push(`🏖️ **${attraction.name}**`);
  if (attraction.attraction_type) parts.push(`Type: ${attraction.attraction_type}`);
  if (attraction.description) parts.push(`Description: ${attraction.description}`);
  
  // Tags
  if (Array.isArray(attraction.tags) && attraction.tags.length) {
    parts.push(`Tags: ${attraction.tags.join(", ")}`);
  }
  
  // Location details
  const loc = attraction.location || {};
  const region = loc.region || (loc.Region || undefined);
  const addr = loc.address || undefined;
  if (addr || region) {
    parts.push(`📍 Location: ${[addr, region].filter(Boolean).join(" | ")}`);
  }
  
  // Prerequisites and access
  const prereq = attraction.prerequisites || {};
  if (prereq.booking_required) parts.push(`📅 Booking Required: Yes`);
  if (prereq.fee_required) parts.push(`💰 Entry Fee: Required`);
  if (prereq.notes) parts.push(`ℹ️ Access Notes: ${prereq.notes}`);
  
  // Reviews and ratings
  const reviews = Array.isArray(attraction.reviews) ? attraction.reviews : [];
  if (reviews.length) {
    const r0 = reviews[0];
    const summary = r0?.summary ? `\n  Summary: ${r0.summary}` : "";
    parts.push(`⭐ Reviews: ${r0?.source || ""} ${r0?.rating || ""} (${r0?.reviewCount ?? "n/a"} reviews)${summary}`);
  }
  
  // Additional information
  const info = attraction.additional_info || {};
  if (info?.notes) parts.push(`📝 Notes: ${info.notes}`);
  
  // Safety advisories
  if (info?.advisories) {
    const adv = info.advisories;
    if (adv.safety) parts.push(`⚠️ Safety: ${adv.safety}`);
    if (adv.travel) parts.push(`🚗 Travel: ${adv.travel}`);
  }
  
  // Popularity
  const pop = attraction.audience_popularity || {};
  if (pop.locals || pop.tourists) {
    const popularity = [];
    if (pop.locals) popularity.push(`Locals: ${pop.locals}`);
    if (pop.tourists) popularity.push(`Tourists: ${pop.tourists}`);
    parts.push(`👥 Popularity: ${popularity.join(", ")}`);
  }
  
  return parts.join("\n");
}

/**
 * Creates or recreates the Qdrant collection.
 */
async function createCollection() {
  try {
    // Check if collection exists and delete it to ensure a clean start
    try {
      await client.get_collection(COLLECTION_NAME);
      console.log(
        `🔄 Collection ${COLLECTION_NAME} already exists, deleting it...`
      );
      await client.delete_collection(COLLECTION_NAME);
      // Wait a bit for deletion to complete
      await new Promise((resolve) => setTimeout(resolve, 2000));
    } catch (error) {
      console.log(
        `📝 Collection ${COLLECTION_NAME} doesn't exist, will create it`
      );
    }

    // Create a new collection with a Cosine distance for vector search
    console.log(`🔄 Creating collection ${COLLECTION_NAME}...`);
    await client.create_collection(COLLECTION_NAME, {
      vectors: {
        size: VECTOR_SIZE,
        distance: "Cosine",
      },
    });

    console.log(`✅ Collection ${COLLECTION_NAME} created successfully`);
  } catch (error) {
    console.error("❌ Error creating collection:", error);
    throw error;
  }
}

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

    // Search the Qdrant collection for similar vectors
    const { err, response } = await client.search_collection(
      COLLECTION_NAME,
      queryVector,
      5
    );

    if (err) {
      throw new Error(typeof err === "string" ? err : JSON.stringify(err));
    }

    const hits = response?.result || [];
    console.log(`✅ Found ${hits.length} results.\n`);
    
    if (hits.length === 0) {
      console.log("❌ No results found for this query.");
      return;
    }
    
    hits.forEach((result, index) => {
      console.log(`\n${"🎯".repeat(index + 1)} RESULT ${index + 1} ${"🎯".repeat(index + 1)}`);
      console.log(`📊 Relevance Score: ${(result.score * 100).toFixed(1)}%`);
      console.log(`🏷️ Type: ${result.payload?.type || 'Unknown'}`);
      console.log(`📁 Source: ${result.payload?.source_file || 'Unknown'}`);
      if (result.payload?.region) {
        console.log(`🗺️ Region: ${result.payload.region}`);
      }
      console.log(`\n📄 ATTRACTION DETAILS:`);
      console.log("─".repeat(40));
      console.log(result.payload?.content || 'No content available');
      console.log("─".repeat(40));
    });
  } catch (error) {
    console.error("❌ Error during query:", error);
  }
}

/**
 * Main function to ingest data and perform queries.
 */
async function main() {
  try {
    console.log("🔄 Attempting to connect to Qdrant...");

    // Test connection to Qdrant
    try {
      console.log("✅ Qdrant client initialized successfully");
    } catch (error) {
      console.error("❌ Could not connect to Qdrant. Error details:");
      console.error("   Error message:", error.message);
      console.error("   Make sure Qdrant is running on localhost:6333");
      process.exit(1);
    }

    // Create collection
    await createCollection();

    let counter = 0;
    const points = [];

    // Ingest travel_docs.json (array of { title, content })
    const travelPath = path.join(process.cwd(), "Dataset", "travel_docs.json");
    if (fs.existsSync(travelPath)) {
      console.log("\n📖 Ingesting travel_docs.json...");
      const docs = JSON.parse(fs.readFileSync(travelPath, "utf-8"));
      for (const doc of docs) {
        const text = cleanText(doc.content || "");
        const chunks = chunkText(text);
        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          try {
            const embedResp = await embeddingModel.embedContent(chunk);
            const embedding = embedResp.embedding.values;
            points.push({
              id: counter,
              vector: embedding,
              payload: {
                type: "doc",
                title: doc.title,
                content: chunk,
                chunk_index: i,
                source_file: "travel_docs.json",
                document_id: `doc_${doc.title?.replace(/\s+/g, '_')}`,
              },
            });
            console.log(`✅ Processed travel_docs chunk ${counter} (${doc.title})`);
            counter++;
          } catch (error) {
            console.error(`❌ Error processing travel_docs chunk ${counter}:`, error.message);
          }
        }
      }
    } else {
      console.warn("⚠️ Skipped: Dataset/travel_docs.json not found");
    }

    // Helper to ingest attractions arrays
    async function ingestAttractionsFile(fileName, sourceLabel) {
      const filePath = path.join(process.cwd(), "Dataset", fileName);
      if (!fs.existsSync(filePath)) {
        console.warn(`⚠️ Skipped: Dataset/${fileName} not found`);
        return;
      }
      console.log(`\n📖 Ingesting ${fileName}...`);
      const raw = JSON.parse(fs.readFileSync(filePath, "utf-8"));
      const attractions = Array.isArray(raw?.attractions) ? raw.attractions : (Array.isArray(raw) ? raw : []);
      for (const attr of attractions) {
        const region = attr?.location?.region;
        const baseText = buildAttractionText(attr);
        const text = cleanText(baseText);
        const chunks = chunkText(text);
        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          try {
            const embedResp = await embeddingModel.embedContent(chunk);
            const embedding = embedResp.embedding.values;
            const safeId = (attr.id || attr.name || `${sourceLabel}_item`).toString().replace(/\s+/g, '_');
            points.push({
              id: counter,
              vector: embedding,
              payload: {
                type: "attraction",
                title: attr.name,
                content: chunk,
                chunk_index: i,
                source_file: sourceLabel,
                region: region,
                document_id: `${sourceLabel}_${safeId}`,
              },
            });
            console.log(`✅ Processed ${sourceLabel} chunk ${counter} (${attr.name})`);
            counter++;
          } catch (error) {
            console.error(`❌ Error processing ${sourceLabel} chunk ${counter}:`, error.message);
          }
        }
      }
    }

    // Ingest attractions datasets
    await ingestAttractionsFile("attractions.json", "attractions.json");
    await ingestAttractionsFile("attractions_ver2.json", "attractions_ver2.json");

    // Insert points in batches
    if (points.length > 0) {
      await client.upload_points(COLLECTION_NAME, points);
      console.log(
        `🎯 Successfully inserted ${counter} chunks into Qdrant collection: ${COLLECTION_NAME}`
      );
    } else {
      console.warn("⚠️ No points to upload.");
    }

    console.log("✅ Data ingestion complete.\n");

    // Perform queries
    console.log("\n" + "=".repeat(60));
    console.log("🔍 TESTING QUERIES");
    console.log("=".repeat(60));
    
    await queryCollection("best beaches in Mauritius");
    await queryCollection("water sports and snorkeling");
    await queryCollection("Black River Gorges National Park");
    await queryCollection("historical places and museums");
    await queryCollection("family friendly attractions");

    process.exit(0);
  } catch (error) {
    console.error("❌ Error:", error);
    process.exit(1);
  }
}

main();
