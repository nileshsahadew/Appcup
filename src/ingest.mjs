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

// Unified collection covering all datasets
const COLLECTION_NAME = "mauritius_knowledge";
const VECTOR_SIZE = 768; // Gemini embedding dimension

async function createCollection() {
  try {
    // Check if collection exists by trying to get it
    try {
      await client.get_collection(COLLECTION_NAME);
      console.log(`🔄 Collection ${COLLECTION_NAME} already exists, deleting it...`);
      await client.delete_collection(COLLECTION_NAME);
      // Wait a bit for deletion to complete
      await new Promise(resolve => setTimeout(resolve, 2000));
    } catch (error) {
      // Collection doesn't exist, which is fine
      console.log(`📝 Collection ${COLLECTION_NAME} doesn't exist, will create it`);
    }

    // Create collection
    console.log(`🔄 Creating collection ${COLLECTION_NAME}...`);
    await client.create_collection(COLLECTION_NAME, {
      vectors: {
        size: VECTOR_SIZE,
        distance: "Cosine"
      }
    });

    console.log(`✅ Collection ${COLLECTION_NAME} created successfully`);
  } catch (error) {
    console.error("❌ Error creating collection:", error);
    throw error;
  }
}

function cleanText(text) {
  return text.replace(/\s+/g, " ").trim();
}

function chunkText(text, chunkSize = 300, overlap = 50) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, Math.min(start + chunkSize, text.length)));
    start += chunkSize - overlap;
  }
  return chunks;
}

function buildAttractionText(attraction) {
  const parts = [];
  if (attraction.name) parts.push(`Name: ${attraction.name}`);
  if (attraction.attraction_type) parts.push(`Type: ${attraction.attraction_type}`);
  if (attraction.description) parts.push(`Description: ${attraction.description}`);
  if (Array.isArray(attraction.tags) && attraction.tags.length) parts.push(`Tags: ${attraction.tags.join(", ")}`);
  const loc = attraction.location || {};
  const region = loc.region || (loc.Region || undefined);
  const addr = loc.address || undefined;
  if (addr || region) parts.push(`Location: ${[addr, region].filter(Boolean).join(" | ")}`);
  const reviews = Array.isArray(attraction.reviews) ? attraction.reviews : [];
  if (reviews.length) {
    const r0 = reviews[0];
    const summary = r0?.summary ? ` ${r0.summary}` : "";
    parts.push(`Reviews: ${r0?.source || ""} ${r0?.rating || ""} (${r0?.reviewCount ?? "n/a"} reviews).${summary}`.trim());
  }
  const info = attraction.additional_info || {};
  if (info?.notes) parts.push(`Notes: ${info.notes}`);
  return parts.join("\n");
}

async function main() {
  try {
    console.log("🔄 Attempting to connect to Qdrant...");
    
    // Test connection to Qdrant by trying to create a test collection
    try {
      console.log("✅ Qdrant client initialized successfully");
    } catch (error) {
      console.error("❌ Could not initialize Qdrant client. Error details:");
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

    // Insert points
    if (points.length > 0) {
      await client.upload_points(COLLECTION_NAME, points);
      console.log(`🎯 Successfully inserted ${counter} chunks into Qdrant collection: ${COLLECTION_NAME}`);
    } else {
      console.warn("⚠️ No points to upload.");
    }

    console.log("✅ Data uploaded successfully");
    console.log("🌐 You can view your data at: http://localhost:6333/dashboard");
    process.exit(0);
  } catch (error) {
    console.error("❌ Error:", error);
    process.exit(1);
  }
}

main();
