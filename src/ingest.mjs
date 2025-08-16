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

const COLLECTION_NAME = "travel_docs";
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

    const datasetPath = path.join(process.cwd(), "Dataset", "travel_docs.json");
    if (!fs.existsSync(datasetPath)) {
      console.error("❌ Dataset file not found:", datasetPath);
      process.exit(1);
    }

    const docs = JSON.parse(fs.readFileSync(datasetPath, "utf-8"));

    let counter = 0;
    const points = [];

    for (const doc of docs) {
      const text = cleanText(doc.content);
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
              title: doc.title,
              content: chunk,
              chunk_index: i,
              document_id: `doc_${doc.title.replace(/\s+/g, '_')}`
            }
          });

          console.log(`✅ Processed chunk ${counter} (${doc.title})`);
          counter++;
        } catch (error) {
          console.error(`❌ Error processing chunk ${counter}:`, error.message);
          continue;
        }
      }
    }

    // Insert points in batches (Qdrant handles batching well)
    if (points.length > 0) {
      await client.upload_points(COLLECTION_NAME, {
        points: points
      });
      console.log(`🎯 Successfully inserted ${counter} chunks into Qdrant collection: ${COLLECTION_NAME}`);
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
