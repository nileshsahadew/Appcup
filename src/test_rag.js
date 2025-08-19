import fs from "fs";
import path from "path";
import dotenv from "dotenv";
dotenv.config();

import { GoogleGenerativeAI } from "@google/generative-ai";

const KEY = process.env.GEMINI_API_KEY;

if (!KEY) {
  console.error("❌ Please set GEMINI_API_KEY in .env.local");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" });

/**
 * Load all embeddings from the embeddings directory
 */
function loadEmbeddings() {
  const embeddingsDir = path.join(process.cwd(), "embeddings");
  const allEmbeddingsPath = path.join(embeddingsDir, "all_embeddings.json");
  
  if (!fs.existsSync(allEmbeddingsPath)) {
    console.error("❌ No embeddings found. Please run: npm run ingest-local");
    process.exit(1);
  }
  
  const embeddings = JSON.parse(fs.readFileSync(allEmbeddingsPath, "utf-8"));
  console.log(`✅ Loaded ${embeddings.length} embeddings from local storage`);
  return embeddings;
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

/**
 * Search embeddings for similar content
 */
async function searchEmbeddings(query, embeddings, topK = 5) {
  console.log(`\n🔍 Searching for: "${query}"`);
  console.log("-".repeat(50));
  
  try {
    // Generate embedding for the query
    const embedResp = await embeddingModel.embedContent(query);
    const queryVector = embedResp.embedding.values;
    
    // Calculate similarities
    const similarities = embeddings.map((embedding, index) => ({
      index,
      score: cosineSimilarity(queryVector, embedding.embedding),
      data: embedding
    }));
    
    // Sort by similarity score (descending)
    similarities.sort((a, b) => b.score - a.score);
    
    // Return top K results
    const results = similarities.slice(0, topK);
    
    console.log(`✅ Found ${results.length} results.\n`);
    
    if (results.length === 0) {
      console.log("❌ No results found for this query.");
      return;
    }
    
    results.forEach((result, index) => {
      console.log(`\n${"🎯".repeat(index + 1)} RESULT ${index + 1} ${"🎯".repeat(index + 1)}`);
      console.log(`📊 Relevance Score: ${(result.score * 100).toFixed(1)}%`);
      console.log(`📄 DOCUMENT:`);
      console.log("─".repeat(40));
      console.log(result.data.document);
      console.log("─".repeat(40));
      console.log(`📝 Metadata: ${JSON.stringify(result.data.metadata, null, 2)}`);
    });
    
  } catch (error) {
    console.error("❌ Error during search:", error);
  }
}

/**
 * Main function
 */
async function main() {
  try {
    console.log("🔄 Loading local embeddings...");
    const embeddings = loadEmbeddings();
    
    console.log("\n" + "=".repeat(60));
    console.log("🔍 TESTING QUERIES (Local Embeddings)");
    console.log("=".repeat(60));
    
    // Test queries
    await searchEmbeddings("best beaches in Mauritius", embeddings);
    await searchEmbeddings("water sports and snorkeling", embeddings);
    await searchEmbeddings("Black River Gorges National Park", embeddings);
    await searchEmbeddings("historical places and museums", embeddings);
    await searchEmbeddings("family friendly attractions", embeddings);
    
    process.exit(0);
  } catch (error) {
    console.error("❌ Error:", error);
    process.exit(1);
  }
}

main();
