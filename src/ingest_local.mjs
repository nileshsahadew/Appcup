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

const datasetPath = path.join(process.cwd(), "Dataset", "travel_docs.json");
if (!fs.existsSync(datasetPath)) {
  console.error("❌ Dataset file not found:", datasetPath);
  process.exit(1);
}

const docs = JSON.parse(fs.readFileSync(datasetPath, "utf-8"));

// Create embeddings directory if it doesn't exist
const embeddingsDir = path.join(process.cwd(), "embeddings");
if (!fs.existsSync(embeddingsDir)) {
  fs.mkdirSync(embeddingsDir, { recursive: true });
}

let counter = 0;
const allEmbeddings = [];

for (const doc of docs) {
  const text = cleanText(doc.content);
  const chunks = chunkText(text);

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    try {
      const embedResp = await embeddingModel.embedContent(chunk);

      const embedding = embedResp.embedding.values;

      const embeddingData = {
        id: `doc${counter}`,
        embedding: embedding,
        document: chunk,
        metadata: { title: doc.title, chunkIndex: i }
      };

      allEmbeddings.push(embeddingData);
      
      // Save individual embedding
      const embeddingPath = path.join(embeddingsDir, `embedding_${counter}.json`);
      fs.writeFileSync(embeddingPath, JSON.stringify(embeddingData, null, 2));

      console.log(`✅ Stored chunk doc${counter} (${doc.title})`);
      counter++;
    } catch (error) {
      console.error(`❌ Error processing chunk ${counter}:`, error.message);
      continue;
    }
  }
}

// Save all embeddings in one file
const allEmbeddingsPath = path.join(embeddingsDir, "all_embeddings.json");
fs.writeFileSync(allEmbeddingsPath, JSON.stringify(allEmbeddings, null, 2));

console.log(`🎯 Successfully stored ${counter} embeddings locally in ${embeddingsDir}/`);
console.log(`📁 Individual files: ${embeddingsDir}/embedding_*.json`);
console.log(`📄 Combined file: ${allEmbeddingsPath}`);
process.exit(0);
