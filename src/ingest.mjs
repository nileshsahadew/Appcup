import fs from "fs";
import path from "path";
import dotenv from "dotenv";
dotenv.config();

import { GoogleGenerativeAI } from "@google/generative-ai";
import { ChromaClient } from "chromadb";

const KEY = process.env.GEMINI_API_KEY;
if (!KEY) {
  console.error("❌ Please set GEMINI_API_KEY in .env.local");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(KEY);
// Use simple local ChromaDB configuration
const chroma = new ChromaClient();

await chroma.reset(); // clears previous data — remove this in production
const collection = await chroma.createCollection({ name: "travel_docs" });

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

let counter = 0;
for (const doc of docs) {
  const text = cleanText(doc.content);
  const chunks = chunkText(text);

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    const embedResp = await genAI.embedContent({
      model: "models/embedding-001",
      content: { parts: [{ text: chunk }] }
    });

    const embedding = embedResp.embedding.values;

    await collection.add({
      ids: [`doc${counter}`],
      embeddings: [embedding],
      documents: [chunk],
      metadatas: [{ title: doc.title }]
    });

    console.log(`✅ Stored chunk doc${counter} (${doc.title})`);
    counter++;
  }
}

console.log(`🎯 Ingested ${counter} chunks into ChromaDB`);
process.exit(0);
