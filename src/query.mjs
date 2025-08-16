import dotenv from "dotenv";
dotenv.config();
import { ChromaClient } from "chromadb";
import { GoogleGenerativeAI } from "@google/generative-ai";

const KEY = process.env.GEMINI_API_KEY;
if (!KEY) {
  console.error("❌ Please set GEMINI_API_KEY in .env.local");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(KEY);
const chroma = new ChromaClient();
const collection = await chroma.getCollection({ name: "travel_docs" });

async function embedText(text) {
  const embedResp = await genAI.embedContent({
    model: "models/embedding-001",
    content: { parts: [{ text }] }
  });
  return embedResp.embedding.values;
}

async function search(query, k = 3) {
  const qEmbedding = await embedText(query);

  const results = await collection.query({
    queryEmbeddings: [qEmbedding],
    nResults: k
  });

  console.log(`\n🔍 Query: ${query}`);
  results.documents[0].forEach((doc, i) => {
    console.log(`\nResult ${i + 1}:`);
    console.log("Text:", doc);
    console.log("Meta:", results.metadatas[0][i]);
  });
}

await search("Best time to visit Mauritius");
process.exit(0);
