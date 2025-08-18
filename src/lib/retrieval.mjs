import { Qdrant } from "qdrant";
import { GoogleGenerativeAI } from "@google/generative-ai";

const COLLECTION_NAME = "mauritius_knowledge";
const QDRANT_URL = "http://localhost:6333/";

function assertEnv() {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not set in environment");
  }
}

function getEmbeddingModel() {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  return genAI.getGenerativeModel({ model: "models/embedding-001" });
}

export async function retrieveTopK(queryText, topK = 5) {
  assertEnv();
  const embeddingModel = getEmbeddingModel();
  const client = new Qdrant(QDRANT_URL);

  const embedResp = await embeddingModel.embedContent(queryText);
  const queryVector = embedResp.embedding.values;

  const { err, response } = await client.search_collection(
    COLLECTION_NAME,
    queryVector,
    topK
  );

  if (err) {
    throw new Error(typeof err === "string" ? err : JSON.stringify(err));
  }

  const results = Array.isArray(response?.result) ? response.result : [];
  return results.map((hit) => ({
    id: hit.id,
    score: hit.score,
    title: hit.payload?.title,
    content: hit.payload?.content,
    chunkIndex: hit.payload?.chunk_index,
    documentId: hit.payload?.document_id,
  }));
}

export function formatContextForPrompt(hits) {
  if (!hits || hits.length === 0) return "";
  const blocks = hits.map((h, i) => {
    const header = `Source ${i + 1} — ${h.title || "Untitled"} (chunk ${h.chunkIndex ?? "?"}, score ${h.score?.toFixed?.(3) ?? h.score})`;
    return `${header}\n${h.content || ""}`;
  });
  return blocks.join("\n\n---\n\n");
}


