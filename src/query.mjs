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

const client = new Qdrant("http://localhost:6333/");
const COLLECTION_NAME = "mauritius_knowledge";

async function embedText(text) {
  const embedResp = await embeddingModel.embedContent(text);
  return embedResp.embedding.values;
}

async function search(query, k = 3) {
  const queryVector = await embedText(query);
  const { err, response } = await client.search_collection(COLLECTION_NAME, queryVector, k);
  if (err) {
    throw new Error(typeof err === "string" ? err : JSON.stringify(err));
  }

  console.log(`\n🔍 Query: ${query}`);
  const hits = response?.result || [];
  if (!Array.isArray(hits) || hits.length === 0) {
    console.log("No results found.");
    return;
  }

  hits.forEach((hit, i) => {
    console.log(`\nResult ${i + 1}:`);
    console.log("Score:", hit.score);
    if (hit.payload) {
      console.log(hit);
      //console.log("Title:", hit.payload.title);
      //console.log("Chunk:", hit.payload.content);
      console.log("Meta:", {
        chunk_index: hit.payload.chunk_index,
        document_id: hit.payload.document_id,
      });
    } else {
      console.log(hit);
    }
  });
}

await search("Best time to visit Mauritius");
process.exit(0);
