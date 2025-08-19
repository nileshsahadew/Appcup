import fs from "fs";
import path from "path";
import dotenv from "dotenv";
dotenv.config();

import { GoogleGenerativeAI } from "@google/generative-ai";
import { Qdrant } from "qdrant";

const KEY = "AIzaSyBlob6YbAnKxRpBDJSFoOnMGdDcH2JdWg4";

if (!KEY) {
  console.error("❌ Please set GEMINI_API_KEY in .env");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(KEY);
const embeddingModel = genAI.getGenerativeModel({
  model: "embedding-001",
});

// Initialize Qdrant client
const client = new Qdrant("http://localhost:6333/");

const COLLECTION_NAME = "mauritius_attractions";
const VECTOR_SIZE = 768; // Gemini embedding dimension

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
        embedding_vector: {
          size: VECTOR_SIZE,
          distance: "Cosine",
        },
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

  try {
    // Generate an embedding for the query text
    const embedResp = await embeddingModel.embedContent(queryText);
    const queryVector = embedResp.embedding.values;

    // Search the Qdrant collection for similar vectors
    const searchResults = await client.search_collection(COLLECTION_NAME, {
      vector: { embedding_vector: queryVector },
      limit: 5, // Return top 5 results
      with_payload: true, // Include the original text and metadata in the results
    });

    console.log(searchResults);
    console.log(`✅ Found ${searchResults.length} results.`);
    searchResults.forEach((result, index) => {
      console.log(`\n--- Result ${index + 1} ---`);
      console.log(`Score: ${result.score}`);
      console.log(`Title: ${result.payload.title}`);
      console.log(`Content: ${result.payload.content}`);
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

    // Test connection to Qdrant using a valid method
    try {
      const collections = await client.get_collection();
      console.log("✅ Qdrant client initialized successfully");
    } catch (error) {
      console.error("❌ Could not connect to Qdrant. Error details:");
      console.error("   Error message:", error.message);
      console.error("   Make sure Qdrant is running on localhost:6333");
      process.exit(1);
    }

    // Create collection
    await createCollection();

    const datasetDir = path.join(
      process.cwd(),
      "agents",
      "datasets",
      "mauritius_attractions_dataset"
    );
    if (!fs.existsSync(datasetDir)) {
      console.error("❌ Dataset directory not found:", datasetDir);
      process.exit(1);
    }

    const files = fs.readdirSync(datasetDir);
    let counter = 0;
    const points = [];

    for (const file of files) {
      if (file.endsWith(".json")) {
        const filePath = path.join(datasetDir, file);
        console.log(`\n📖 Ingesting file: ${file}`);
        const docs = JSON.parse(fs.readFileSync(filePath, "utf-8"));

        for (const doc of docs) {
          try {
            const embedResp = await embeddingModel.embedContent(
              JSON.stringify(doc)
            );
            const embedding = Array.from(embedResp.embedding.values);

            points.push({
              id: counter,
              vector: { embedding_vector: embedding },
              payload: {
                type: "doc",
                title: doc.name,
                content: JSON.stringify(doc),
                chunk_index: 0,
                source_file: file,
                document_id: doc.id,
              },
            });

            console.log(`✅ Processed doc ${counter} (${doc.name})`);
            counter++;
          } catch (error) {
            console.error(`❌ Error processing doc ${counter}:`, error.message);
            continue;
          }
        }
      }
      break;
    }

    // Insert points in batches
    console.log(points);
    if (points.length > 0) {
      console.log(points);
      await client.upload_points(COLLECTION_NAME, points);
      console.log(
        `🎯 Successfully inserted ${counter} chunks into Qdrant collection: ${COLLECTION_NAME}`
      );
    }

    console.log("✅ Data ingestion complete.\n");

    // Perform queries
    await queryCollection("What are the best beaches in Mauritius?");
    await queryCollection(
      "Tell me about the Black River Gorges National Park."
    );
    await queryCollection("What historical places can I visit?");

    process.exit(0);
  } catch (error) {
    console.error("❌ Error:", error);
    process.exit(1);
  }
}

main();
