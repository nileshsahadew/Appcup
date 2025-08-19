# Mauritius Travel Assistant

This is a [Next.js](https://nextjs.org) project that provides a RAG (Retrieval-Augmented Generation) system for Mauritius travel information using Google Gemini AI and Qdrant vector database.

## Prerequisites

1. **Google Gemini API Key**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Qdrant Database**: Install and run Qdrant locally or use a cloud instance

## Environment Setup

Create a `.env.local` file in the root directory with the following variables:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=http://localhost:6333/
```

## Installation

```bash
npm install
```

## Setup Qdrant

1. Install Qdrant (if running locally):
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or download from https://qdrant.tech/documentation/guides/installation/
   ```

2. The database will be available at http://localhost:6333/dashboard

## Data Ingestion

The project includes several scripts for ingesting data:

```bash
# Ingest data to local embeddings (no Qdrant required)
npm run ingest-local

# Ingest data to Qdrant vector database
npm run ingest-qdrant

# Run the complete RAG pipeline (ingest + query)
npm run rag

# Query the existing database
npm run query
```

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.js`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
