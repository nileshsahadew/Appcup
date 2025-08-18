import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { retrieveTopK, formatContextForPrompt } from "@/src/lib/retrieval.mjs";

export const runtime = "edge";

export async function POST(req) {
  try {
    const { message, history } = await req.json();

    // Correct the role mapping to use "human" and "ai" as required by LangChain.
    const mappedHistory = history.map((msg) => {
      if (msg.type === "user") {
        return new HumanMessage({ content: msg.message });
      } else if (msg.type === "assistant") {
        return new AIMessage({ content: msg.message });
      }
    });

    // Instantiate the GoogleGenerativeAI model.
    const model = new ChatGoogleGenerativeAI({
      apiKey: process.env.GEMINI_API_KEY, // Replace with your API key or environment variable
      model: "gemini-2.5-flash-preview-05-20",
      temperature: 0.7,
      streaming: true,
    });

    // Retrieve top-K context from Qdrant and prepend as system message
    const retrieved = await retrieveTopK(message, 5);
    const contextBlock = formatContextForPrompt(retrieved);

    const systemPreamble = new AIMessage({
      content:
        (contextBlock
          ? `Use the following context to answer. If context is insufficient, say so.
Context:\n${contextBlock}`
          : "No external context available. Answer concisely.") +
        "\nAlways provide short, factual answers.\n",
    });

    const stream = await model.stream([systemPreamble, ...mappedHistory, new HumanMessage({ content: message })]);

    // We can then pipe the LangChain stream directly to a new Response object.
    const textEncoder = new TextEncoder();
    const readableStream = new ReadableStream({
      async start(controller) {
        for await (const chunk of stream) {
          controller.enqueue(textEncoder.encode(chunk.content));
        }
        controller.close();
      },
    });

    return new Response(readableStream, {
      headers: { "Content-Type": "text/plain" },
    });
  } catch (error) {
    console.error("LangChain API Error:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
