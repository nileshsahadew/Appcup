"use client";
import {
  Button,
  Box,
  Typography,
  Divider,
  Paper,
  TextField,
  IconButton,
  CircularProgress,
} from "@mui/material";
import { KeyboardReturn } from "@mui/icons-material";
import ChatContainer from "./component/chatContainer";
import { useState } from "react";
import Image from "next/image";

export default function Home() {
  const [chatMessages, setChatMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    // Prevent sending empty messages or while loading
    if (inputMessage.trim() === "" || isLoading) return;

    // Add the user's message to the chat history
    const userMessage = { type: "user", message: inputMessage };
    setChatMessages((prevMessages) => [...prevMessages, userMessage]);

    // Clear the input field and set loading state
    setInputMessage("");
    setIsLoading(true);

    try {
      // Send the chat history and new message to the API route
      const response = await fetch("/api/agent", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: inputMessage,
          history: [...chatMessages, userMessage], // Send the full history for context
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Read the streaming response from the server
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = { type: "assistant", message: "" };

      // Initialize the assistant message in the chat state
      setChatMessages((prevMessages) => [...prevMessages, assistantMessage]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        assistantMessage.message += chunk;

        // Mutate the chatMessages state with the streaming chunks
        setChatMessages((prevMessages) => {
          const newMessages = [...prevMessages];
          const lastMessageIndex = newMessages.length - 1;
          newMessages[lastMessageIndex] = {
            ...newMessages[lastMessageIndex],
            message: assistantMessage.message,
          };
          return newMessages;
        });
      }
    } catch (error) {
      console.error("Failed to fetch from chat API:", error);
      // Add a friendly error message to the chat
      setChatMessages((prevMessages) => [
        ...prevMessages,
        { type: "system", message: "An error occurred. Please try again." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };
  return (
   <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        marginLeft: "18%",
        marginRight: "17%",
        marginTop: "1%",
        width: "auto",
        height: "83vh",
      }}
    >
      <ChatContainer chatMessages={chatMessages} />
      <div style={{ display: "flex", gap: "8px" }}>
        <TextField
          id="outlined-basic"
          label="Enter your message here"
          sx={{
            width: "auto",
            flexGrow: 1,
          }}
          multiline
          maxRows={3}
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          disabled={isLoading}
          onKeyPress={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSendMessage();
            }
          }}
        ></TextField>
        <IconButton
          sx={{
            backgroundColor: "primary.main",
            color: "white",
            padding: "12px",
            borderRadius: "25%",
            "&:hover": {
              backgroundColor: "primary.dark",
            },
          }}
          onClick={handleSendMessage}
          disabled={isLoading}
        >
          {isLoading ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            <KeyboardReturn fontSize="large" />
          )}
        </IconButton>
      </div>
    </Box>
  );
}
