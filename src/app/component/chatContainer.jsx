"use client";
import { Typography, Box } from "@mui/material";

function ChatContainer({ chatMessages }) {
  // If there are no messages, display the welcome prompt
  if (!chatMessages || chatMessages.length === 0) {
    return (
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%", // Ensure the welcome message is centered in the container
          textAlign: "center",
          padding: 2,
        }}
      >
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Plan Your Perfect Itinerary
        </Typography>
        <Typography variant="body1" color="text.secondary" gutterBottom>
          To get started, tell me what types of places you'd like to explore in
          Mauritius. I'll use your preferences to suggest some ideal
          destinations just for you.
        </Typography>
      </Box>
    );
  }

  // If there are messages, render the chat history
  return (
    <Box
      sx={{
        marginLeft: "3%",
        marginRight: "5%",
        width: "auto",
        height: "auto",
        flexGrow: 1,
        // The main container for all chat bubbles, using flexbox for vertical stacking
        display: "flex",
        flexDirection: "column",
        gap: 2, // Space between messages
        overflowY: "auto", // Allows scrolling if messages exceed container height
        padding: 2,
      }}
    >
      {chatMessages.map((msg, index) => (
        <Box
          key={index}
          sx={{
            // Use alignSelf to position messages on the left or right
            alignSelf: msg.type === "user" ? "flex-end" : "flex-start",
            maxWidth: "70%", // Prevent messages from taking up the full width
            padding: "12px 16px",
            borderRadius: "20px",
            color: msg.type === "user" ? "white" : "black",
            // Set different background colors for user and system messages
            backgroundColor: msg.type === "user" ? "#1976d2" : "#e0e0e0",
          }}
        >
          <Typography variant="body1">{msg.message}</Typography>
        </Box>
      ))}
    </Box>
  );
}
export default ChatContainer;
