import React, { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);

  const sendQuery = async () => {
    const res = await axios.post("http://localhost:8000/chat", {
      query,
      history: messages
    });

    setMessages([
      ...messages,
      { role: "user", content: query },
      { role: "assistant", content: res.data.answer }
    ]);

    setQuery("");
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Policy Assistant</h2>

      <div>
        {messages.map((m, i) => (
          <div key={i}>
            <b>{m.role}:</b> {m.content}
          </div>
        ))}
      </div>

      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask policy question..."
      />
      <button onClick={sendQuery}>Send</button>
    </div>
  );
}

export default App;
