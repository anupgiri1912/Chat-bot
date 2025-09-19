// import { Box } from "@mui/material";
// import "./App.css";


// import {  Route, Routes } from "react-router-dom";
// import HomePages from "./Homepages";
// import ReportPages from "./ReportPages";
// import ResultsPage from "./ResultsPage";
// import Header from "./Header";


// import React, { useState } from 'react';
// import './Chatbot.css';

// function App() {
//   return (
//     <Box>
//       {/* <Header/>
  
//       <Box>
//         <Routes>
//           <Route path="/" element={<HomePages />} />
//           <Route path="/report" element={<ReportPages />} />
//           <Route path="/result" element={<ResultsPage />} />
//         </Routes>
//       </Box> */}

//     </Box>
//   );
// }

// export default App;
import React, { useState } from 'react';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([
    { text: "Hi! How can I help you today?", sender: "bot" }
  ]);
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: "user" };
    setMessages([...messages, userMessage]);

    // Simulate bot response
    setTimeout(() => {
      const botMessage = { text: `You said: "${input}"`, sender: "bot" };
      setMessages(prev => [...prev, botMessage]);
    }, 1000);

    setInput("");
  };

  return (
    <div className="chatbot-container">
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
      </div>
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Type your message..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
};

export default App;
