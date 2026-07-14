import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Database, DatabaseZap, Send, ChevronDown, ChevronRight, Terminal } from 'lucide-react';
import './index.css';

const API_BASE = 'http://127.0.0.1:8000/api';

// Collapsible panel that shows the agent's reasoning steps after each answer.
function ReasoningPanel({ steps }) {
  const [expanded, setExpanded] = useState(false);

  if (!steps || steps.length === 0) return null;

  return (
    <div className="reasoning-panel">
      <div className="reasoning-header" onClick={() => setExpanded(!expanded)}>
        {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <span>View Agent Reasoning ({steps.length} steps)</span>
      </div>
      {expanded && (
        <div className="reasoning-content">
          {steps.map((step, idx) => (
            <div key={idx} className="step-item">
              <div className="step-action">
                <Terminal size={14} /> {step.action}
              </div>
              <div className="step-input">{step.input}</div>
              <div className="step-result">{step.result_preview}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function App() {
  const [schema, setSchema] = useState({});
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Load the DB schema for the sidebar on first render
    axios.get(`${API_BASE}/schema`)
      .then(res => setSchema(res.data))
      .catch(err => console.error('Failed to fetch schema', err));

    setMessages([
      { role: 'agent', content: 'Hello! I am your AI Data Assistant. Ask me questions about the database!' }
    ]);
  }, []);

  // Keep the latest message in view as the conversation grows
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/chat`, { question: userMessage });
      setMessages(prev => [...prev, {
        role: 'agent',
        content: response.data.final_answer,
        steps: response.data.steps,
      }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'agent', content: 'Sorry, I encountered an error connecting to the backend API.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar -- schema browser */}
      <div className="sidebar">
        <div className="logo-area">
          <div className="logo-icon">
            <DatabaseZap size={20} />
          </div>
          <div className="logo-text">DataInsight AI</div>
        </div>

        <div className="section-title">Database Schema</div>

        {Object.keys(schema).length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Loading schema...</div>
        ) : (
          Object.entries(schema).map(([tableName, columns]) => (
            <div key={tableName} className="schema-table">
              <div className="table-header">
                <Database size={14} color="var(--accent-color)" />
                {tableName}
              </div>
              <div className="table-columns">
                {columns.map(col => (
                  <div key={col.name} className="column-item">
                    <span className="column-name">
                      {col.name} {col.primary_key && <span style={{ color: 'var(--accent-color)' }}>(PK)</span>}
                    </span>
                    <span className="column-type">{col.type}</span>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Main chat area */}
      <div className="chat-area">
        <div className="messages-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-wrapper message-${msg.role}`}>
              <div className="message-content" style={{ display: 'flex', flexDirection: 'column' }}>
                <div className="message-bubble">
                  {msg.content}
                </div>
                {msg.steps && <ReasoningPanel steps={msg.steps} />}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message-wrapper message-agent">
              <div className="message-bubble" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                Thinking...
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="input-area" onSubmit={handleSend}>
          <div className="input-container">
            <input
              type="text"
              className="chat-input"
              placeholder="Ask anything about the data..."
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              disabled={isLoading}
            />
            <button type="submit" className="send-button" disabled={!inputValue.trim() || isLoading}>
              <Send size={18} />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default App;
