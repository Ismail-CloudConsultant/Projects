"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const SUGGESTIONS = [
  "Analyze the health of AAPL",
  "What is the Sharpe ratio?",
  "Compare MSFT and GOOGL",
  "Explain P/E ratio",
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text: string) {
    if (!text.trim() || loading) return;
    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId, user_id: "guest" }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setSessionId(data.session_id);
      setMessages((prev) => [...prev, { role: "assistant", content: data.response }]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Something went wrong.";
      setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${msg}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto px-4">
      {/* Header */}
      <div className="flex items-center gap-3 py-5 border-b border-gray-800">
        <div className="w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center text-sm font-bold">F</div>
        <div>
          <p className="font-semibold text-white">Finbot</p>
          <p className="text-xs text-gray-400">AI Financial Advisor</p>
        </div>
        {sessionId && (
          <button
            onClick={() => { setMessages([]); setSessionId(null); }}
            className="ml-auto text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            New chat
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
            <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-3xl">
              📈
            </div>
            <div>
              <h1 className="text-xl font-semibold text-white mb-1">How can I help you today?</h1>
              <p className="text-sm text-gray-400">Ask me to analyze stocks, explain metrics, or track your portfolio.</p>
            </div>
            <div className="grid grid-cols-2 gap-2 w-full max-w-md">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-left text-sm px-3 py-2.5 rounded-xl bg-gray-800/60 border border-gray-700/50 hover:border-emerald-500/40 hover:bg-gray-800 text-gray-300 transition-all"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            {m.role === "assistant" && (
              <div className="w-7 h-7 rounded-full bg-emerald-500 flex-shrink-0 flex items-center justify-center text-xs font-bold mt-0.5">F</div>
            )}
            <div
              className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-emerald-600 text-white rounded-br-sm"
                  : "bg-gray-800 text-gray-100 rounded-bl-sm"
              }`}
            >
              {m.role === "assistant" ? (
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>,
                    strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
                    h1: ({ children }) => <h1 className="font-bold text-white mb-1">{children}</h1>,
                    h2: ({ children }) => <h2 className="font-semibold text-white mb-1">{children}</h2>,
                    h3: ({ children }) => <h3 className="font-medium text-emerald-400 mb-1">{children}</h3>,
                    code: ({ children }) => <code className="bg-gray-700 px-1 rounded text-emerald-300 text-xs">{children}</code>,
                  }}
                >
                  {m.content}
                </ReactMarkdown>
              ) : (
                m.content
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full bg-emerald-500 flex-shrink-0 flex items-center justify-center text-xs font-bold">F</div>
            <div className="bg-gray-800 px-4 py-3 rounded-2xl rounded-bl-sm">
              <div className="flex gap-1 items-center h-4">
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="py-4 border-t border-gray-800">
        <form
          onSubmit={(e) => { e.preventDefault(); send(input); }}
          className="flex gap-2 bg-gray-800 border border-gray-700 rounded-2xl px-4 py-3 focus-within:border-emerald-500/50 transition-colors"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about stocks, metrics, or your portfolio..."
            className="flex-1 bg-transparent text-sm text-gray-100 placeholder-gray-500 outline-none"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="w-8 h-8 rounded-xl bg-emerald-500 hover:bg-emerald-400 disabled:bg-gray-700 disabled:text-gray-500 flex items-center justify-center transition-colors text-white flex-shrink-0"
          >
            <svg className="w-4 h-4 rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </form>
        <p className="text-center text-xs text-gray-600 mt-2">
          Finbot may make mistakes. Not financial advice.
        </p>
      </div>
    </div>
  );
}
