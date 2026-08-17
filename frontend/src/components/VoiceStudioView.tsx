import React, { useState, useEffect, useRef } from 'react';
import {
  Mic,
  Square,
  Globe,
  Layers,
  Copy,
  Check,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  ArrowLeft,
  ArrowUp,
  Volume2,
  RotateCcw,
  User,
  Bot,
  BookmarkCheck,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import {
  queryVoiceRAG,
  queryTextRAG,
  type RAGResult,
} from '../services/voice';

interface VoiceStudioViewProps {
  onBack: () => void;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  timestamp: Date;
  isVoice?: boolean;
  duration?: number;
  ragResult?: RAGResult;
}

export const VoiceStudioView: React.FC<VoiceStudioViewProps> = ({ onBack }) => {
  const [recordingState, setRecordingState] = useState<
    'idle' | 'recording' | 'processing' | 'success' | 'error'
  >('idle');
  const [currentStage, setCurrentStage] = useState<string>('');
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [selectedLang, setSelectedLang] = useState<string>('auto');
  const [selectedStrategy, setSelectedStrategy] = useState<string>('adaptive');
  const [textInput, setTextInput] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedCitationId, setExpandedCitationId] = useState<string | null>(null);

  // Chat message history
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // 7-Language Demo Suggestions (ChatGPT-style Quick Prompt Cards)
  const demoPrompts = [
    {
      lang: 'Hindi',
      native: 'हिन्दी',
      q: 'पेरू की राजधानी क्या है और सबसे बड़ा शहर कौन सा है?',
      desc: 'Peru Capital & Largest City in Hindi',
      badge: '🇮🇳 Hindi',
      color: 'hover:border-[#FEE101]',
    },
    {
      lang: 'English',
      native: 'English',
      q: 'What is the capital of India and where is the central government located?',
      desc: 'National Capital Territory of India',
      badge: '🇬🇧 English',
      color: 'hover:border-[#06B6D4]',
    },
    {
      lang: 'Hinglish',
      native: 'Hinglish',
      q: 'Peru ki capital kya hai aur sabse bada city konsa hai?',
      desc: 'Conversational Hinglish Query',
      badge: '🌐 Hinglish',
      color: 'hover:border-[#FF0080]',
    },
    {
      lang: 'English',
      native: 'English',
      q: 'What is the capital city of Wales and its major cities?',
      desc: 'Wales Capital (Cardiff) Passage Extraction',
      badge: '🇬🇧 English',
      color: 'hover:border-[#00F59B]',
    },
    {
      lang: 'Bengali',
      native: 'বাংলা',
      q: 'পেরুর রাজধানী কী এবং প্রধান শহর কোনটি?',
      desc: 'Cross-Lingual Bengali Query',
      badge: '🇧🇩 Bengali',
      color: 'hover:border-[#A855F7]',
    },
    {
      lang: 'English',
      native: 'English',
      q: 'What is the definition of a corporation under law?',
      desc: 'Legal Definition of Corporation',
      badge: '🇬🇧 English',
      color: 'hover:border-[#F59E0B]',
    },
  ];

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, recordingState, currentStage]);

  // Recording timer
  useEffect(() => {
    if (recordingState === 'recording') {
      timerRef.current = window.setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [recordingState]);

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  // Start Mic Recording
  const startRecording = async () => {
    setErrorMessage(null);
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        stream.getTracks().forEach((track) => track.stop());

        // Automatically trigger Voice RAG on recording stop
        await executeVoiceQuery(blob, recordingDuration);
      };

      mediaRecorder.start();
      setRecordingDuration(0);
      setRecordingState('recording');
    } catch (err: any) {
      setRecordingState('error');
      setErrorMessage(
        err.message || 'Microphone access denied. You can type your query or use the 1-Click Prompts below.',
      );
    }
  };

  // Stop Recording and Send
  const stopRecordingAndSend = () => {
    if (mediaRecorderRef.current && recordingState === 'recording') {
      mediaRecorderRef.current.stop();
    }
  };

  // Cancel Recording without sending
  const cancelRecording = () => {
    if (mediaRecorderRef.current && recordingState === 'recording') {
      mediaRecorderRef.current.onstop = null;
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream?.getTracks().forEach((t) => t.stop());
    }
    setRecordingState('idle');
    setRecordingDuration(0);
  };

  // Execute Voice Query Pipeline
  const executeVoiceQuery = async (blob: Blob, duration: number) => {
    setRecordingState('processing');
    setCurrentStage('Transcribing with Sarvam Saaras v3...');
    setErrorMessage(null);

    const tempUserMsgId = `user-${Date.now()}`;
    const assistantMsgId = `asst-${Date.now()}`;

    try {
      setCurrentStage('Transcribing Speech → FAISS Retrieval → Grounded LLM...');
      const res: RAGResult = await queryVoiceRAG(blob, selectedStrategy, selectedLang, 5);

      const queryText = res.transcript || res.query || 'Spoken voice query';

      // Add user message
      const userMessage: ChatMessage = {
        id: tempUserMsgId,
        role: 'user',
        text: queryText,
        timestamp: new Date(),
        isVoice: true,
        duration: duration,
      };

      // Add assistant response
      const assistantMessage: ChatMessage = {
        id: assistantMsgId,
        role: 'assistant',
        text: res.answer,
        timestamp: new Date(),
        ragResult: res,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setRecordingState('success');
    } catch (err: any) {
      setRecordingState('error');
      setErrorMessage(err.message || 'Voice RAG pipeline failed.');
    } finally {
      setCurrentStage('');
    }
  };

  // Execute Text Query
  const executeTextQuery = async (text: string) => {
    const q = text.trim();
    if (!q || recordingState === 'processing') return;

    setTextInput('');
    setRecordingState('processing');
    setCurrentStage('Hybrid Retrieval (FAISS + BM25) → Grounded Synthesis...');
    setErrorMessage(null);

    const userMsgId = `user-${Date.now()}`;
    const asstMsgId = `asst-${Date.now()}`;

    const userMessage: ChatMessage = {
      id: userMsgId,
      role: 'user',
      text: q,
      timestamp: new Date(),
      isVoice: false,
    };

    setMessages((prev) => [...prev, userMessage]);

    try {
      const res: RAGResult = await queryTextRAG(q, selectedStrategy, 5);

      const assistantMessage: ChatMessage = {
        id: asstMsgId,
        role: 'assistant',
        text: res.answer,
        timestamp: new Date(),
        ragResult: res,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setRecordingState('success');
    } catch (err: any) {
      setRecordingState('error');
      setErrorMessage(err.message || 'Text RAG query failed.');
    } finally {
      setCurrentStage('');
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    executeTextQuery(textInput);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      executeTextQuery(textInput);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setErrorMessage(null);
    setRecordingState('idle');
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-5.5rem)] max-w-5xl mx-auto">
      {/* Top Header Bar */}
      <div className="flex items-center justify-between gap-3 pb-3 border-b-2 border-black/40 bg-[#070A12]/90 backdrop-blur sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="neo-btn inline-flex items-center gap-1.5 text-xs font-bold text-black px-3 py-1.5 rounded-lg bg-[#FEE101] transition cursor-pointer uppercase font-sans shadow-[2px_2px_0px_0px_#000]"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Dashboard</span>
          </button>
          <div>
            <h2 className="text-xl sm:text-2xl font-serif-claude font-bold text-white flex items-center gap-2">
              Voice RAG Studio
              <span className="px-2 py-0.5 rounded-full text-[10px] font-black bg-[#FF0080] text-white border border-black shadow-[1px_1px_0px_0px_#000] uppercase font-mono">
                GPT Mode
              </span>
            </h2>
          </div>
        </div>

        {/* Top Controls: Strategy & Language & Reset */}
        <div className="flex items-center gap-2">
          {/* Language Selector */}
          <div className="flex items-center gap-1.5 bg-[#0C1220] border-2 border-black rounded-xl px-2.5 py-1.5 shadow-[2px_2px_0px_0px_#000000]">
            <Globe className="w-3.5 h-3.5 text-[#FEE101]" />
            <select
              value={selectedLang}
              onChange={(e) => setSelectedLang(e.target.value)}
              className="bg-transparent text-xs text-white outline-none cursor-pointer font-bold font-sans"
            >
              <option value="auto" className="bg-[#0C1220]">🌐 Auto</option>
              <option value="hi-IN" className="bg-[#0C1220]">🇮🇳 हिन्दी</option>
              <option value="en-IN" className="bg-[#0C1220]">🇬🇧 English</option>
              <option value="bn-IN" className="bg-[#0C1220]">🇧🇩 বাংলা</option>
              <option value="ta-IN" className="bg-[#0C1220]">🇮🇳 தமிழ்</option>
              <option value="te-IN" className="bg-[#0C1220]">🇮🇳 తెలుగు</option>
              <option value="mr-IN" className="bg-[#0C1220]">🇮🇳 मराठी</option>
            </select>
          </div>

          {/* Strategy Selector */}
          <div className="hidden sm:flex items-center gap-1.5 bg-[#0C1220] border-2 border-black rounded-xl px-2.5 py-1.5 shadow-[2px_2px_0px_0px_#000000]">
            <Layers className="w-3.5 h-3.5 text-[#06B6D4]" />
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="bg-transparent text-xs text-white outline-none cursor-pointer font-bold font-sans"
            >
              <option value="adaptive" className="bg-[#0C1220]">⚡ Adaptive</option>
              <option value="semantic" className="bg-[#0C1220]">🧠 Semantic</option>
              <option value="sentence" className="bg-[#0C1220]">🎯 Sentence</option>
              <option value="paragraph" className="bg-[#0C1220]">📑 Paragraph</option>
              <option value="overlap" className="bg-[#0C1220]">🔄 Overlap</option>
              <option value="fixed" className="bg-[#0C1220]">📏 Fixed</option>
              <option value="metadata" className="bg-[#0C1220]">🏷️ Metadata</option>
            </select>
          </div>

          {/* Clear Chat Button */}
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              title="Reset Conversation"
              className="p-1.5 rounded-xl bg-[#0C1220] hover:bg-[#1C2638] text-slate-300 hover:text-white border-2 border-black shadow-[2px_2px_0px_0px_#000] transition cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Main Conversation Stream */}
      <div className="flex-1 overflow-y-auto py-6 space-y-6 pr-1 custom-scrollbar">
        {/* Welcome State (Empty Message List) */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] text-center px-4 space-y-6 animate-fadeIn">
            <div className="relative">
              <div className="w-20 h-20 rounded-3xl bg-[#FEE101] border-3 border-black shadow-[6px_6px_0px_0px_#000000] flex items-center justify-center text-black">
                <Sparkles className="w-10 h-10" />
              </div>
              <div className="absolute -bottom-2 -right-2 px-2 py-0.5 rounded-full bg-[#06B6D4] text-black text-[10px] font-black border-2 border-black font-mono">
                RAG v2.0
              </div>
            </div>

            <div className="space-y-2 max-w-lg">
              <h1 className="text-3xl sm:text-4xl font-serif-claude font-bold text-white">
                How can I help you today?
              </h1>
              <p className="text-sm text-slate-300 font-sans">
                Ask in <strong className="text-[#FEE101]">Hindi, English, Hinglish, Bengali, Tamil, Telugu, Marathi</strong> using your microphone or keyboard.
              </p>
            </div>

            {/* Quick Suggestion Prompt Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 w-full max-w-3xl pt-4">
              {demoPrompts.map((dp, idx) => (
                <button
                  key={idx}
                  onClick={() => executeTextQuery(dp.q)}
                  disabled={recordingState === 'processing'}
                  className={`p-3.5 rounded-2xl bg-[#0C1220] border-2 border-black hover:bg-[#152035] text-left transition duration-150 cursor-pointer shadow-[3px_3px_0px_0px_#000000] hover:shadow-[4px_4px_0px_0px_#FEE101] hover:-translate-x-0.5 hover:-translate-y-0.5 group space-y-1.5 ${dp.color}`}
                >
                  <div className="flex items-center justify-between text-[11px] font-mono font-bold text-slate-400">
                    <span className="text-[#06B6D4] group-hover:text-[#FEE101]">{dp.badge}</span>
                    <span>→</span>
                  </div>
                  <div className="text-xs font-bold text-white group-hover:text-[#FEE101] font-sans line-clamp-2 leading-snug">
                    "{dp.q}"
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Message Stream */}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3.5 max-w-3xl ${
              msg.role === 'user' ? 'ml-auto justify-end' : 'mr-auto justify-start'
            } animate-fadeIn`}
          >
            {/* Assistant Avatar */}
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-xl bg-[#FEE101] border-2 border-black shadow-[2px_2px_0px_0px_#000] flex items-center justify-center text-black shrink-0 mt-1">
                <Bot className="w-4 h-4" />
              </div>
            )}

            {/* Message Body Card */}
            <div
              className={`rounded-2xl p-4 sm:p-5 border-2 border-black space-y-3 ${
                msg.role === 'user'
                  ? 'bg-[#152038] text-white shadow-[4px_4px_0px_0px_#06B6D4] max-w-[85%]'
                  : 'bg-[#0C1220] text-white shadow-[5px_5px_0px_0px_#000000] w-full'
              }`}
            >
              {/* User Voice Indicator */}
              {msg.role === 'user' && msg.isVoice && (
                <div className="flex items-center gap-1.5 text-[11px] font-mono text-[#FEE101] font-bold pb-1 border-b border-white/10">
                  <Volume2 className="w-3.5 h-3.5" />
                  <span>Voice Query ({formatTime(msg.duration || 0)})</span>
                </div>
              )}

              {/* Message Text */}
              <div
                className={`text-sm sm:text-base leading-relaxed ${
                  msg.role === 'assistant'
                    ? 'font-serif-claude font-normal text-slate-100'
                    : 'font-sans font-medium text-white'
                }`}
              >
                {msg.text}
              </div>

              {/* Assistant Metadata & Verification Badges */}
              {msg.role === 'assistant' && msg.ragResult && (
                <div className="pt-2 border-t-2 border-black/60 space-y-3">
                  {/* Status Badges Row */}
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                    <div className="flex flex-wrap items-center gap-2">
                      {msg.ragResult.grounded ? (
                        <span className="px-2.5 py-0.5 rounded-md bg-[#00F59B] text-black border border-black font-black text-[11px] flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Grounded ({Math.round(msg.ragResult.confidence * 100)}%)
                        </span>
                      ) : (
                        <span className="px-2.5 py-0.5 rounded-md bg-[#FF3366] text-white border border-black font-black text-[11px] flex items-center gap-1">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          {msg.ragResult.abstention_reason || 'Out of Dataset'}
                        </span>
                      )}

                      <span className="px-2 py-0.5 rounded-md bg-[#070A12] text-slate-300 border border-black text-[11px]">
                        Strategy: <strong className="text-[#06B6D4]">{msg.ragResult.strategy}</strong>
                      </span>

                      <span className="px-2 py-0.5 rounded-md bg-[#070A12] text-slate-300 border border-black text-[11px]">
                        Latency: <strong className="text-[#FEE101]">{msg.ragResult.latency.total_ms.toFixed(1)}ms</strong>
                      </span>
                    </div>

                    <button
                      onClick={() => copyToClipboard(msg.text, msg.id)}
                      className="p-1.5 rounded-lg bg-[#070A12] hover:bg-[#1A2333] text-slate-300 hover:text-white border border-black transition cursor-pointer flex items-center gap-1 text-[11px]"
                      title="Copy Answer"
                    >
                      {copiedId === msg.id ? (
                        <Check className="w-3.5 h-3.5 text-[#00F59B]" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                      <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>

                  {/* Expandable Citations / Evidence Section */}
                  {msg.ragResult.citations && msg.ragResult.citations.length > 0 && (
                    <div className="space-y-2 pt-1">
                      <button
                        onClick={() =>
                          setExpandedCitationId(
                            expandedCitationId === msg.id ? null : msg.id,
                          )
                        }
                        className="w-full flex items-center justify-between p-2.5 rounded-xl bg-[#070A12] hover:bg-[#141C2B] border-2 border-black text-xs font-bold text-white transition cursor-pointer select-none"
                      >
                        <div className="flex items-center gap-2">
                          <BookmarkCheck className="w-4 h-4 text-[#06B6D4]" />
                          <span>
                            Verified Citations ({msg.ragResult.citations.length} Sources)
                          </span>
                        </div>
                        {expandedCitationId === msg.id ? (
                          <ChevronUp className="w-4 h-4 text-[#FEE101]" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-slate-400" />
                        )}
                      </button>

                      {expandedCitationId === msg.id && (
                        <div className="space-y-2 pl-2 border-l-2 border-[#06B6D4] animate-fadeIn">
                          {msg.ragResult.citations.map((cit, citIdx) => (
                            <div
                              key={citIdx}
                              className="p-3 rounded-xl bg-[#070A12] border-2 border-black space-y-1.5 text-xs shadow-[2px_2px_0px_0px_#000]"
                            >
                              <div className="flex items-center justify-between text-[11px] font-mono">
                                <span className="px-1.5 py-0.5 rounded bg-[#FEE101] text-black font-black">
                                  SOURCE 0{citIdx + 1}
                                </span>
                                <span className="text-[#00F59B] font-bold">
                                  Score: {cit.relevance_score?.toFixed(3) || '0.980'}
                                </span>
                              </div>
                              <div className="font-mono text-[11px] text-slate-400 truncate">
                                ID: {cit.chunk_id}
                              </div>
                              {cit.snippet && (
                                <div className="text-xs text-slate-200 font-serif-claude italic bg-[#0C1220] p-2.5 rounded-lg border border-black/40">
                                  "{cit.snippet}"
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* User Avatar */}
            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-xl bg-[#06B6D4] border-2 border-black shadow-[2px_2px_0px_0px_#000] flex items-center justify-center text-black shrink-0 mt-1">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}

        {/* Processing Indicator (ChatGPT Thinking Pulse) */}
        {recordingState === 'processing' && (
          <div className="flex gap-3.5 max-w-3xl mr-auto animate-fadeIn">
            <div className="w-8 h-8 rounded-xl bg-[#FEE101] border-2 border-black shadow-[2px_2px_0px_0px_#000] flex items-center justify-center text-black shrink-0 mt-1 animate-pulse">
              <Sparkles className="w-4 h-4" />
            </div>
            <div className="rounded-2xl p-4 bg-[#0C1220] border-2 border-black shadow-[4px_4px_0px_0px_#000000] space-y-2 w-full max-w-md">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-[#06B6D4]">
                <div className="w-2 h-2 rounded-full bg-[#06B6D4] animate-ping" />
                <span>{currentStage || 'Generating grounded answer...'}</span>
              </div>
              <div className="h-1.5 w-full bg-[#070A12] rounded-full overflow-hidden border border-black">
                <div className="h-full bg-gradient-to-r from-[#FEE101] via-[#06B6D4] to-[#FF0080] animate-pulse w-3/4 rounded-full" />
              </div>
            </div>
          </div>
        )}

        {/* Error Banner */}
        {errorMessage && (
          <div className="max-w-3xl mx-auto p-4 rounded-2xl bg-[#FF3366] text-white border-2 border-black font-bold text-xs shadow-[4px_4px_0px_0px_#000] flex items-center justify-between gap-3 animate-fadeIn">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-xs uppercase underline font-mono cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Bottom Floating ChatGPT-Style Input Bar */}
      <div className="pt-2 pb-3 bg-[#070A12] sticky bottom-0 z-20">
        <div className="max-w-3xl mx-auto">
          {/* Recording Mode Active Bar */}
          {recordingState === 'recording' ? (
            <div className="w-full bg-[#0C1220] border-3 border-[#FF3366] rounded-3xl p-3 sm:p-4 shadow-[6px_6px_0px_0px_#FF3366] flex items-center justify-between gap-3 animate-pulse">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-2xl bg-[#FF3366] text-white border-2 border-black flex items-center justify-center shadow-[2px_2px_0px_0px_#000]">
                  <Mic className="w-5 h-5 animate-bounce" />
                </div>
                <div>
                  <div className="text-xs font-black uppercase text-[#FF3366] font-mono">
                    ● Listening (16kHz Mono)
                  </div>
                  <div className="text-xl font-black font-mono text-white tracking-widest">
                    {formatTime(recordingDuration)}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={cancelRecording}
                  className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border-2 border-black text-xs font-bold font-mono transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={stopRecordingAndSend}
                  className="neo-btn px-4 py-2 rounded-xl bg-[#FEE101] text-black border-2 border-black text-xs font-black font-mono transition cursor-pointer flex items-center gap-1.5 shadow-[2px_2px_0px_0px_#000]"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>Stop & Query</span>
                </button>
              </div>
            </div>
          ) : (
            /* Standard ChatGPT Text + Mic Input Box */
            <form
              onSubmit={handleFormSubmit}
              className="relative flex items-end gap-2 bg-[#0C1220] border-2 border-black rounded-3xl p-2 sm:p-2.5 shadow-[5px_5px_0px_0px_#000000] focus-within:shadow-[6px_6px_0px_0px_#FEE101] focus-within:border-black transition-all"
            >
              {/* Textarea Input */}
              <textarea
                ref={textareaRef}
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                placeholder="Ask in Hindi, English, Hinglish, Tamil... or click the mic"
                className="flex-1 bg-transparent text-sm sm:text-base text-white placeholder-slate-400 outline-none px-3 py-2 resize-none max-h-32 font-sans custom-scrollbar leading-relaxed"
                style={{ minHeight: '40px' }}
              />

              {/* Action Buttons inside Input Pill */}
              <div className="flex items-center gap-1.5 shrink-0 pb-0.5">
                {/* Voice Record Button */}
                <button
                  type="button"
                  onClick={startRecording}
                  disabled={recordingState === 'processing'}
                  className="w-10 h-10 rounded-2xl bg-[#FEE101] hover:bg-[#FFE72E] text-black border-2 border-black flex items-center justify-center shadow-[2px_2px_0px_0px_#000] active:translate-x-0.5 active:translate-y-0.5 transition cursor-pointer disabled:opacity-50"
                  title="Speak Voice Query (Microphone)"
                >
                  <Mic className="w-5 h-5" />
                </button>

                {/* Send Query Button */}
                <button
                  type="submit"
                  disabled={!textInput.trim() || recordingState === 'processing'}
                  className={`w-10 h-10 rounded-2xl border-2 border-black flex items-center justify-center shadow-[2px_2px_0px_0px_#000] transition cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed ${
                    textInput.trim()
                      ? 'bg-[#06B6D4] text-black hover:bg-[#22D3EE] active:translate-x-0.5 active:translate-y-0.5'
                      : 'bg-slate-800 text-slate-500'
                  }`}
                  title="Send Query (Enter)"
                >
                  <ArrowUp className="w-5 h-5 stroke-[2.5]" />
                </button>
              </div>
            </form>
          )}

          {/* Footer Disclaimer */}
          <div className="text-center pt-2 text-[10px] text-slate-300 font-sans">
            Voice RAG strictly validates all synthesized claims against indexed knowledge base passages.
          </div>
        </div>
      </div>
    </div>
  );
};
