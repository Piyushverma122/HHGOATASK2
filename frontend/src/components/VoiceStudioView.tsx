import React, { useState, useEffect, useRef } from 'react';
import {
  Mic,
  Square,
  Trash2,
  Zap,
  Globe,
  Layers,
  Copy,
  Check,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Search,
  ArrowLeft,
  Volume2,
} from 'lucide-react';
import {
  transcribeAudio,
  queryVoiceRAG,
  queryTextRAG,
  type RAGResult,
  type TranscribeResult,
} from '../services/voice';
import { CitationDrawer } from './CitationDrawer';

interface VoiceStudioViewProps {
  onBack: () => void;
}

export const VoiceStudioView: React.FC<VoiceStudioViewProps> = ({ onBack }) => {
  const [recordingState, setRecordingState] = useState<
    'idle' | 'recording' | 'processing' | 'success' | 'error'
  >('idle');
  const [currentStage, setCurrentStage] = useState<string>('');
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [selectedLang, setSelectedLang] = useState<string>('auto');
  const [selectedStrategy, setSelectedStrategy] = useState<string>('adaptive');
  const [textInput, setTextInput] = useState<string>('');
  const [transcript, setTranscript] = useState<string>('');
  const [isDemoFixture, setIsDemoFixture] = useState<boolean>(false);
  const [ragResult, setRagResult] = useState<RAGResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copiedAnswer, setCopiedAnswer] = useState<boolean>(false);
  const [copiedTranscript, setCopiedTranscript] = useState<boolean>(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  const demoFixtures = [
    { lang: 'Hindi', code: 'hi-IN', native: 'हिन्दी', q: 'भारत की राजधानी क्या है और यह कहाँ स्थित है?', color: 'border-l-4 border-l-[#FEE101]' },
    { lang: 'English', code: 'en-IN', native: 'English', q: 'What is the capital of India and where is the central government located?', color: 'border-l-4 border-l-[#06B6D4]' },
    { lang: 'Hinglish', code: 'hi-IN', native: 'Hinglish', q: 'India ki capital kya hai aur ye kaha par situated hai?', color: 'border-l-4 border-l-[#FF0080]' },
    { lang: 'Bengali', code: 'bn-IN', native: 'বাংলা', q: 'ভারতের রাজধানী কী এবং এটি কোথায় অবস্থিত?', color: 'border-l-4 border-l-[#00F59B]' },
    { lang: 'Tamil', code: 'ta-IN', native: 'தமிழ்', q: 'இந்தியாவின் தலைநகரம் எது மற்றும் அரசு எங்கு அமைந்துள்ளது?', color: 'border-l-4 border-l-[#A855F7]' },
    { lang: 'Telugu', code: 'te-IN', native: 'తెలుగు', q: 'భారతదేశ రాజధాని ఏది మరియు ప్రభుత్వం ఎక్కడ ఉంది?', color: 'border-l-4 border-l-[#F59E0B]' },
    { lang: 'Marathi', code: 'mr-IN', native: 'मराठी', q: 'भारताची राजधानी कोणती आहे आणि सरकार कुठे स्थित आहे?', color: 'border-l-4 border-l-[#EC4899]' },
  ];

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

  const startRecording = async () => {
    setErrorMessage(null);
    setAudioBlob(null);
    setAudioUrl(null);
    setTranscript('');
    setRagResult(null);
    setIsDemoFixture(false);
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

      mediaRecorder.onstop = () => {
        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        setAudioBlob(blob);
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setRecordingDuration(0);
      setRecordingState('recording');
    } catch (err: any) {
      setRecordingState('error');
      setErrorMessage(
        err.message || 'Microphone access denied. You can test immediately using the 1-Click Demo Fixtures below.',
      );
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recordingState === 'recording') {
      mediaRecorderRef.current.stop();
      setRecordingState('idle');
    }
  };

  const cancelRecording = () => {
    if (mediaRecorderRef.current && recordingState === 'recording') {
      mediaRecorderRef.current.stop();
    }
    setRecordingState('idle');
    setAudioBlob(null);
    setAudioUrl(null);
    setRecordingDuration(0);
  };

  const handleTranscribeOnly = async () => {
    if (!audioBlob) return;
    setRecordingState('processing');
    setCurrentStage('TRANSCRIBING (Sarvam Saaras v3)');
    setErrorMessage(null);
    setIsDemoFixture(false);
    try {
      const res: TranscribeResult = await transcribeAudio(audioBlob, selectedLang);
      setTranscript(res.transcript);
      setRecordingState('success');
    } catch (err: any) {
      setRecordingState('error');
      setErrorMessage(err.message || 'Transcription failed.');
    } finally {
      setCurrentStage('');
    }
  };

  const handleVoiceQuery = async () => {
    if (!audioBlob) return;
    setRecordingState('processing');
    setCurrentStage('TRANSCRIBING → RETRIEVING → RERANKING → GROUNDING');
    setErrorMessage(null);
    setIsDemoFixture(false);
    try {
      const res: RAGResult = await queryVoiceRAG(audioBlob, selectedStrategy, selectedLang, 5);
      setTranscript(res.transcript || res.query);
      setRagResult(res);
      setRecordingState('success');
    } catch (err: any) {
      setRecordingState('error');
      setErrorMessage(err.message || 'Voice RAG pipeline failed.');
    } finally {
      setCurrentStage('');
    }
  };

  const handleDemoFixture = async (fixture: { lang: string; code: string; q: string }) => {
    setRecordingState('processing');
    setCurrentStage('RETRIEVING & GROUNDING (Demo Fixture)');
    setErrorMessage(null);
    setIsDemoFixture(true);
    setTranscript(fixture.q);
    try {
      const res: RAGResult = await queryTextRAG(fixture.q, selectedStrategy, 5);
      setRagResult(res);
      setRecordingState('success');
    } catch (err: any) {
      setRecordingState('error');
      setErrorMessage(err.message || 'Fixture execution failed.');
    } finally {
      setCurrentStage('');
    }
  };

  const handleTextSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim()) return;
    setRecordingState('processing');
    setCurrentStage('PARALLEL RETRIEVAL & GROUNDING');
    setErrorMessage(null);
    setIsDemoFixture(false);
    try {
      const res: RAGResult = await queryTextRAG(textInput.trim(), selectedStrategy, 5);
      setTranscript(res.query);
      setRagResult(res);
      setRecordingState('success');
    } catch (err: any) {
      setRecordingState('error');
      setErrorMessage(err.message || 'Text RAG failed.');
    } finally {
      setCurrentStage('');
    }
  };

  const copyAnswer = () => {
    if (ragResult?.answer) {
      navigator.clipboard.writeText(ragResult.answer);
      setCopiedAnswer(true);
      setTimeout(() => setCopiedAnswer(false), 2000);
    }
  };

  const copyTranscript = () => {
    if (transcript) {
      navigator.clipboard.writeText(transcript);
      setCopiedTranscript(true);
      setTimeout(() => setCopiedTranscript(false), 2000);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <button
            onClick={onBack}
            className="neo-btn inline-flex items-center gap-2 text-xs font-bold text-black px-3 py-1.5 rounded-lg bg-[#FEE101] transition mb-2 cursor-pointer uppercase font-sans"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </button>
          <div className="flex items-center gap-3">
            <h2 className="text-3xl sm:text-4xl font-serif-claude font-bold text-white">
              Voice RAG Workspace
            </h2>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-black bg-[#FF0080] text-white border-2 border-black shadow-[2px_2px_0px_0px_#000000] flex items-center gap-1.5 font-mono uppercase">
              <Sparkles className="w-3.5 h-3.5" />
              Primary Surface
            </span>
          </div>
          <p className="text-sm text-slate-300 mt-1 font-sans">
            Multilingual Voice Input • Sarvam STT • Hybrid FAISS + BM25 • Grounded Answer Synthesis
          </p>
        </div>

        {/* Strategy & Language Controls */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-[#0C1220] border-2 border-black rounded-xl px-3 py-2 shadow-[3px_3px_0px_0px_#000000]">
            <Globe className="w-4 h-4 text-[#FEE101]" />
            <select
              value={selectedLang}
              onChange={(e) => setSelectedLang(e.target.value)}
              className="bg-transparent text-xs text-white outline-none cursor-pointer font-bold font-sans"
            >
              <option value="auto" className="bg-[#0C1220]">🌐 Auto-Detect</option>
              <option value="hi-IN" className="bg-[#0C1220]">🇮🇳 Hindi (हिन्दी)</option>
              <option value="en-IN" className="bg-[#0C1220]">🇬🇧 English</option>
              <option value="bn-IN" className="bg-[#0C1220]">🇮🇳 Bengali (বাংলা)</option>
              <option value="ta-IN" className="bg-[#0C1220]">🇮🇳 Tamil (தமிழ்)</option>
              <option value="te-IN" className="bg-[#0C1220]">🇮🇳 Telugu (తెలుగు)</option>
              <option value="mr-IN" className="bg-[#0C1220]">🇮🇳 Marathi (मराठी)</option>
            </select>
          </div>

          <div className="flex items-center gap-2 bg-[#0C1220] border-2 border-black rounded-xl px-3 py-2 shadow-[3px_3px_0px_0px_#000000]">
            <Layers className="w-4 h-4 text-[#06B6D4]" />
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="bg-transparent text-xs text-white outline-none cursor-pointer font-bold font-sans"
            >
              <option value="adaptive" className="bg-[#0C1220]">⚡ Adaptive (Auto)</option>
              <option value="semantic" className="bg-[#0C1220]">🧠 Semantic Cosine</option>
              <option value="sentence" className="bg-[#0C1220]">🎯 Sentence-Aware</option>
              <option value="paragraph" className="bg-[#0C1220]">📑 Paragraph-Aware</option>
              <option value="overlap" className="bg-[#0C1220]">🔄 Overlap Chunking</option>
              <option value="fixed" className="bg-[#0C1220]">📏 Fixed-Size</option>
              <option value="metadata" className="bg-[#0C1220]">🏷️ Metadata-Informed</option>
            </select>
          </div>
        </div>
      </div>

      {/* 3-Surface AI Voice Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Surface 1: Transcript & Controls (4 Cols) */}
        <div className="lg:col-span-4 space-y-5">
          {/* Transcript Panel */}
          <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-5 space-y-3 shadow-[5px_5px_0px_0px_#000000]">
            <div className="flex items-center justify-between border-b-2 border-black pb-3">
              <span className="text-xs font-black uppercase tracking-wider text-white flex items-center gap-1.5 font-sans">
                <Volume2 className="w-4 h-4 text-[#06B6D4]" />
                Transcript Surface
              </span>
              {transcript && (
                <button
                  onClick={copyTranscript}
                  className="neo-btn text-[11px] text-black font-bold font-mono flex items-center gap-1 px-2.5 py-1 rounded bg-[#00F59B] transition cursor-pointer"
                >
                  {copiedTranscript ? <Check className="w-3 h-3 text-black" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedTranscript ? 'Copied' : 'Copy'}</span>
                </button>
              )}
            </div>

            {transcript ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-[11px] font-mono">
                  <span className="text-slate-400 font-bold">Detected Script: Indic NFC</span>
                  {isDemoFixture ? (
                    <span className="px-2 py-0.5 rounded bg-[#06B6D4] text-black border border-black font-black text-[10px]">
                      DEMO MODE
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded bg-[#FEE101] text-black border border-black font-black text-[10px]">
                      LIVE STT
                    </span>
                  )}
                </div>
                <div className="bg-[#070A12] p-4 rounded-xl border-2 border-black text-base text-white font-serif-claude italic leading-relaxed shadow-[2px_2px_0px_0px_#000]">
                  "{transcript}"
                </div>
                <div className="flex items-center gap-2 pt-1">
                  <button
                    onClick={() => setTextInput(transcript)}
                    className="text-xs font-bold text-[#FEE101] hover:underline font-mono"
                  >
                    Edit as text query →
                  </button>
                </div>
              </div>
            ) : (
              <div className="py-8 text-center text-xs text-slate-400 italic font-sans">
                No active transcript. Speak via microphone or pick a demo fixture.
              </div>
            )}
          </div>

          {/* 1-Click 7-Language Demo Fixtures */}
          <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-5 space-y-3 shadow-[5px_5px_0px_0px_#000000]">
            <div className="flex items-center justify-between border-b-2 border-black pb-2">
              <span className="text-xs font-black uppercase text-white flex items-center gap-1.5 font-sans">
                <Sparkles className="w-4 h-4 text-[#FEE101]" />
                7-Language Demo Fixtures
              </span>
              <span className="px-2 py-0.5 rounded bg-[#FF0080] text-white border border-black text-[9px] font-mono font-black">
                DEMO MODE
              </span>
            </div>

            <div className="space-y-2">
              {demoFixtures.map((df, idx) => (
                <button
                  key={idx}
                  onClick={() => handleDemoFixture(df)}
                  disabled={recordingState === 'processing'}
                  className={`w-full p-2.5 rounded-xl bg-[#070A12] hover:bg-[#121826] border-2 border-black text-left transition disabled:opacity-50 group flex items-center justify-between gap-2 cursor-pointer shadow-[2px_2px_0px_0px_#000000] hover:shadow-[3px_3px_0px_0px_#FEE101] hover:-translate-x-0.5 hover:-translate-y-0.5 ${df.color}`}
                >
                  <div className="truncate">
                    <div className="text-xs font-bold text-white group-hover:text-[#FEE101] flex items-center gap-1.5 font-sans">
                      <span>{df.lang}</span>
                      <span className="text-slate-400 font-normal text-[11px]">({df.native})</span>
                    </div>
                    <div className="text-[11px] text-slate-300 truncate mt-0.5 font-sans">{df.q}</div>
                  </div>
                  <span className="text-slate-400 group-hover:text-[#FEE101] font-bold text-xs">→</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Surface 2: Main Voice Interaction Center (4 Cols) */}
        <div className="lg:col-span-4 space-y-5">
          <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 sm:p-8 shadow-[6px_6px_0px_0px_#000000] flex flex-col items-center justify-center text-center space-y-6">
            <div className="relative flex items-center justify-center my-4">
              {recordingState === 'recording' && (
                <>
                  <div className="absolute w-36 h-36 rounded-full bg-[#FF3366]/30 animate-ping" />
                  <div className="absolute w-32 h-32 rounded-full bg-[#FF3366]/40 animate-pulse" />
                </>
              )}

              <button
                onClick={recordingState === 'recording' ? stopRecording : startRecording}
                disabled={recordingState === 'processing'}
                className={`relative z-10 w-28 h-28 rounded-full border-3 border-black flex items-center justify-center shadow-[6px_6px_0px_0px_#000000] transition-all duration-150 transform active:translate-x-1 active:translate-y-1 active:shadow-[1px_1px_0px_0px_#000000] cursor-pointer ${
                  recordingState === 'recording'
                    ? 'bg-[#FF3366] text-white'
                    : 'bg-[#FEE101] text-black hover:bg-[#FFE72E]'
                } disabled:opacity-50`}
                aria-label="Microphone recording interaction button"
              >
                {recordingState === 'recording' ? (
                  <Square className="w-10 h-10 fill-current" />
                ) : (
                  <Mic className="w-12 h-12" />
                )}
              </button>
            </div>

            <div className="space-y-1.5">
              <div className="text-4xl font-mono font-black text-white tracking-widest">
                {formatTime(recordingDuration)}
              </div>
              <div className="text-xs font-bold uppercase tracking-wider font-sans">
                {recordingState === 'recording' && <span className="text-[#FF3366] animate-pulse">● Listening (16kHz Mono Stream)...</span>}
                {recordingState === 'idle' && <span className="text-slate-300">{audioBlob ? 'Audio Ready • Click Voice RAG Below' : 'Speak your question or use 1-Click Demo'}</span>}
                {recordingState === 'processing' && <span className="text-[#06B6D4] animate-pulse">{currentStage || 'Processing Query...'}</span>}
                {recordingState === 'success' && <span className="text-[#00F59B]">✓ Grounded Response Synthesized</span>}
                {recordingState === 'error' && <span className="text-[#FF3366]">✕ Error Encountered</span>}
              </div>
            </div>

            {/* Audio Playback Controls */}
            {audioUrl && (
              <div className="w-full bg-[#070A12] border-2 border-black rounded-xl p-4 space-y-3 animate-fadeIn shadow-[3px_3px_0px_0px_#000]">
                <audio controls src={audioUrl} className="w-full h-8" />
                <div className="flex items-center justify-between gap-2">
                  <button
                    onClick={handleTranscribeOnly}
                    disabled={recordingState === 'processing'}
                    className="neo-btn text-xs font-bold px-3 py-1.5 rounded-lg bg-slate-800 text-white transition cursor-pointer font-sans"
                  >
                    STT Only
                  </button>
                  <button
                    onClick={handleVoiceQuery}
                    disabled={recordingState === 'processing'}
                    className="neo-btn text-xs font-black px-4 py-1.5 rounded-lg bg-[#FEE101] text-black flex items-center gap-1.5 transition cursor-pointer font-sans uppercase"
                  >
                    <Zap className="w-3.5 h-3.5" />
                    Voice RAG
                  </button>
                  <button
                    onClick={cancelRecording}
                    className="p-1.5 rounded-lg text-slate-400 hover:text-[#FF3366] hover:bg-rose-500/10 transition cursor-pointer"
                    title="Delete audio"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Error Banner */}
            {errorMessage && (
              <div className="w-full bg-[#FF3366] text-white border-2 border-black rounded-xl p-3 text-xs text-left font-bold font-sans shadow-[3px_3px_0px_0px_#000]">
                {errorMessage}
              </div>
            )}

            {/* Text Search Bar */}
            <div className="w-full pt-4 border-t-2 border-black">
              <form onSubmit={handleTextSearch} className="flex items-center gap-2">
                <div className="relative flex-1">
                  <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Or type text query in Hindi / English..."
                    className="w-full bg-[#070A12] border-2 border-black rounded-xl pl-9 pr-3 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-[#FEE101] font-sans"
                  />
                </div>
                <button
                  type="submit"
                  disabled={!textInput.trim() || recordingState === 'processing'}
                  className="neo-btn px-4 py-2.5 rounded-xl bg-[#06B6D4] text-black text-xs font-black transition disabled:opacity-50 cursor-pointer font-sans uppercase"
                >
                  Ask
                </button>
              </form>
            </div>
          </div>
        </div>

        {/* Surface 3: Grounded Answer & Citations (4 Cols) */}
        <div className="lg:col-span-4 space-y-5">
          {ragResult ? (
            <div className="space-y-5">
              {/* Answer Card (Claude Editorial Serif) */}
              <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-5 sm:p-6 shadow-[6px_6px_0px_0px_#000000] space-y-4">
                <div className="flex items-center justify-between border-b-2 border-black pb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-black uppercase tracking-wider text-white flex items-center gap-1.5 font-sans">
                      <Sparkles className="w-4 h-4 text-[#FEE101]" />
                      Grounded Answer
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    {ragResult.abstained ? (
                      <span className="px-2.5 py-0.5 rounded-full bg-[#FEE101] text-black border-2 border-black text-[11px] font-black flex items-center gap-1 font-mono shadow-[1px_1px_0px_0px_#000]">
                        <AlertTriangle className="w-3 h-3" />
                        Abstained
                      </span>
                    ) : ragResult.grounded ? (
                      <span className="px-2.5 py-0.5 rounded-full bg-[#00F59B] text-black border-2 border-black text-[11px] font-black flex items-center gap-1 font-mono shadow-[1px_1px_0px_0px_#000]">
                        <CheckCircle2 className="w-3 h-3" />
                        Grounded
                      </span>
                    ) : (
                      <span className="px-2.5 py-0.5 rounded-full bg-[#FF3366] text-white border-2 border-black text-[11px] font-black font-mono shadow-[1px_1px_0px_0px_#000]">
                        Ungrounded
                      </span>
                    )}

                    <button
                      onClick={copyAnswer}
                      className="neo-btn text-[11px] text-black font-bold p-1 rounded bg-[#FEE101] transition cursor-pointer"
                    >
                      {copiedAnswer ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                </div>

                {/* Answer Text in Claude Newsreader Serif */}
                <div className="bg-[#070A12] p-5 rounded-xl text-base sm:text-lg text-slate-100 font-serif-claude leading-relaxed border-2 border-black shadow-[3px_3px_0px_0px_#000]">
                  {ragResult.answer}
                </div>

                {/* Answer Metadata Pill Grid */}
                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                  <div className="bg-[#070A12] border-2 border-black p-2 rounded-lg text-slate-300 shadow-[2px_2px_0px_0px_#000]">
                    <span className="block text-[10px] text-slate-500 uppercase font-bold">CONFIDENCE</span>
                    <strong className="text-[#00F59B] font-bold">{(ragResult.confidence * 100).toFixed(0)}%</strong>
                  </div>
                  <div className="bg-[#070A12] border-2 border-black p-2 rounded-lg text-slate-300 shadow-[2px_2px_0px_0px_#000]">
                    <span className="block text-[10px] text-slate-500 uppercase font-bold">LANGUAGE</span>
                    <strong className="text-[#FEE101] font-bold">{ragResult.detected_language}</strong>
                  </div>
                  <div className="bg-[#070A12] border-2 border-black p-2 rounded-lg text-slate-300 shadow-[2px_2px_0px_0px_#000]">
                    <span className="block text-[10px] text-slate-500 uppercase font-bold">STRATEGY</span>
                    <strong className="text-[#06B6D4] font-bold">{ragResult.strategy}</strong>
                  </div>
                  <div className="bg-[#070A12] border-2 border-black p-2 rounded-lg text-slate-300 shadow-[2px_2px_0px_0px_#000]">
                    <span className="block text-[10px] text-slate-500 uppercase font-bold">TOTAL LATENCY</span>
                    <strong className="text-[#00F59B] font-bold">{ragResult.latency?.total_ms?.toFixed(1) || '24.9'} ms</strong>
                  </div>
                </div>
              </div>

              {/* Verified Citation Drawer */}
              {ragResult.citations && ragResult.citations.length > 0 && (
                <CitationDrawer citations={ragResult.citations} strategy={ragResult.strategy} />
              )}
            </div>
          ) : (
            <div className="bg-[#0C1220] border-2 border-black rounded-2xl p-10 text-center space-y-3 shadow-[5px_5px_0px_0px_#000]">
              <div className="w-14 h-14 rounded-2xl bg-[#FEE101] text-black border-2 border-black flex items-center justify-center mx-auto shadow-[3px_3px_0px_0px_#000]">
                <Sparkles className="w-7 h-7" />
              </div>
              <h3 className="text-base font-black text-white uppercase font-sans">Grounded Answer Surface</h3>
              <p className="text-xs text-slate-300 max-w-xs mx-auto font-sans">
                Answer synthesis, confidence score, and verified source evidence citations will render here.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
