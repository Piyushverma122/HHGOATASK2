import React from 'react';
import { Globe } from 'lucide-react';

interface MultilingualSectionProps {
  onSelectFixture?: (langCode: string) => void;
}

export const MultilingualSection: React.FC<MultilingualSectionProps> = ({ onSelectFixture }) => {
  const languages = [
    { name: 'Hindi', native: 'हिन्दी', code: 'hi-IN', sample: 'भारत की राजधानी क्या है?', flag: '🇮🇳', shadow: 'hover:shadow-[4px_4px_0px_0px_#FEE101]' },
    { name: 'English', native: 'English', code: 'en-IN', sample: 'What is the capital of India?', flag: '🇬🇧', shadow: 'hover:shadow-[4px_4px_0px_0px_#06B6D4]' },
    { name: 'Hinglish', native: 'Hinglish', code: 'hi-IN', sample: 'India ki capital kya hai?', flag: '🇮🇳', shadow: 'hover:shadow-[4px_4px_0px_0px_#FF0080]' },
    { name: 'Bengali', native: 'বাংলা', code: 'bn-IN', sample: 'ভারতের রাজধানী কী?', flag: '🇮🇳', shadow: 'hover:shadow-[4px_4px_0px_0px_#00F59B]' },
    { name: 'Tamil', native: 'தமிழ்', code: 'ta-IN', sample: 'இந்தியாவின் தலைநகரம் எது?', flag: '🇮🇳', shadow: 'hover:shadow-[4px_4px_0px_0px_#A855F7]' },
    { name: 'Telugu', native: 'తెలుగు', code: 'te-IN', sample: 'భారతదేశ రాజధాని ఏది?', flag: '🇮🇳', shadow: 'hover:shadow-[4px_4px_0px_0px_#F59E0B]' },
    { name: 'Marathi', native: 'मराठी', code: 'mr-IN', sample: 'भारताची राजधानी कोणती?', flag: '🇮🇳', shadow: 'hover:shadow-[4px_4px_0px_0px_#EC4899]' },
  ];

  return (
    <section className="bg-[#0C1220] border-2 border-black rounded-2xl p-6 shadow-[5px_5px_0px_0px_#000000] space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b-2 border-black pb-3">
        <div className="flex items-center gap-2">
          <Globe className="w-5 h-5 text-[#FEE101]" />
          <h2 className="text-lg font-black text-white uppercase font-sans">
            7 Supported Indic & Multilingual Scripts
          </h2>
        </div>
        <span className="text-xs text-slate-300 font-mono">
          Sarvam Saaras v3 + E5-Small Tokenizer
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        {languages.map((lang, idx) => (
          <div
            key={idx}
            onClick={() => onSelectFixture?.(lang.code)}
            className={`bg-[#070A12] border-2 border-black rounded-xl p-3.5 space-y-1.5 transition-all cursor-pointer group flex flex-col justify-between shadow-[3px_3px_0px_0px_#000000] ${lang.shadow} hover:-translate-x-0.5 hover:-translate-y-0.5`}
          >
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-white group-hover:text-[#FEE101] font-sans">
                {lang.name}
              </span>
              <span className="text-base">{lang.flag}</span>
            </div>
            <div className="text-lg font-black text-[#FEE101] font-sans tracking-wide">
              {lang.native}
            </div>
            <div className="text-[10px] text-slate-400 truncate font-mono">
              {lang.sample}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
